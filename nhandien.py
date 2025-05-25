import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import time

# ====== Cấu hình ======
gesture_names = {
    0: 'eight',
    1: 'five',
    2: 'four',
    3: 'nine',
    4: 'one',
    5: 'seven',
    6: 'six',
    7: 'ten',
    8: 'three',
    9: 'two'
}



datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

print(train_generator.class_indices)


image_size = 224
threshold = 60
blurValue = 41
bgSubThreshold = 50
learningRate = 0
predThreshold = 95

bgModel = None
isBgCaptured = 0
prediction = ''
score = 0

# ====== Load model đã huấn luyện ======
model = load_model('models/mymodel.h5')

# ====== Hàm dự đoán từ ảnh ======
def predict_image(img):
    img = img.astype('float32') / 255.0
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    conf = float(np.max(pred)) * 100
    return gesture_names[class_idx], conf

# ====== Hàm xóa nền ======
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# ====== Mở webcam ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

cap_region_x_begin = 0.5
cap_region_y_end = 0.8

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)

    x1 = int(cap_region_x_begin * frame.shape[1])
    y2 = int(cap_region_y_end * frame.shape[0])
    cv2.rectangle(frame, (x1, 0), (frame.shape[1], y2), (255, 0, 0), 2)

    if isBgCaptured == 1:
        img = remove_background(frame)
        roi = img[0:y2, x1:frame.shape[1]]

        # ====== Xử lý ảnh thành nhị phân ======
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Chuyển về đúng định dạng model yêu cầu (ảnh nhị phân 3 kênh)
        if thresh is not None:
            thresh_3ch = np.stack((thresh,)*3, axis=-1)  # từ 1 kênh sang 3 kênh
            input_img = cv2.resize(thresh_3ch, (image_size, image_size))
            input_img = input_img.reshape(1, image_size, image_size, 3)

            prediction, score = predict_image(input_img)

            if score >= predThreshold:
                cv2.putText(frame, f"{prediction} ({score:.1f}%)", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            else:
                cv2.putText(frame, "Confidence too low", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị ảnh nhị phân
        cv2.imshow("Thresh", cv2.resize(thresh, None, fx=0.5, fy=0.5))

    # Hiển thị khung chính
    cv2.imshow("Realtime Hand Sign Detection", cv2.resize(frame, None, fx=0.5, fy=0.5))

    # ====== Phím điều khiển ======
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print("✅ Background Captured")
        time.sleep(1)
    elif key == ord('r'):
        bgModel = None
        isBgCaptured = 0
        print("🔁 Background Reset")
        time.sleep(1)

cap.release()
cv2.destroyAllWindows()
