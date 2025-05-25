import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# ======= CẤU HÌNH ============
image_size = 224
batch_size = 32
epochs = 50
data_dir = 'data'  # <-- Thư mục này cần có các sub-folder theo class
model_save_path = 'models/mymodel.h5'

# ======= KIỂM TRA THƯ MỤC ============
if not os.path.exists(data_dir):
    raise Exception(f"❌ Thư mục {data_dir} không tồn tại!")

# ======= IMAGE GENERATOR ============
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Số lượng ảnh train:", train_generator.samples)
print("Số lượng ảnh validation:", val_generator.samples)
print("Nhãn phân loại:", train_generator.class_indices)

# ======= MÔ HÌNH ============
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # tự động theo số class

model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các lớp VGG16 gốc
for layer in base_model.layers:
    layer.trainable = False

# ======= TRAIN ============
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping]
)

# ======= LƯU MODEL ============
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print("✅ Đã huấn luyện xong và lưu model tại:", model_save_path)
