# ফুলের ছবি চেনার জন্য মডেল ট্রেনিং - train_model.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ডেটাসেট পাথ
DATASET_PATH = 'flowers/'

# ইমেজ সাইজ
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

# ডেটা প্রসেসিং
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# মডেল বানানো
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# মডেল কম্পাইল
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ট্রেনিং
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# মডেল সংরক্ষণ
os.makedirs('model', exist_ok=True)
model.save('model/flower_model.h5')

print("\u2705 মডেল ট্রেনিং শেষ এবং সংরক্ষিত হয়েছে!")
