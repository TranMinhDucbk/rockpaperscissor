import os
os.environ["KERAS_BACKEND"] = "torch"      # phải đặt TRƯỚC import keras

import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "D:/ROCKPAPERSCISSOR/rps/"
training_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest')

VALIDATION_DIR = "D:/ROCKPAPERSCISSOR/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR, target_size=(150, 150), class_mode='categorical', batch_size=126)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=(150, 150), class_mode='categorical', batch_size=126)

model = keras.models.Sequential([
    keras.layers.Input(shape=(150, 150, 3)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=5, steps_per_epoch=20,
                    validation_data=validation_generator, verbose=1, validation_steps=3)

model.save("rps.keras")