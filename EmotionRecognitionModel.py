# from google.colab import drive
# drive.mount('/content/drive')
from keras import models
from keras import layers
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, BatchNormalization, Dropout
import numpy as np
from keras import optimizers
from keras import backend as K
K.set_image_dim_ordering('th')


train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'temp/Training',
        color_mode="grayscale",
        target_size=(48, 48),
        batch_size=8,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'temp/PrivateTest',
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=8,
        class_mode='categorical')

#print(len(train_generator))
#print(len(train_generator[0][0][0][0]))


model = Sequential()
model.add(Conv2D(filters = 16, kernel_size= (3,3), activation= 'relu', input_shape= (1, 48, 48)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(Conv2D(filters = 32, kernel_size= (3,3)))
model.add(Activation('relu'))

model.add(Conv2D(filters = 64, kernel_size= (3,3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(7, activation='softmax'))
#model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])
#model.add(BatchNormalization())

# Output Layer
#model.add(Dense(7))
#model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#  metrics=['accuracy'])
# model.summary()

# input_shape = (1,64,64)
# model = Sequential([
#     Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
#            activation='relu'),
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     Conv2D(128, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Flatten(),
#     Dense(4096, activation='relu'),
#     Dense(4096, activation='relu'),
#     Dense(7, activation='softmax')
# ])

# vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(1, 64, 64))
# model = models.Sequential()
#
# # Add the vgg convolutional base model
# model.add(vgg_conv)
#
# # Add new layers
# model.add(layers.Flatten())
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(7, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
#model.summary()




#sgd = optimizers.SGD(lr=0.00001, decay=0.0, momentum=0.0, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.summary()
model.fit_generator(train_generator, epochs = 150, steps_per_epoch=3500, shuffle=True)
model.save_weights('weights.h5')

# loss, accuracy = model.evaluate(validation_generator, verbose = 0)
# print(loss)
# print(accuracy)

#print(model.summary())