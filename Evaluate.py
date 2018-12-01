import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import backend as K
from sklearn.metrics import roc_curve
K.set_image_dim_ordering('th')

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'temp/Training',
        color_mode="grayscale",
        target_size=(48, 48),
        batch_size=8,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'temp/PublicTest',
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=1,
        class_mode='categorical')

with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('weights.h5')

# print(model)
sgd = optimizers.SGD(lr=0.0, momentum=0.0)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit_generator(train_generator, epochs = 1, steps_per_epoch=1, shuffle=True)

loss, accuracy = model.evaluate_generator(test_generator, 3500)

print(accuracy)