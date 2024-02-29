import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plot
import keras
from keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

from sklearn.ensemble import RandomForestClassifier

# Reshape Data
if len(x_train.shape) == 3:  # grayscale images
    height, width = x_train.shape[1:]
    x_train = x_train.reshape((x_train.shape[0], height * width))
    x_test = x_test.reshape((x_test.shape[0], height * width))
elif len(x_train.shape) == 4:  # color images
    height, width, channels = x_train.shape
    x_train = x_train.reshape((x_train.shape[0], height * width * channels))
    x_test = x_test.reshape((x_test.shape[0], height * width * channels))

# Create model
rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train, y_train)

# Predictions
y_pred_train = rf.predict(x_train)
y_pred_test = rf. predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Metrics
print('Training Data')
print(classification_report(y_train, y_pred_train))
cm_train = confusion_matrix(y_train, y_pred_train)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
disp_train.plot()
plot.show()
plot.savefig('rf_train.png')

print('Testing Data')
print(classification_report(y_test, y_pred_test))
cm_test = confusion_matrix(y_test, y_pred_test)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp_test.plot()
plot.show()
plot.savefig('rf_test.png')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Plotting Data
classes = ['0','1', '2', '3', '4',
           '5', '6', '7', '8', '9']

print('Y VALUES\n')
for i in range(9):
  img = x_train[i]
  plot.subplot(330+1+i)
  plot.imshow(img)
  print(classes[y_train[i]])

print('\nX VALUES')
plot.show()
plot.savefig('examine.png')

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Create Validation Set
(x_train, x_val) = x_train[5000:], x_train[:5000]
(y_train, y_val) = y_train[5000:], y_train[:5000]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

from keras.preprocessing.image import ImageDataGenerator

# Image Generator
datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(x_train)


model = keras.Sequential(
    [
        # Idea 1: ~47% accuracy
        # keras.Input(shape=input_shape),
        # layers.Conv2D(32, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.Conv2D(64, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.BatchNormalization(),
        # layers.Dropout(0.4),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(32, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.Conv2D(16, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.BatchNormalization(),
        # layers.Dropout(0.3),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(8, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.Conv2D(4, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.BatchNormalization(),
        # layers.Dropout(0.2),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(2, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.Conv2D(1, kernel_size=(3, 3), padding='SAME', activation="relu"),
        # layers.BatchNormalization(),
        # layers.Dropout(0.1),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Flatten(),
        # layers.Dropout(0.5),
        # layers.Dense(64, activation="relu"),
        # layers.Dense(32, activation="relu"),
        # layers.Dense(16, activation="relu"),
        # layers.Dense(num_classes, activation="softmax"),

        #Idea 2
        # keras.Input(shape=input_shape),
        # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Flatten(),
        # layers.Dropout(0.5),
        # layers.Dense(num_classes, activation="softmax"),

        #Idea 3
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), padding='SAME', activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), padding='SAME', activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), padding='SAME', activation="relu"),
        layers.Conv2D(16, kernel_size=(3, 3), padding='SAME', activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

print(model.summary())

from keras.optimizers import RMSprop

# compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

"""### Training and Validation"""

# train model
history = model.fit(x_train, y_train, batch_size=50, epochs=100,
                    validation_data=(x_val, y_val))

# look at model history
print(history.history)

# Plot Loss
f, ax = plot.subplots(1, 2, figsize=(12,3))
ax[0].plot(history.history['loss'], label='Loss', color='r')
ax[0].plot(history.history['val_loss'], label='Validation', color='b')
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Accuracy
ax[1].plot(history.history['accuracy'], label='Accuracy', color='r')
ax[1].plot(history.history['val_accuracy'], label='Validation', color='b')
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plot.tight_layout()
plot.show()
plot.savefig('model.png')

# evaluate data
results = model.evaluate(x_test, y_test, batch_size=100)
print('Test Loss, Test Accuracy: ' , results)

# Generate predictions
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)
