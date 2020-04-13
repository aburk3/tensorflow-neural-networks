# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# For this tutorial we will use the MNIST Fashion Dataset.
# This is a dataset that is included in keras.

# This dataset includes 60, 000 images for training
# and 10, 000 images for validation/testing.
# load dataset
# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images,
#                                test_labels) = fashion_mnist.load_data()  # split into tetsing and training

# Let's have a look at this data to see what we are working with.
# train_images.shape

# So we've got 60,000 images that are made up of 28x28 pixels (784 in total).
# let's have a look at one pixel
# train_images[0, 23, 23]

# let's have a look at the first 10 training labels
# train_labels[:10]

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# second test
fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    # hidden layer (2) - 128: neurons
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # output layer (3)
])

model.compile(optimizer='adam',  # algorithm performing the gradient descent
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# we pass the data, labels and epochs and watch the magic!
model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
