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
fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()  # split into tetsing and training

# Let's have a look at this data to see what we are working with.
train_images.shape

# So we've got 60,000 images that are made up of 28x28 pixels (784 in total).
# let's have a look at one pixel
train_images[0, 23, 23]

# let's have a look at the first 10 training labels
train_labels[:10]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
