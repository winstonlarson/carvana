# runs on the test set

import numpy as np
import tensorflow as tf
from tensorflow import keras
from unet import unet_base
from keras.preprocessing.image import ImageDataGenerator

import os

img_width = 128
img_height = 128
batch_size = 16

model = unet_base()
model.load_weights(filepath='weights/best_weights.h5')

dataset_dir = '/home/ubuntu/carvana/input/'
test_dir = dataset_dir + 'test/'
test_image_dir = test_dir + 'images/'
test_mask_dir = test_dir + 'masks/'

test_datagen = ImageDataGenerator(rescale=1./255)

test_image_generator = test_datagen.flow_from_directory(
    test_image_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode=None)

test_mask_generator = test_datagen.flow_from_directory(
    test_mask_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode=None)

test_generator = zip(test_image_generator, test_mask_generator)

num_samples_test = test_image_generator.n

test_loss, test_acc = model.evaluate_generator(test_generator, steps=np.ceil(num_samples_test/batch_size))
print('test acc:', test_acc)