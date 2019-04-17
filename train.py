# pulls everything together for training

# import libraries
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import os
import matplotlib.pyplot as plt

from unet import unet_base, unet_dropout

# define directories
dataset_dir = '/home/ubuntu/carvana/input/'
train_dir = dataset_dir + 'train/'
test_dir = dataset_dir + 'test/'

train_image_dir = train_dir + 'images/'
train_mask_dir = train_dir + 'masks/'

# training parameters
img_width = 128
img_height = 128
batch_size = 16
epochs = 80

# Define data augmentations for training set
data_gen_args = dict(rescale=1./255,
                     shear_range=0.1,
                     rotation_range=4,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     validation_split=0.2) # 20% validation set

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 42

# Create generator for training images
train_image_generator = image_datagen.flow_from_directory(
    train_image_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    seed=seed,
    subset='training')

# Create generator for training masks
train_mask_generator = mask_datagen.flow_from_directory(
    train_mask_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    seed=seed,
    subset='training')

# Create generator for validation images
val_image_generator = image_datagen.flow_from_directory(
    train_image_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    seed=seed,
    subset='validation')

# Create generator for validation masks
val_mask_generator = mask_datagen.flow_from_directory(
    train_mask_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    seed=seed,
    subset='validation')

num_samples_train = train_image_generator.n
num_samples_val = val_image_generator.n

# Combine generators into single training and validation generators for model training
train_generator = zip(train_image_generator, train_mask_generator)
validation_generator = zip(val_image_generator, val_mask_generator)

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               min_delta=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.h5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs/run3')]

model = unet_dropout()

model.fit_generator(generator=train_generator,
                    steps_per_epoch=np.ceil(num_samples_train/batch_size),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=np.ceil(num_samples_val/batch_size))