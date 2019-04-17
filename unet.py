# import libraries
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Conv2DTranspose, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l1, l2

def dice_coeff(true, pred):
    smooth = 1.
    true_flat = K.flatten(true)
    pred_flat = K.flatten(pred)
    intersection = K.sum(true_flat * pred_flat)
    score = (2. * intersection + smooth) / (K.sum(true_flat) + K.sum(pred_flat) + smooth)
    return score

def dice_loss(true, pred):
    loss = 1 - dice_coeff(true, pred)
    return loss

def unet_base(input_shape=(128,128,3),
              num_classes=3, 
              first_filters=16):
    
    inputs = Input(shape=input_shape)
    
    down1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1_pool = MaxPooling2D((2,2), strides=(2,2))(down1)
    print(down1.shape)
    
    down2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2_pool = MaxPooling2D((2,2), strides=(2,2))(down2)
    print(down2.shape)
    
    down3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3_pool = MaxPooling2D((2,2), strides=(2,2))(down3)
    print(down3.shape)
    
    center = Conv2D(first_filters*8, (3,3), activation='relu', padding='same')(down3_pool)
    center = BatchNormalization()(center)
    center = Conv2D(first_filters*8, (3,3), activation='relu', padding='same')(center)
    center = BatchNormalization()(center)
    print(center.shape)
    
    # up3 = UpSampling2D((2,2))(center)
    up3 = Conv2DTranspose(first_filters*4, (2,2), activation='relu', strides=(2,2), padding='same')(center)
    up3 = concatenate([up3, down3], axis=3)
    up3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    
    # up2 = UpSampling2D((2,2))(up3)
    up2 = Conv2DTranspose(first_filters*2, (2,2), activation='relu', strides=(2,2), padding='same')(up3)
    up2 = concatenate([up2, down2], axis=3)
    up2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    
    # up1 = UpSampling2D((2,2))(up2)
    up1 = Conv2DTranspose(first_filters, (2,2), activation='relu', strides=(2,2), padding='same')(up2)
    up1 = concatenate([up1, down1], axis=3)
    up1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    
    model = Model(inputs=inputs, outputs=classify)
    
    model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])
    
    return model

def unet_dropout(input_shape=(128,128,3),
              num_classes=3, 
              first_filters=16):
    
    inputs = Input(shape=input_shape)
    
    down1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1_pool = MaxPooling2D((2,2), strides=(2,2))(down1)
    down1_pool = Dropout(0.25)(down1_pool)
    
    down2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2_pool = MaxPooling2D((2,2), strides=(2,2))(down2)
    down2_pool = Dropout(0.25)(down2_pool)
    
    down3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3_pool = MaxPooling2D((2,2), strides=(2,2))(down3)
    down3_pool = Dropout(0.25)(down3_pool)
    
    center = Conv2D(first_filters*8, (3,3), activation='relu', padding='same')(down3_pool)
    center = BatchNormalization()(center)
    center = Conv2D(first_filters*8, (3,3), activation='relu', padding='same')(center)
    center = BatchNormalization()(center)
    
    # up3 = UpSampling2D((2,2))(center)
    up3 = Conv2DTranspose(first_filters*4, (2,2), activation='relu', strides=(2,2), padding='same')(center)
    up3 = concatenate([up3, down3], axis=3)
    up3 = Dropout(0.25)(up3)
    up3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    
    # up2 = UpSampling2D((2,2))(up3)
    up2 = Conv2DTranspose(first_filters*2, (2,2), activation='relu', strides=(2,2), padding='same')(up3)
    up2 = concatenate([up2, down2], axis=3)
    up2 = Dropout(0.25)(up2)
    up2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    
    # up1 = UpSampling2D((2,2))(up2)
    up1 = Conv2DTranspose(first_filters, (2,2), activation='relu', strides=(2,2), padding='same')(up2)
    up1 = concatenate([up1, down1], axis=3)
    up2 = Dropout(0.25)(up2)
    up1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(first_filters, (3,3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    
    model = Model(inputs=inputs, outputs=classify)
    
    model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])
    
    return model

def unet_wreg(input_shape=(128,128,3),
              num_classes=3, 
              first_filters=16):
    
    inputs = Input(shape=input_shape)
    
    down1 = Conv2D(first_filters, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Conv2D(first_filters, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(down1)
    down1 = BatchNormalization()(down1)
    down1_pool = MaxPooling2D((2,2), strides=(2,2))(down1)
    down1_pool = Dropout(0.25)(down1_pool)
    
    down2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(down2)
    down2 = BatchNormalization()(down2)
    down2_pool = MaxPooling2D((2,2), strides=(2,2))(down2)
    down2_pool = Dropout(0.25)(down2_pool)
    
    down3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(down3)
    down3 = BatchNormalization()(down3)
    down3_pool = MaxPooling2D((2,2), strides=(2,2))(down3)
    down3_pool = Dropout(0.25)(down3_pool)
    
    center = Conv2D(first_filters*8, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(down3_pool)
    center = BatchNormalization()(center)
    center = Conv2D(first_filters*8, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(center)
    center = BatchNormalization()(center)
    
    # up3 = UpSampling2D((2,2))(center)
    up3 = Conv2DTranspose(first_filters*4, (2,2), activation='relu', strides=(2,2), padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(center)
    up3 = concatenate([up3, down3], axis=3)
    up3 = Dropout(0.25)(up3)
    up3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(first_filters*4, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up3)
    up3 = BatchNormalization()(up3)
    
    # up2 = UpSampling2D((2,2))(up3)
    up2 = Conv2DTranspose(first_filters*2, (2,2), activation='relu', strides=(2,2), padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up3)
    up2 = concatenate([up2, down2], axis=3)
    up2 = Dropout(0.25)(up2)
    up2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(first_filters*2, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up2)
    up2 = BatchNormalization()(up2)
    
    # up1 = UpSampling2D((2,2))(up2)
    up1 = Conv2DTranspose(first_filters, (2,2), activation='relu', strides=(2,2), padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up2)
    up1 = concatenate([up1, down1], axis=3)
    up2 = Dropout(0.25)(up2)
    up1 = Conv2D(first_filters, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(first_filters, (3,3), activation='relu', padding='same', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up1)
    up1 = BatchNormalization()(up1)
    
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', kernal_regularizer=l2(0.01), activity_regularizer=l1(0.01))(up1)
    
    model = Model(inputs=inputs, outputs=classify)
    
    model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])
    
    return model