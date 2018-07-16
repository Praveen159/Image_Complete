# -*- coding: utf-8 -*-
import urllib
from IPython.display import Image, display, clear_output
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style('whitegrid')

import os
import h5py
import numpy as np
import pandas as pd
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History

location = 'C:/Users/hi'
top_model_weights_path=location+'/top_model_weights.h5' # will be saved into when we create our model
# model_path = location + '/initial_data2_model.h5'
fine_tuned_model_path = location+'/ft_model.h5'

# dimensions of our images
img_width, img_height = 256,256

train_data_dir = 'D:/DS/Train_Images'
validation_data_dir = 'D:/DS/Val_Images'

train_samples = [len(os.listdir(train_data_dir+'/'+i)) for i in sorted(os.listdir(train_data_dir))]
nb_train_samples = sum(train_samples)
validation_samples = [len(os.listdir(validation_data_dir+'/'+i)) for i in sorted(os.listdir(validation_data_dir))]
nb_validation_samples = sum(validation_samples)

size_batch=5
nb_epoch = 5

model1 = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width,img_height,3))

train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=size_batch, 
                                            class_mode=None, 
                                            shuffle=False) 
    
bottleneck_features_train = model1.predict_generator(train_generator, nb_train_samples//size_batch)
np.save(open(location+'/bottleneck_features_train5', 'wb'), bottleneck_features_train)
    
    # repeat with the validation data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(validation_data_dir,
                                           target_size=(img_width, img_height),
                                           batch_size=3,
                                           class_mode=None,
                                           shuffle=False)
bottleneck_features_validation = model1.predict_generator(test_generator, nb_validation_samples//size_batch)
np.save(open(location+'/bottleneck_features_validation5', 'wb'), bottleneck_features_validation)


train_data = np.load(open('C:/Users/hi/POC/bottleneck_features_train1.npy',"rb"))
train_labels = np.array([0] * train_samples[0] + 
                            [1] * 217)

validation_data = np.load(open(location+'/bottleneck_features_validation1.npy',"rb"))
validation_labels = np.array([0] * 2 + 
                                 [1] * 1)
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:])) # 512, 4, 4
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5)) 
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizers.SGD(lr=0.0001, momentum=0.9),
              loss='binary_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(top_model_weights_path, monitor='val_acc',verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
fit = model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=16,
              validation_data=(validation_data, validation_labels),callbacks=[checkpoint])


urllib.request.urlretrieve('https://www.carfax.com/media/zoo/images/rsz_frame-damage_85730e0a843d155e25e4b0f0e100bf65.jpg', 'save.jpg') # or other way to upload image
img = load_img('save.jpg', target_size=(img_width, img_height)) # this is a PIL image 
x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)
pred = model1.predict(x)
preds = model.predict(x)
print ("Validating that damage exists...")
print (preds)
if preds[0][0] <=.5:
    print ("Validation complete - proceed to location and severity determination")
else:
    print ("Are you sure that your car is damaged? Please submit another picture of the damage.")
    
        
        
