from __future__ import print_function
import os
import math
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import utils
from keras import backend as K
from keras.layers import Layer
from keras.layers import *
from keras import activations
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from collections import Counter

angle = 45
angle_2 = 90
angle_3 = 180
angle_4 = 270
input_size = 139
TEST_SIZE = 0.25
epochs = 100
BATCH_SIZE = 128
num_classes = 7


# load data
x = []
y = []

csv_input = pd.read_csv("HAM10000_metadata.csv")
classtonumber_dict = {"akiec":0, "bcc":1, "bkl":2, "df":3, "mel":4, "nv":5, "vasc":6}

if os.path.exists('np_save_' + str(input_size) + 'x.npy'):
  file_exists = 1
else:
  file_exists = 0

if file_exists == 0:
  for i in range(csv_input["image_id"].size):
    file = csv_input["image_id"][i]
    label = classtonumber_dict[csv_input["dx"][i]]
    file1 = 'HAM10000_images_part_1/' + file + '.jpg'
    file2 = 'HAM10000_images_part_2/' + file + '.jpg'
    if os.path.exists(file1):
        img = Image.open(file1)
    elif os.path.exists(file2):
        img = Image.open(file2)
    else:
        ("File doesn't exsist.")
        #continue

    img = img.resize((input_size, input_size))
    img = image.img_to_array(img)
    x.append(img)
    y.append(label)

  x = np.asarray(x)
  y = np.asarray(y)
  print(x.shape)

  np.save('np_save_' + str(input_size) + 'x', x)
  np.save('np_save_' + str(input_size) + 'y', y)

x = np.load('np_save_' + str(input_size) + 'x.npy')
y = np.load('np_save_' + str(input_size) + 'y.npy')
x = x.astype('float32')

x = (x-128) / 128 # -1~1 normalization

y = np_utils.to_categorical(y, num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = TEST_SIZE, random_state = 111)

np.save('x_train_' + str(input_size) + 'melanoma', x_train)
np.save('x_test_' + str(input_size) + 'melanoma', x_test)
np.save('y_train_' + str(input_size) + 'melanoma', y_train)
np.save('y_test_' + str(input_size) + 'melanoma', y_test)

x_train = np.load('x_train_' + str(input_size) + 'melanoma.npy')
x_test = np.load('x_test_' + str(input_size) + 'melanoma.npy')
y_train = np.load('y_train_' + str(input_size) + 'melanoma.npy')
y_test = np.load('y_test_' + str(input_size) + 'melanoma.npy')

#pillow rotate x_train data
if not os.path.exists('np_save_' + str(input_size) + 'x_train_rotate' + str(angle) + '.npy'):
  x_train_rotate = []
  for img_j in x_train:
    q = Image.fromarray(np.uint8(img_j))
    q = q.rotate(angle)
    q = np.asarray(q)
    x_train_rotate.append(q)

  x_train_rotate = np.asarray(x_train_rotate)
  print("x_train_rotate[0]")
  plt.imshow(x_train_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_train_rotate' + str(angle), x_train_rotate)

if not os.path.exists('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_2) + '.npy'):
  x_train_rotate = []
  for img_j in x_train:
    q = Image.fromarray(np.uint8(img_j))
    q = q.rotate(angle_2)
    q = np.asarray(q)
    x_train_rotate.append(q)

  x_train_rotate = np.asarray(x_train_rotate)
  print("x_train_rotate[0]")
  plt.imshow(x_train_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_2), x_train_rotate)

if not os.path.exists('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_3) + '.npy'):
  x_train_rotate = []
  for img_j in x_train:
    q = Image.fromarray(np.uint8(img_j))
    q = q.rotate(angle_3)
    q = np.asarray(q)
    x_train_rotate.append(q)

  x_train_rotate = np.asarray(x_train_rotate)
  print("x_train_rotate[0]")
  plt.imshow(x_train_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_3), x_train_rotate)

if not os.path.exists('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_4) + '.npy'):
  x_train_rotate = []
  for img_j in x_train:
    q = Image.fromarray(np.uint8(img_j))
    q = q.rotate(angle_4)
    q = np.asarray(q)
    x_train_rotate.append(q)

  x_train_rotate = np.asarray(x_train_rotate)
  print("x_train_rotate[0]")
  plt.imshow(x_train_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_4), x_train_rotate)

#pillow rotate x_test data
if not os.path.exists('np_save_' + str(input_size) + 'x_test_rotate' + str(angle) + '.npy'):
  x_test_rotate = []
  for img_i in x_test:
    p = Image.fromarray(np.uint8(img_i))
    p = p.rotate(angle)
    p = np.asarray(p)
    x_test_rotate.append(p)
  
  x_test_rotate = np.asarray(x_test_rotate)
  print("x_test_rotate[0]")
  plt.imshow(x_test_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_test_rotate' + str(angle), x_test_rotate)

if not os.path.exists('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_2) + '.npy'):
  x_test_rotate = []
  for img_i in x_test:
    p = Image.fromarray(np.uint8(img_i))
    p = p.rotate(angle_2)
    p = np.asarray(p)
    x_test_rotate.append(p)
  
  x_test_rotate = np.asarray(x_test_rotate)
  print("x_test_rotate[0]")
  plt.imshow(x_test_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_2), x_test_rotate)

if not os.path.exists('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_3) + '.npy'):
  x_test_rotate = []
  for img_i in x_test:
    p = Image.fromarray(np.uint8(img_i))
    p = p.rotate(angle_3)
    p = np.asarray(p)
    x_test_rotate.append(p)
  
  x_test_rotate = np.asarray(x_test_rotate)
  print("x_test_rotate[0]")
  plt.imshow(x_test_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_3), x_test_rotate)  

if not os.path.exists('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_4) + '.npy'):
  x_test_rotate = []
  for img_i in x_test:
    p = Image.fromarray(np.uint8(img_i))
    p = p.rotate(angle_4)
    p = np.asarray(p)
    x_test_rotate.append(p)
  
  x_test_rotate = np.asarray(x_test_rotate)
  print("x_test_rotate[0]")
  plt.imshow(x_test_rotate[0])
  plt.show()

  np.save('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_4), x_test_rotate)  

#train model
base_model = ResNet50(weights='imagenet', include_top=False)
X = base_model.output
X = GlobalAveragePooling2D()(X)
predictions = Dense(classes, activation='softmax')(X)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers:
    layer.trainable = True

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#modelCheckPoint
modelCheckpoint = ModelCheckpoint(filepath = 'resnet50_melanoma_' +str(input_size) +'x' + str(input_size) + '.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='min',
                                  period=1)

early_stopping = EarlyStopping(
    monitor='val_acc',
    min_delta=0.0,
    patience=4)

history = model.fit(x_train, y_train, \
    batch_size=BATCH_SIZE, \
    epochs=epochs, \
    validation_data = (x_test, y_test), \
    verbose = 1, \
    callbacks=[early_stopping, modelCheckpoint])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#load model
model=load_model('resnet50_melanoma_' +str(input_size) +'x' + str(input_size) + '.h5')

x_rotate_train = np.load('np_save_' + str(input_size) + 'x_train_rotate' + str(angle) + '.npy')
x_rotate_test = np.load('np_save_' + str(input_size) + 'x_test_rotate' + str(angle) + '.npy')

x_rotate_train_2 = np.load('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_2) + '.npy')
x_rotate_test_2 = np.load('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_2) + '.npy')

x_rotate_train_3 = np.load('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_3) + '.npy')
x_rotate_test_3 = np.load('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_3) + '.npy')

x_rotate_train_4 = np.load('np_save_' + str(input_size) + 'x_train_rotate' + str(angle_4) + '.npy')
x_rotate_test_4 = np.load('np_save_' + str(input_size) + 'x_test_rotate' + str(angle_4) + '.npy')


# train data accuracy
y_real = np.argmax(y_train, axis=1)
y_pred = np.argmax(model.predict(x_train), axis=1)
print("Train accuracy (original): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_train), axis=1)
print("Train accuracy (rotated45): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_train_2), axis=1)
print("Train accuracy (rotated90): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_train_3), axis=1)
print("Train accuracy (rotated180): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_train_4), axis=1)
print("Train accuracy (rotated270): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))

#test data accuracy 
y_real = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)
print("Test accuracy (original): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_test), axis=1)
print("Test accuracy (rotated45): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_test_2), axis=1)
print("Test accuracy (rotated90): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_test_3), axis=1)
print("Test accuracy (rotated180): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))
y_pred = np.argmax(model.predict(x_rotate_test_4), axis=1)
print("Test accuracy (rotated270): %.2f%%" % (accuracy_score(y_real, y_pred) * 100))