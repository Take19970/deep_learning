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
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from collections import Counter

# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule
        }
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])
        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


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

# A common Conv2D model
input_image = Input(shape=(None, None, 3))  #standard
x = Conv2D(64, (3, 3), activation='relu')(input_image)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = AveragePooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)


"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.

the output of final model is the lengths of 10 Capsule, whose dim=16.

the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""

x = Reshape((-1, 128))(x)
capsule = Capsule(7, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=input_image, outputs=output)

# we use a margin loss
model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])  #standard
model.summary()

#modelcheckpoint
modelCheckpoint = ModelCheckpoint(filepath = 'Capsnet_melanoma_' +str(input_size) +'x' + str(input_size) + '.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='min',
                                  period=1)


# we can compare the performance with or without data augmentation
data_augmentation = False

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[modelCheckpoint])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)


    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=math.ceil(x_train.shape[0] / batch_size), #cifar10 = 391
        validation_data=(x_test, y_test),
        workers=4,
        callbacks=[modelCheckpoint])


    
y = model.predict(x_test)
plt.imshow(x_test[0])
plt.show()
print(y[0])
print('pred_y:', np.argmax(y[0]), ', label:', np.argmax(y_test[0]))

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
model=load_model('capsnet_melanoma_' +str(input_size) +'x' + str(input_size) + '.h5', custom_objects={'Capsule': Capsule, 'margin_loss': margin_loss})

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