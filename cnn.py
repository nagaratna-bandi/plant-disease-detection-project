import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm

##from keras.preprocessing.image import ImageDataGenerator
##from keras.models import Sequential
##from keras.layers import Conv2D, MaxPooling2D
##from keras.layers import Activation, Dropout, Flatten, Dense
##from keras import backend as K
##from keras.preprocessing import image 
##
##import numpy as np


##TRAIN_DIR = 'E:/P_2020/PlantDisease-master/leaf-disease/Datasets/D'
##TEST_DIR = 'E:/P_2020/PlantDisease-master/leaf-disease/Datasets/D/test'

TRAIN_DIR = 'C:\\Users\\ADMIN\Desktop\\fianl code\\GUI\\leafdiseasefinal(hardware) - Copy\\train'
TEST_DIR = 'C:\\Users\\ADMIN\\Desktop\\fianl code\\GUI\\leafdiseasefinal(hardware) - Copy\\test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'healthyvsunhealthynew-{}-{}.model'.format(LR, '2conv-basic')

##
##if K.image_data_format() == 'channels_first':
##    shape = (3, IMG_SIZE, IMG_SIZE)
##else:
##    shape = (IMG_SIZE, IMG_SIZE, 3)
##
##    
def label_img(img):
    word_label = img[0]
    print(word_label)
  
    if word_label == 'h':
        print('healthy')
        return [1,0,0,0,0,0]
    
    elif word_label == 'b':
        print('Bacterial')
        return [0,1,0,0,0,0]
    elif word_label == 'v':
        print('virus')
        return [0,0,1,0,0,0]
    elif word_label == 's':
        print('Spectoria')
        return [0,0,0,1,0,0]
    elif word_label == 'm':
        print('leafmold')
        return [0,0,0,0,1,0]
    elif word_label == 'x':
        print('leafmold')
        return [0,0,0,0,0,1]    

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        print('##############')
        print(label)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]
print(X.shape)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]
print(test_x.shape)

model.fit({'input': X}, {'targets': Y},n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)











        
