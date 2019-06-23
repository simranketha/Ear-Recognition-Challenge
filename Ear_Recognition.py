
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.mobilenet_v2 import MobileNetV2


# ## loading dataset

PATH = 'Dataset/'


def dataset(mode):
    
    total_images = []
    label=[]
    mode_path = os.path.join(PATH,mode)
    subjects = os.listdir(mode_path)

    for subject in subjects:
        image_path  = os.path.join(mode_path,subject)
        images = os.listdir(image_path)
        for image in images:
            if(image.endswith(".png")):
                file = os.path.join(image_path,image)
                total_images.append(cv2.resize(cv2.imread(file),(100,100)))
                label.append(int(subject))
    total_images = np.array(total_images)
    label = np.array(label)
    return total_images , label



train,label = dataset('Train Dataset/')


# ## train test split



def split_data(data,label,valid_len):
    valid_len = int(valid_len*len(data)/100)
    return (data[0:len(data)-valid_len],label[0:len(data)-valid_len],
            data[len(data)-valid_len:len(data)],label[len(data)-valid_len:len(data)])


x_train,y_train,x_valid,y_valid = split_data(train,label,20)


print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# ## label encoding



y_train = keras.utils.to_categorical(y_train, 1000)
y_valid = keras.utils.to_categorical(y_valid, 1000)




modelmain=MobileNetV2(input_shape=x_train[0].shape, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)




modelmain.summary()




model = Sequential()
model.add(modelmain)
model.add(GlobalAveragePooling2D())
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 150, activation = 'softmax'))
model.summary()



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])




model.fit_generator(datagen.flow(x_train,y_train, batch_size=70),verbose=1,validation_data=(x_valid,y_valid))

