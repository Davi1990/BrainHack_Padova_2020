from __future__ import print_function
import numpy as np
import nibabel as nib
import glob
import os
import pandas as pd
import csv as csv
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Merge,Lambda, Embedding, Bidirectional, LSTM, Dense, RepeatVector, Dropout
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy

print (keras.backend.image_data_format())
if True:

    mri_all=sorted(glob.glob("/home/davide/Desktop/DAVE/BHG_Padova/pre_central_gyrus/PIOP1_precentral/combined/resempled/*"))

    demographic = pd.read_excel('/home/davide/Desktop/DAVE/BHG_Padova/PIOP1_participant-info.xlsx', sep='\t')
    demographic['handedness'] = demographic['handedness'].fillna(0)
    demographic['handedness'].replace(['right','left', 'ambidextrous'],[0,1,1],inplace=True)
    y = np.array(demographic['handedness'])
    y = to_categorical(y)

    mri_all, test_files, y, test_labels = train_test_split(mri_all,y,test_size=0.2)


batch_size = 10
num_classes = 100
epochs = 100
file_size = 110

dimx,dimy,channels = 64, 64, 37

# Convert class vectors to binary class matrices.

inpx = Input(shape=(dimx,dimy,channels,1),name='inpx')
x = Convolution3D(2, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv1')(inpx)
#x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       #border_mode='valid', name='pool1')(x)
x = Convolution3D(4, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv2')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       border_mode='valid', name='pool2')(x)
x = Convolution3D(8, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv3')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       border_mode='valid', name='pool3')(x)


hx = Flatten()(x)



score = Dense(100, activation='softmax', name='fc8')(hx)

model = Model(inputs=inpx, outputs=score)

opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


model.summary()
for i in range(1):
    files = mri_all[i*file_size:(i+1)*file_size]
    train_x = []
    for f in files:
        img = nib.load(f)
        img_data = img.get_data()
        img_data = np.asarray(img_data)
        if(img_data.shape==(176,256,256)):
            img_data = img_data.reshape([256,256,176])
        img_data = img_data[:,:,0:144]
        train_x.append(img_data)
    x_train = np.asarray(train_x)
    print('\n iteration number :', i,'\n')

    x_train = np.expand_dims(x_train,4)
    print ('\n', x_train.shape)
    y_train = y[i*file_size:(i+1)*file_size]
    y_train = keras.utils.to_categorical(y_train, num_classes)

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,verbose=2)


test_x,test_y = [], test_labels

for i,f in enumerate(test_files):
    img=nib.load(f)
    img_data=img.get_data()
    img_data= np.asarray(img_data)
    if(img_data.shape==(176,256,256)):
        img_data=img_data.reshape([256,256,176])
    img_data=img_data[:,:,0:144]
    test_x.append(img_data)

test_x = np.asarray(test_x)
test_x = np.expand_dims(test_x,4)
test_y = keras.utils.to_categorical(test_y, num_classes)
pred = model.predict([test_x])
print('/n', pred.shape)

pred = [i.argmax() for i in pred]

print ('\n assertion::\n',len(pred),len(test_y))



mae = mean_absolute_error(test_y,pred)
pearson = scipy.stats.pearsonr(test_y,pred)
r2 = r2_score(test_y,pred)
mse = mean_squared_error(test_y,pred)

scores =[mae,pearson,r2,mse]

pd.to_pickle(scores,'/output/scores_out')
pd.to_pickle(pred,'/output/pred_out')

print('\n\n MAE is :- ', mae)
print('\n\n pearsonr is:-',pearson)
print('\n\n R2 is :- ', r2)
print('\n\n MSE is :- ', mse)
print('\n\n')
