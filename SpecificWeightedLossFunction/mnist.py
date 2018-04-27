from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop,SGD
import functools
from itertools import product
from keras import backend as K
import keras.callbacks as cbks
import tensorflow as tf
from itertools import product

batch_size = 100
num_classes = 10
epochs = 1


#Weighted custom loss function
def w_categorical_crossentropy(weights):
  def loss(y_true, y_pred):
    Weights = weights[:num_classes,:num_classes]
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(num_classes), range(num_classes)):
        final_mask += Weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t]
    return K.categorical_crossentropy(y_true, y_pred,from_logits=False) * final_mask
  return loss
 
#Calculate Confusion Matrix 
def Get_ConfMatrix(pred,actual,num_classes):
  ConfMatrix = np.zeros((num_classes,num_classes))  
  PredY = np.argmax(pred, axis=1)
  RealY = np.argmax(actual, axis=1)
  for i in range(PredY.shape[0]):
    Val= ConfMatrix.item(RealY.item(i),PredY.item(i))
    Val+=1
    ConfMatrix.itemset((RealY.item(i),PredY.item(i)),Val)    
  RowSum=ConfMatrix.sum(axis=1)
  return ConfMatrix/RowSum[:,None]
  
  
def SaveMatrixToHeatmap(data,ImgName):
  plt.gcf().clear()
  heatmap = plt.pcolor(data)
  ax=plt.gca()
  for y in range(data.shape[0]):
      for x in range(data.shape[1]):
          plt.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10,
                   fontweight='bold'
                   )
          if ( x != y ):
           ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='black', lw=1))

  for y in range(data.shape[0]):
      for x in range(data.shape[1]):
          if ( x == y ):
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=3))

  ax.set_ylim(ax.get_ylim()[::-1])
  ax.xaxis.tick_top()
  ax.yaxis.set_ticks(np.arange(0, 8, 1))
  ax.yaxis.tick_left()
  ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
  ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
  column_labels = list('01234567')
  row_labels = list('01234567')
  ax.set_xticklabels(column_labels, minor=False)
  ax.set_yticklabels(row_labels, minor=False)
  plt.colorbar(heatmap)
  Fig=plt.gcf()
  plt.draw()
  Fig.savefig(ImgName)  

#You can change weight matrix construction rule. This function simply add 1 to validation matrix ratio.
def CreateWeightMat(ConfMat):
  R,C = ConfMat.shape
  weightMat = np.ones((R,C))
  for i in range(R):
   for j in range(C):
     if ( i != j ):
       weightMat[i,j] = 1 + ConfMat[i,j]
  return  weightMat  
  
#create train and test set
(x_total, y_total), (x_test, y_test) = mnist.load_data()

x_total = x_total.reshape(60000, 784)

x_val = x_total[:5000,:]
x_train = x_total[5000:,:]

y_val = y_total[:5000,]
y_train = y_total[5000:,]

x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_val /= 255
x_test /= 255

#convert label
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#build model
input_tensor = Input(shape=(784,), name='input')
w_array = Input(shape=(num_classes,), name='weight')
x1=Dense(512, activation='relu')(input_tensor)
x2=Dropout(0.2)(x1)
x3=Dense(512, activation='relu')(x2)
x4=Dropout(0.2)(x3)
out= Dense(num_classes, activation='softmax', name='output')(x4)
model = Model(inputs = [input_tensor,w_array], outputs = out)

model.compile(loss=w_categorical_crossentropy(w_array),
              optimizer='rmsprop',
              metrics=['accuracy'])

#Create Weight array. You can modify it. All inputs should have same length.
Weights_train = np.ones((x_train.shape[0],num_classes))
Weights_val = np.ones((x_val.shape[0],num_classes))
              
for i in range(10):  
  #Create Validation Data
  validation_data=({'input': x_val,'weight':Weights_val},{'output': y_val})  
  
  history = model.fit({'input': x_train, 'weight': Weights_train},
                      {'output' : y_train},
                      batch_size=batch_size,
                      validation_data = validation_data,
                      epochs=epochs)
                      
  Val_Pred = model.predict([x_val,Weights_val])
  Conf_Mat = Get_ConfMatrix(Val_Pred,y_val,num_classes)
  WeightMat = CreateWeightMat(Conf_Mat)
  Weights_val[:num_classes, :num_classes] = WeightMat
  Weights_train[:num_classes, :num_classes] = WeightMat
  
  #you can save your weight matrix in heatmap form.
  #WeightName =  str(i) + "_WeigtMat.png"

Weights_test = np.ones((x_test.shape[0],num_classes))

score = model.evaluate({'input': x_test, 'weight': Weights_test},
                       {'output' : y_test})
                       
print('Test loss:', score[0])
print('Test accuracy:', score[1])
