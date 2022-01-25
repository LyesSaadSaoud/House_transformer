# Lyes @2022
# The data could be found in http://keddiyan.com/files/PowerForecast.html
# and T.-Y. Kim and S.-B. Cho, ‘‘Predicting residential energy consumption using CNN–LSTM neural networks,’’ Energy, vol. 182, pp. 72–81, Sep. 2019.

import numpy as np
import pandas as pd
import math
import os, datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pywt
import tensorflow
from numpy.random import seed
plt.style.use('seaborn')
def save_result(y_test,predicted_values):
    np.savetxt('./T_SWT_house4_min5_test.csv',y_test) # save path
    np.savetxt('./T_SWT_house4_min5_predicted.csv',predicted_values) # save path
df=pd.read_csv('./house1_min5_KWh.csv',header=None)# path to data

def data_preparation(dataset, window, lev):
    da = []
    for i in range(len(dataset) -window):
        coeffs = pywt.swt(dataset[i:window+i], wavelet='db2', level=lev)
        da.append(coeffs);
    return da
def data_reconstruction(dataset,window):
    da = []
    for i in range(len(dataset)):
        recon = pywt.iswt(dataset[i,:,:,:].tolist(), 'db2')
        da.append(recon[window-1])
    return da
def create_dataset(dataset, look_back):
    dataX,dataY=[],[]

    for i in range(len(dataset)-look_back):
        a=dataset[i:(i+look_back),0:4]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0:4])
    return np.array(dataX),np.array(dataY)
class Time2Vector(Layer):
    ''' https://arxiv.org/abs/1907.05321'''
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):

        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        return tf.concat([time_linear, time_periodic], axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config
class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

        self.key = Dense(self.d_k,
                         input_shape=input_shape,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')

        self.value = Dense(self.d_v,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out
class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))

            # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        self.linear = Dense(input_shape[0][-1],
                            input_shape=input_shape,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear
class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config
class TransformerDecoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerDecoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config

def create_model():
  time_embedding = Time2Vector(seq_len)
  layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  layer4 = TransformerDecoder(d_k, d_v, n_heads, ff_dim)
  layer5 = TransformerDecoder(d_k, d_v, n_heads, ff_dim)
  in_seq = Input(shape=(seq_len, inp_len))
  x = time_embedding(in_seq)
  x = Concatenate(axis=-1)([in_seq, x])
  x = layer1((x, x, x))
  x = layer2((x, x, x))
  x = layer3((x, x, x))
  x = layer4((x, x, x))
  x = layer5((x, x, x))
  x = GlobalAveragePooling1D(data_format='channels_first')(x)
  x = Dropout(0.1)(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.1)(x)
  out = Dense(out_len, activation='linear')(x)

  model = Model(inputs=in_seq, outputs=out)
  model.compile(loss='mse', optimizer='RMSProp', metrics=['mae', 'mape'])
  return model
batch_size = 32
seq_len = 1
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
lev=3
inp_len=2*lev
out_len=2*lev
window=200
look_back = 12
dataset = df.values
dataset=dataset.astype('float32')

s = np.squeeze(dataset[:12000*3], axis=1)  #

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
da=data_preparation(s, window, lev)
Vv = np.array(da)
vv = Vv.reshape(Vv.shape[0],2*lev*Vv.shape[3])

dataset = scaler.fit_transform(vv)

dat = dataset.reshape(Vv.shape[0],2*lev,Vv.shape[3])
alpha=0.6667
trainX,trainY=dat[:int(dat.shape[0]*alpha),:,window-seq_len-1:window-1],dat[:int(dat.shape[0]*alpha),:,window-1]
testX,testY=dat[int(dat.shape[0]*alpha):,:,window-seq_len-1:window-1],dat[int(dat.shape[0]*alpha):,:,window-1]
testX_a, testY_a = dat[:,:,window-seq_len-1:window-1],dat[:,:,window-1]
testX_a=np.transpose(testX_a, (0, 2, 1))
trainX=np.transpose(trainX, (0, 2, 1))
testX =np.transpose(testX, (0, 2, 1))
model = create_model()
model.summary()
# Training data
X_train, y_train = trainX,trainY
###############################################################################
# Validation data
X_val, y_val = testX,testY
###############################################################################
# Test data
X_test, y_test = testX_a,testY_a
callback = tf.keras.callbacks.ModelCheckpoint('Transformer_5min.hdf5',
                                                      monitor='val_loss',
                                                      save_best_only=True,
                                                      verbose=1)
history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=50,
                            validation_data=(X_val, y_val),
                            callbacks=[callback])
model = tf.keras.models.load_model('Transformer_5min.hdf5',
                                           custom_objects={'Time2Vector': Time2Vector,
                                                           'SingleAttention': SingleAttention,
                                                           'MultiAttention': MultiAttention,
                                                           'TransformerEncoder': TransformerEncoder}
                                           )
# Use the whole signal (both train and validation data)
# The metrics are computed only using the validation part.
# This is needed for the signal processing
testPredict_a =  model.predict(testX_a)
d=dat
d[:,:,window-1]=testPredict_a
D = d.reshape(d.shape[0],d.shape[1]*d.shape[2])
R = scaler.inverse_transform(D)
R = R.reshape(d.shape[0],lev,2,d.shape[2])
re=data_reconstruction(R, window)
Re = np.array(re)
tY = s[window:].reshape(s[window:].shape[0],1)
P1 = Re.reshape(Re.shape[0],1)
testYa=tY[int(tY.shape[0]*alpha):,:] # take onle the validation part
testPredicta = P1[int(tY.shape[0]*alpha):, :] # take onle the validation part
predicted_values, y_test=testPredicta[1:], testYa[:-1]
test_rmse = math.sqrt( mean_squared_error(y_test, predicted_values))
test_mae=mean_absolute_error(y_test, predicted_values)
mape=100*np.mean(np.divide(abs(y_test- predicted_values),y_test))

fig = plt.figure()
plt.plot(y_test)
plt.plot(predicted_values)
plt.xlabel('Time/5min')
plt.ylabel('Electricity load (kWh)')
plt.legend(['True', 'Predict'], loc='upper left')
plt.show()
print('RMSE:  %.4f' % test_rmse)
print('MAE:  %.4f' % test_mae)
print('MAPE:  %.4f' % mape)
