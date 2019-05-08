import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation

def initNeuralNet(nInputs, nHidden, nOutputs):
    # Define the model architecture
    model = Sequential()
    model.add(Dense(nHidden, input_dim = nInputs,
                    kernel_initializer="normal",  activation = "relu"))
    model.add(Dense(nOutputs, kernel_initializer="normal"))

    model.compile(loss='mean_squared_error', optimizer="Adam",
    metrics=['mean_absolute_error'])
    return model


def stringToArray(s):
    return np.array(s.split(), dtype='float')
