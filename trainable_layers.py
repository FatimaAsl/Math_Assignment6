from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_with_hidden_layers(input_length, 
                                          activation_func_array=['sigmoid','sigmoid'],
                                          hidden_layers_sizes=[50, 20],
                                          output_function='softmax',
                                          output_length=10):

    model = keras.Sequential()
    # Create the input layer
    model.add(layers.input(shape=(input_length)))
    # Create the hidden layers
    for i, in enumerate(hidden_layers_sizes):
        activation_func =activation_func_array[i] if i< len(activation_func_array) else activation_func[-1]
        model.add(layers.Dense(size, activation=activation_func))
    # Create the output layer
    model.add(layers.Dense(output_length, activation=output_function))
    return model


def set_layers_to_trainable(model, trainable_layer_numbers):
    for i, layer in enumerate(model.layers):
        layer.trainable= i in trainable_layer_numbers
    return model