import tensorflow as tf
from keras import layers
import os


def consolidate_outputs(inputs, outputs, loss, learning_rate):

    # Optmizer & Learning Rate
    opt = tf.keras.optimizers.Adam(learning_rate)
    
    # Save the model 
    keras_log_path = os.path.join('models/KERAS_LOGS')
    if not os.path.exists(keras_log_path): os.mkdir(keras_log_path)
    
    # Create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=opt, loss=loss)

    # Write out model specs to help load in saved model
    #self.export(self.json_file)
    
    return model


F_arr = [64,128,256,512]
N_arr = [2,2,2,2]
def build(
    input_shape = [120, 120, 3],
    batch_size = 32,
    learning_rate = 1e-3,
):
    inputs = layers.Input(shape=input_shape, batch_size=batch_size)
    x = inputs

    """
    Dimensions (assume X == imshape, often 64)
    1.) Input shape:           [batch,   X,    X,    X,    1] 
    2.) After first  layer:    [batch,   X,    X,    X,   64]
    3.) After second layer:    [batch,  X/2,  X/2,  X/2,  64]
    4.) After third  layer:    [batch,  X/4,  X/4,  X/4, 128]
    5.) After fourth layer:    [batch,  X/8,  X/8,  X/8, 256]
    6.) After fifth  layer:    [batch, X/16, X/16, X/16, 512] 
    7.) Start of last layer:   [batch, X/32, X/32, X/32, 512]
    8.) Final output:          [batch, 18]
    """

    x = first_layer(x)
    F_arr_practical = F_arr + [F_arr[0]] # make first same as last, so it gets called by F_arr[-1]
    for i in range(len(N_arr)):
        x = general_layer(x, F_arr_practical[i], N_arr[i], F_arr_practical[i-1])
    outputs = build_last_layer(x)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy
    model = consolidate_outputs(inputs, outputs, loss_fn, learning_rate)
    return model

# region layers and blocks
def first_layer(inputs):
    conv = layers.Conv2D(filters=64, kernel_size=[7,7], padding='same')(inputs)

    gn = layers.GroupNormalization()(conv)
    relu = layers.Activation('relu')(gn)
    return relu

def general_layer(inputs, F, N, prev_F):
    downsampled = layers.MaxPool2D(pool_size=[2,2], strides=2)(inputs)

    # If first pass -- pass in downsampled layer & prev_Filters
    # If not first pass -- pass in previous layer & current_filters
    intermediate = single_block(downsampled, F, prev_F)
    for i in range(N-1):
        intermediate = single_block(intermediate, F, F)
    return intermediate

def single_block(inputs, F, prev_F):
    
    first_conv = layers.Conv2D(filters=F, kernel_size=[3,3],padding='same')(inputs)
    gn1 = layers.GroupNormalization()(first_conv)
    relu = layers.Activation('relu')(gn1)
    
    second_conv = layers.Conv2D(filters=F,kernel_size=[3,3],padding='same')(relu)
    gn2 = layers.GroupNormalization()(second_conv)

    # channels last, has shape [batch_size, 64, 64, 64, channels]
    # pad by (channels_new - channels_old) / 2 on top and bottom
    # since want to pad along channels dim, have to lie and tell it channels first
    if F == prev_F:
        padded = inputs
    else:
        padded = layers.ZeroPadding2D(padding=(0,int((F-prev_F)/2)), data_format='channels_first')(inputs)

    combined = layers.Add()([padded, gn2])
    outputs = layers.Activation('relu')(combined)
    return outputs

def build_last_layer(inputs):
    #averaged = layers.AveragePooling2D(pool_size=[2,4,4])(inputs) # four 2x poolings before this, so reduced by 16x
    averaged = layers.AveragePooling2D(pool_size=[2,2])(inputs) 

    flat = layers.Flatten()(averaged)
    outputs = layers.Dense(18)(flat)
    return outputs

# endregion