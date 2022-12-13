import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import os
import glob
import shutil

globalreg = 0.0005
in_shape = [345, 280, 3]

encoder_input = keras.Input(in_shape, name="original_img")

# conv1 Filter dimensions: [3, 3, 3, 16]  Outputs: (None, 115, 94, 16)
x = layers.Conv2D(16, 3, activation="relu", padding = "same")(encoder_input)
x = layers.MaxPooling2D(pool_size=(3,3), padding="same")(x)
x = layers.BatchNormalization(scale=True).call(inputs=x,training=False)
print(x.shape)
# conv2 Filter dimensions: [3, 3, 16, 32] Inputs: [16, 160, 91, 16] Outputs: [16, 160, 91, 32]
x = layers.Conv2D(32, 3, activation="relu", padding = "same")(x)
x = layers.MaxPooling2D(pool_size=(3,3), padding="same")(x)
x = layers.BatchNormalization(scale=True).call(inputs=x,training=False)
print(x.shape)
# conv3 Filter dimensions: [3, 3, 32, 32] Inputs: [16, 54, 31, 32] Outputs: [16, 54, 31, 32]
x = layers.Conv2D(32, 3, activation="relu", padding = "same")(x)
x = layers.MaxPooling2D(pool_size=(3,3), padding="same")(x)
x = layers.BatchNormalization(scale=True).call(inputs=x,training=False)
print(x.shape)
# conv4 Filter dimensions: [3, 3, 32, 32] Inputs: [16, 18, 11, 32] Outputs: [16, 18, 11, 32]
x = layers.Conv2D(32, 3, activation="relu", padding = "same")(x)
x = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)
x = layers.BatchNormalization(scale=True).call(inputs=x,training=False)
print(x.shape)
# fully connected Inputs: [16, 9, 6, 32] Outputs: [16, 64]
x = layers.Reshape((x.shape[1]*x.shape[2]*x.shape[3],))(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.L2(globalreg))(x)
print(x.shape)
# output Inputs: [16, 64] Outputs: [16, 16] encoder_output will be used for two branches, lstm and decoder
encoder_output = layers.Dense(16, activation=None, kernel_regularizer=regularizers.L2(globalreg))(x)
print(encoder_output.shape)

# do I need to build an encoder here, and start a decoder later, or just use the last layer of encoder to the decoder?
#encoder = keras.Model(encoder_input, encoder_output, name="encoder")
#encoder.summary()

# start to build decoder
#decoder_input = keras.Input(shape=(16,), name="encoded_img")

#x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Reshape((4, 4, 1))(encoder_output)
# deconv1 Filter dimensions: [3, 3, 16, 16] Inputs: [16, 1, 1, 17] Outputs: [16, 1, 1, 16]
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.UpSampling2D(2)(x)

# deconv2 Filter dimensions: [3, 3, 16, 32] Inputs: [16, 2, 2, 16] Outputs: [16, 2, 2, 32]
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)

# deconv3 Filter dimensions: [3, 3, 16, 32] Inputs: [16, 2, 2, 16] Outputs: [16, 2, 2, 32]
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)

# deconv4 Filter dimensions: [3, 3, 32, 32] Inputs: [16, 4, 4, 32] Outputs: [16, 4, 4, 32]
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)

# deconv5 Filter dimensions: [3, 3, 32, 32] Inputs: [16, 4, 4, 32] Outputs: [16, 4, 4, 32]
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)

# deconv6 Filter dimensions: [3, 3, 32, 32] Inputs: [16, 4, 4, 32] Outputs: [16, 4, 4, 32]
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(16, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)

# deconv6 Filter dimensions: [3, 3, 32, 32] Inputs: [16, 4, 4, 32] Outputs: [16, 4, 4, 32]
x = layers.Conv2DTranspose(16, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)

# linear activation
decoder_output = layers.Conv2DTranspose(3, 3, activation=None)(x)


# attach lstm to encoder output
activations = layers.LSTM(16)(encoder_output)
step_last = activations[-1]
xaiverinitializer = tf.keras.initializers.GlorotNormal()
lstm_output = layers.Dense(16, activation="tanh", kernel_initializer=xaiverinitializer, kernel_regularizer=regularizers.L2(globalreg))(step_last)
lstm_output = layers.Dense(1, activation=None, kernel_initializer=xaiverinitializer, kernel_regularizer=regularizers.L2(globalreg))(lstm_output)

predicted_treatment = keras.backend.squeeze(lstm_output)



modela = keras.Model(inputs=encoder_input, outputs=predicted_treatment, name="modela")

modela.complie(optimizer="adam", loss=keras.losses.BinaryCrossentropy())


modelb = keras.Model(inputs=encoder_output, outputs=decoder_output, name="modelb")

modelb.compile(optimizer="sgd", loss=keras.losses.MeanAbsoluteError())





allimgs = glob.glob("Sorghumnutrient/**/Vis_SV*/*.png.sml.png")

dates = []
angles = []
trts = []
genotypes = []
ids = []
imgpaths = allimgs

metadata = pd.read_csv("HTP_meta.csv")

for fn in allimgs:
    fpath, fn = os.path.split(fn)
    fpath = fpath.split("/")
    label = fpath[-2]
    angle = fpath[-1]
    label = label.split("_")
    plant = label[1]
    date = label[2]
    plant = plant.split("-")
    plantid = plant[2]
    trt = plant[3]
    genotype = metadata.loc[metadata.Identifier==plantid, "Genotype"].items()
    dates.append(date)
    trts.append(trt)
    genotypes.append("genotype")
    angles.append(angle)
    ids.append(plantid)

metadata2 = pd.DataFrame({'ids':ids, 'genotypes':genotypes, 'trts':trts, 'dates':dates, 'angles':angles, 'imgpaths':imgpaths})



