import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class modelFactory():

  def __init__(self, latent_dim=2):
    self.latent_dim = latent_dim

  def getDNN(self, aD):
    """## Build a deep neural network"""

    # IN: ~3000
    # OUT: ~1200

    dnn_inputs = keras.Input(shape=(aD.maxpixels, 3, 1))

    # x1 = layers.Reshape((aD.maxpixels*3, 1))(dnn_inputs)
    x1 = layers.Conv2D(aD.maxhits, kernel_size=5, strides=2, padding='same')(dnn_inputs)
    # x1 = pooling(dnn_inputs)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dense(aD.maxhits)(x1)

    x = layers.Flatten()(dnn_inputs)
    
    masking_layer = layers.Masking(mask_value=0.0)
    # unmasked_embedding = tf.cast(tf.tile(tf.expand_dims(x, axis=-1), [1, 1, aD.maxpixels*3]), tf.float32)
    x = masking_layer(x)
    

    # x = layers.Flatten()(masked_inputs)
    # x = layers.Dense(1024*2, activation="relu")(x)
    # x = layers.Dropout(0.10)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.10)(x)

    x_x     = layers.Dense(512, activation="relu")(x)
    x_x = layers.Dropout(0.10)(x_x)

    x_y     = layers.Dense(512, activation="relu")(x)
    x_y = layers.Dropout(0.10)(x_y)

    x_edep  = layers.Dense(512, activation="relu")(x)
    x_edep = layers.Dropout(0.10)(x_edep)

    x_esum  = layers.Dense(512, activation="relu")(x)
    x_esum = layers.Dropout(0.10)(x_esum)

    x_bragg = layers.Dense(512, activation="relu")(x)
    x_bragg = layers.Dropout(0.10)(x_bragg)
    
    x_x     = layers.Dense(aD.maxhits,activation="relu")(x_x)
    x_y     = layers.Dense(aD.maxhits,activation="relu")(x_y)
    x_edep  = layers.Dense(aD.maxhits,activation="relu")(x_edep)
    x_esum  = layers.Dense(aD.maxhits,activation="relu")(x_esum)
    x_bragg = layers.Dense(aD.maxhits,activation="relu")(x_bragg)

    x_x     = layers.Multiply()([x_x    , x1])
    x_y     = layers.Multiply()([x_y    , x1])
    x_edep  = layers.Multiply()([x_edep , x1])
    x_esum  = layers.Multiply()([x_esum , x1])
    x_bragg = layers.Multiply()([x_bragg, x1])

    x = layers.Concatenate()([x_x, x_y, x_edep, x_esum, x_bragg])

    # print(dnn_outputs)
    # print(dnn_outputs.shape)


    # x = layers.Dropout(0.10)(x)
    # x = layers.Dense(256, activation="sigmoid")(x)
    # x = layers.Dropout(0.10)(x)
    # x = layers.Dense(128, activation="sigmoid")(x)
    # x = layers.Dropout(0.10)(x)
    # x = layers.Dense(64, activation="sigmoid")(x)
    # x = layers.Dropout(0.10)(x)
    # x = layers.Dense(32, activation="sigmoid")(x)
    
    
    x = layers.Dense(aD.maxhits*5, activation="sigmoid")(x)
    dnn_outputs = layers.Reshape((aD.maxhits, 5, 1))(x)

    dnn = keras.Model(dnn_inputs, dnn_outputs, name="DNN")

    if aD.verbose:
      dnn.summary()

    return dnn


  def getEncoder(self, aD):
    """## Build the encoder"""


    # encoder_inputs = keras.Input(shape=(aD.maxx, aD.maxy, 1))
    encoder_inputs = keras.Input(shape=(aD.maxpixels, 3, 1))
    
    embedding = layers.Embedding(input_dim=aD.maxpixels, output_dim=aD.maxpixels, mask_zero=True)
    masked_inputs = embedding(encoder_inputs)
    

    # x = layers.Flatten()(masked_inputs)
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.Conv2D(32, 2, strides=2, activation="relu", padding="same")(masked_inputs)
    # x = layers.Conv2D(64, 2, strides=2, activation="relu", padding="same")(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(self.latent_dim)(x)


    z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    if aD.verbose:
      encoder.summary()

    return encoder

  def getDecoder(self, aD):
    """## Build the decoder"""

    latent_inputs = keras.Input(shape=(self.latent_dim,))
    
    # x = layers.Dense(16, activation="relu")(latent_inputs)
    # x = layers.Dense(aD.maxhits*3*1, activation="relu")(latent_inputs)
    # # x = layers.Dense(7*7*32, activation="relu")(latent_inputs)

    # # x = layers.Dropout(0.25)(x)

    # # x = layers.Reshape((aD.maxhits, 3, 1))(x)
    # x = layers.Reshape((7, 7, 32))(x)
    # x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding="same")(x)
    # x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding="same")(x)
    # x = layers.Conv2DTranspose(1, 3, strides=1, padding="same")(x)
    # decoder_outputs = layers.Reshape((aD.maxhits, 3, 1))(x)




    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.Reshape((32, 2, 1))(x)
    # x = layers.UpSampling2D()(x)
    # x = layers.Conv2D(64, 2, strides=2, activation="relu", padding="same")(x)

    x = layers.Dense(aD.maxhits, activation="tanh")(x)
    x = layers.Dense(aD.maxhits*3, activation="sigmoid")(x)
    decoder_outputs = layers.Reshape((aD.maxhits, 3, 1))(x)


    # decoder_outputs = layers.Conv2DTranspose(32, 3, activation='sigmoid', padding="same")(x)
    # x = layers.Conv2DTranspose(64, 3, activation="relu",
    #                           strides=2, padding="same")(x)
    # x = layers.Conv2DTranspose(32, 3, activation="relu",
    #                           strides=2, padding="same")(x)
    # decoder_outputs = layers.Conv2DTranspose(
    #     1, 3, activation="sigmoid", padding="same")(x)

    # decoder_outputs = layers.Reshape((aD.maxhits, 3))(x)
    # decoder_outputs = layers.Dense(256)(x)

    # print(np.shape(decoder_outputs))

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    if aD.verbose:
      decoder.summary()

    return decoder
