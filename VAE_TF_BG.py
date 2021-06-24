#!/usr/bin/env python
# coding: utf-8

import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras import metrics
from tensorflow.keras.layers.experimental import preprocessing
import random as rn
from alive_progress import alive_bar
from PIL import Image
import argparse
import modelDefinitions as mD

# plt.rcParams.update({
#     "text.usetex": True
#     })
# params = {'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
# plt.rcParams.update(params)

example1 = "Train a model named 'myModel' with testing enabled:              ./VAE_TF_BG.py --save myModel --tests"
example2 = "Train a model with 10 epochs and batch size 32, latent dim 4:    ./VAE_TF_BG.py --epochs 10 --batch 32 --latent_dim 4"
example3 = "Only scan the data without any training or testing:              ./VAE_TF_BG.py --scanonly -v"
example4 = "Load a pretrained model for testing, don't use GPU:              ./VAE_TF_BG.py --load myModel --noGPU --tests"
exampleRuns = "Examples:\n"+example1+"\n"+example2+"\n"+example3+"\n"+example4+"\n"

argparser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    prog="VAE_TF_BG",
    epilog=exampleRuns)
argparser.add_argument('--data', dest='fileNameIn', type=str, default="head_input_230MeV_cs.txt", help="path for the raw data (default: %(default)s)")
argparser.add_argument('--save', dest='fileNameOut', type=str, default="test", help="filename for the model to save (default: %(default)s)")
argparser.add_argument('--load', dest='modelNameIn', type=str, default=None, help="filename for the model to load (default: %(default)s)")
argparser.add_argument('--lnum', dest='lnum', type=str, default="all", help="the number of lines to be processed from the datafile (default: %(default)s)")
argparser.add_argument('--ratio', dest='ratio', type=float, default=0.98, help="ratio of training/validation data (default: %(default)s)")
argparser.add_argument('--eval', dest='eval', action='store_true', default=False, help="enable evaluation (default: %(default)s)")
argparser.add_argument('--tests', dest='tests', action='store_true', default=False, help="enable testing with plots as output (default: %(default)s)")
argparser.add_argument('--nosave', dest='nosave', action='store_true', default=False, help="do not save the train results (default: %(default)s)")
argparser.add_argument('--scanonly', dest='scanonly', action='store_true', default=False, help="only scan the whole dataset without training (default: %(default)s)")
argparser.add_argument('--epochs', dest='epochs', type=int, default=10, help="number of epochs for training (default: %(default)s)")
argparser.add_argument('--batch', dest='batch', type=int, default=32, help="batch size for training (default: %(default)s)")
argparser.add_argument('--GPU', dest='GPU', action='store_true', default=False, help="use GPU if available (default: %(default)s)")
argparser.add_argument('--noGPU', dest='noGPU', action='store_true', default=False, help="don't use GPU (default: %(default)s)")
argparser.add_argument('--maxhits', dest='maxhits', type=int, default=256, help="maximum number of allowed hits on a detector chip (default: %(default)s)")
argparser.add_argument('--maxpixels', dest='maxpixels', type=int, default=1024, help="maximum number of allowed active pixels on a detector chip (default: %(default)s)")
argparser.add_argument('--maxsumedep', dest='maxsumedep', type=float, default=16, help="maxsumedep on a detector chip (default: %(default)s)")
argparser.add_argument('--latent_dim', dest='latent_dim', type=int, default=2, help="latend dimension (default: %(default)s)")
argparser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-5, help="learning rate for optimizer (default: %(default)s)")
argparser.add_argument('-v', dest='verbose', action='store_true', help='verbose mode')
argparser.add_argument('--version', action='version', version='%(prog)s 0.1')

class auxData:
  maxx = 0
  maxy = 0
  maxedep = 0
  lnum = 0
  totalNum = 0
  maxlayers = 0
  sumEdep = 0
  
  def __init__(self, args):
    self.fileNameIn = args.fileNameIn
    self.verbose = args.verbose
    self.scanonly = args.scanonly
    self.maxhits = args.maxhits
    self.maxpixels = args.maxpixels
    self.lnum = args.lnum
    self.ratio = args.ratio
    self.maxsumedep = args.maxsumedep

  def addLine(self):
    self.lnum+=1

  def show(self):
    if self.verbose:
      print("The maximum values:")
      print("maxx=%d maxy=%d maxedep=%lf maxlayers=%d maxsumedep=%lf, from " %(self.maxx, self.maxy, self.maxedep, self.maxlayers, self.maxsumedep)+str(self.lnum)+" lines")


def rescaler(data, a, b, min, max):
    return (b - a) * (data - min) / (max - min) + a


def getMax(args):
    
    aD = auxData(args)
    lnum = 0

    with open(aD.fileNameIn, "rb") as f:
        while True:
            line = f.readline()
            line_fields = line.split()
            if len(line) == 0:
                # End of file
                break
            lnum+=1
            hits = line_fields[7:]

            for c in zip(hits[::2], hits[1::2]):
                aD.maxx = max(int(c[0]), aD.maxx)
                aD.maxy = max(int(c[1]), aD.maxy)
                
            aD.maxedep = max(float(line_fields[5]), aD.maxedep)

            aD.maxlayers = max(int(line_fields[4]), aD.maxlayers)

    aD.maxedep = np.ceil(1.02*aD.maxedep)
    aD.maxx+=1
    aD.maxy+=1

    if aD.lnum != "all" and not aD.scanonly:
      aD.totalNum = int(aD.lnum)
      # totalNum = 14042
      # totalNum = 1000
    else:
      aD.totalNum = lnum

    return aD


def loadData(aD):
    # i = 0
    labels = []
    frames = []
    maxhits = 0
    nhits = 0
    maxpixels = 0
    maxSumEdep = 0
    npixels = 0
    whichIm = 0
    whichLn = 0
    iLine = 0
    iIm = 0
    iPixel = 0
    
    with open(aD.fileNameIn, "rb") as f:

        last = True
        empty = True
        BraggPeak = 0.1
        sumEdep = 0

        with alive_bar(aD.totalNum) as bar:
          for i in range(aD.totalNum):
            bar()
            
            line = f.readline()
            
            last_pos = f.tell()
            line_next = f.readline()
            if len(line) == 0:
              # End of file
              break
            line_fields = line.split()
            try:
                line_next_fields = line_next.split()
            except:
                line_next_fields = None


            if last:
                last = False
                empty = True

                # frame = np.zeros((aD.maxx, aD.maxy), dtype=np.uint8)
                frame = np.zeros((aD.maxpixels, 3), dtype=float)
                flabels = np.zeros((aD.maxhits, 5), dtype=float)                

            chip = np.array([int(line_fields[0]), int(line_fields[1])])
            try:
                chip_next = np.array([int(line_next_fields[0]), int(line_next_fields[1])])
            except:
                chip_next = np.array([None, None])

            nLayer = int(line_fields[4])

            try:
                nLayer_next = int(line_next_fields[4])
            except:
                nLayer_next = None

            if line_next_fields is None:
                last = True
            elif chip[0] != chip_next[0] or chip[1] != chip_next[1]:
                last = True

            if nLayer_next is not None:
              if nLayer_next < nLayer:
                BraggPeak = 1.0

            f.seek(last_pos)
            
            # To be used later...
            cS = int(line_fields[6])
            

            # # Drop the hits with very small size
            if cS > 2 and float(line_fields[5])!=0.0:

              # if nhits > aD.maxhits:
              #   continue

              sumEdep += float(line_fields[5])

              label = np.array([
                  float(line_fields[2])/float(aD.maxx),
                  float(line_fields[3])/float(aD.maxy),
                  # rescaler(float(line_fields[2]), 0., 1., 0., aD.maxx),
                  # rescaler(float(line_fields[3]), 0., 1., 0., aD.maxy),
                  
                  rescaler(float(line_fields[5]), 0., 1., 0., aD.maxedep),
                  rescaler(sumEdep, 0., 1., 0., aD.maxsumedep),
                  BraggPeak
              ])

              BraggPeak = 0.1
              flabels[nhits] = label
              nhits += 1
              # flabels.append(label)

              hits = line_fields[7:]
              # print("Data:")
              # print(hits)

              for c in zip(hits[::2], hits[1::2]):
                  # frame[int(c[0]), int(c[1])] = 255
                  
                  # if iPixel > aD.maxpixels:
                  #   continue
                  
                  frame[iPixel] = np.array([
                    float(c[0])/float(aD.maxx),
                    float(c[1])/float(aD.maxy),
                    # rescaler(float(c[0]), 0., 1., 0., aD.maxx),
                    # rescaler(float(c[1]), 0., 1., 0., aD.maxy),
                    rescaler(float(nLayer), 0., 1., 0., aD.maxlayers),
                  ])
                  iPixel+=1

              npixels += cS

            if last:
                if nhits > maxhits:
                    maxhits = nhits
                    whichIm = iIm
                    whichLn = i
                    # img = Image.fromarray(frame, 'L')
                    # img.save("outputs/"+str(i)+".png")
                maxpixels = max(maxpixels, npixels)
                maxSumEdep = max(maxSumEdep, sumEdep)
                nhits = 0
                iPixel = 0
                npixels = 0
                sumEdep = 0
                iIm+=1
                if not aD.scanonly:
                  for ii in range(10):
                    if frame[ii][0] != 0.0 and frame[ii][1] != 0.0:
                      # print("Not empty 1")
                      empty = False
                    if flabels[ii][0] != 0.0 and flabels[ii][1] != 0.0 and flabels[ii][2] != 0.0:
                      # print("Not empty 2")
                      empty = False

                  if not empty:
                    
                    frames.append(frame)
                    labels.append(flabels)
    if aD.verbose:                
      print("Maximum number of hits on one chip: %d at im %d, line %d (from a total of %d images)" % (maxhits, whichIm, whichLn, iIm))
      print("Maximum number of active pixels on one chip: %d" % (maxpixels))
      print("Maximum sumEdep on one chip: %lf" % (maxSumEdep))
    return frames, labels

def splitData(inputs, outputs, aD, shuffle=False):
  x_train = []
  y_train = []
  x_valid = []
  y_valid = []

  if shuffle:
    data = list(zip(inputs, outputs))
    rn.shuffle(data)
    inputs, outputs = zip(*data)

    inputs = list(inputs)
    outputs = list(outputs)

  if len(inputs) != len(outputs):
    print("Fatal error: lengths of inputs and outputs don't match!")
    print("%d\t\t%d" % (len(inputs), len(outputs)))


  for i in range(len(inputs)):
    if i < aD.ratio*len(inputs):
      x_train.append(inputs[i])
      y_train.append(outputs[i])
    else:
      x_valid.append(inputs[i])
      y_valid.append(outputs[i])

  x_train = np.array(x_train)
  y_train = np.array(y_train)
  x_valid = np.array(x_valid)
  y_valid = np.array(y_valid)

  if aD.verbose:
    print("The ratio used for the split: %.2f" %aD.ratio)
    print("#x-train\t#y-train\t#x-valid\t#y-valid")
    print("%d\t\t%d\t\t%d\t\t%d" % (len(x_train), len(y_train), len(x_valid), len(y_valid)))


  del inputs
  del outputs

  return x_train, y_train, x_valid, y_valid


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:


            z_mean, z_log_var, z = self.encoder(x)
            y_pred = self.decoder(z)
            # pred_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # pred_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.binary_crossentropy(y, y_pred), axis=(1,2)
            #     )
            # )
            pred_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(layers.Flatten()(y), layers.Flatten()(y_pred))))

            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = pred_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(pred_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
      x, y = data

      z_mean, z_log_var, z = self.encoder(x, training=False)
      y_pred = self.decoder(z, training=False)


      pred_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
      kl_loss = -0.5 * (1 + z_log_var -
                        tf.square(z_mean) - tf.exp(z_log_var))
      kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
      total_loss = pred_loss + kl_loss
      
      # Update the metrics.
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(pred_loss)
      self.kl_loss_tracker.update_state(kl_loss)
      self.compiled_metrics.update_state(y, y_pred)
      # Return a dict mapping metric names to current value.
      # Note that it will include the loss (tracked in self.metrics).
      return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
      # print("Now predicting")
      # print(inputs.shape)
      z_mean, z_log_var, z = self.encoder(inputs, training=False)
      y_pred = self.decoder(z, training=False)
      # print("The pred:")
      # print(y_pred.shape)
      # print(y_pred)
      return np.array(y_pred[0])



def _main_(args): 

  if args.noGPU:
    print("TODO.")
  if args.GPU:
    print("Tensorflow version:")
    print(tf.__version__)
    print("Keras version:")
    print(keras.__version__)
    # print("Trying to setup GPU environment...")
    nGPU = len(tf.config.list_physical_devices('GPU'))
    if args.verbose:
      print("Available devices:")    
      print(device_lib.list_local_devices())
      print("Number of GPUs Available: ", nGPU)
    if nGPU > 0:
      print("There are available GPU(s), it is already the default device to use.")
    else:
      print("There are no available GPU :'(")
    # config = tf.ConfigProto( device_count = {'GPU': 1} ) 
    # sess = tf.Session(config=config) 
    # keras.backend.set_session(sess)

  print("Parsing and loading training data...")  
  aD = getMax(args)

  aD.show()


  inputs, outputs = loadData(aD)
  print("Done.")

  if aD.verbose:
    print("Lenght of inputs and outputs: %d %d" % (len(inputs), len(outputs)))
    for i in range(26):
      print("   ", i, outputs[0][i])
      print("Rescaled:")
      print("    ", rescaler(outputs[0][i][0], 0., aD.maxx, 0., 1.))
      print("    ", rescaler(outputs[0][i][1], 0., aD.maxy, 0., 1.))
      print("    ", rescaler(outputs[0][i][2], 0., aD.maxedep, 0., 1.))
      print("    ", rescaler(outputs[0][i][3], 0., aD.maxsumedep, 0., 1.))
      print("    ", outputs[0][i][4])
      print()

  if aD.scanonly:
    exit(1)

  print("Splitting data into train/validation sets...")

  x_train, y_train, x_valid, y_valid = splitData(inputs, outputs, aD, shuffle=False)
  print("Done.")


  if args.modelNameIn is None:
    print("Setup the model...")
    mF = mD.modelFactory(latent_dim=args.latent_dim)

    # encoder = mF.getEncoder(aD)
    # decoder = mF.getDecoder(aD)
    # vae = VAE(encoder, decoder)
    
    dnn = mF.getDNN(aD)

    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # vae.compile(
    dnn.compile(
      optimizer=opt,
      loss='MeanSquaredError',
      metrics=['accuracy', metrics.MeanSquaredError()]
      )
    print("Start training...")
    # vae.fit(x=x_train, y=y_train, epochs=args.epochs,
    dnn.fit(x=x_train, y=y_train, epochs=args.epochs,
            batch_size=args.batch, use_multiprocessing=True, workers=8)

    if not args.nosave:
      # vae.encoder.save("./models_local/"+args.fileNameOut+"_encoder")
      # vae.decoder.save("./models_local/"+args.fileNameOut+"_decoder")
      dnn.save("./models_local/"+args.fileNameOut+"_dnn")
      # print("Model saved into folder ./models_local/"+args.fileNameOut+"_[encoder, decoder]")
      print("Model saved into folder ./models_local/"+args.fileNameOut+"_[dnn]")

    modelName = args.fileNameOut
    print("Done.")
  else:
    print("Loading saved model %s" % (args.modelNameIn))
    # encoder = keras.models.load_model("./models_local/"+args.modelNameIn+"_encoder", compile=False)
    # decoder = keras.models.load_model("./models_local/"+args.modelNameIn+"_decoder", compile=False)
    # vae = VAE(encoder, decoder)
    
    dnn = keras.models.load_model("./models_local/"+args.modelNameIn+"_dnn", compile=False)
    
    modelName = args.modelNameIn

    # For evaluation, the model needs to be compiled...
    # vae.compile(
    dnn.compile(
      loss='MeanSquaredError',
      metrics=['accuracy', metrics.MeanSquaredError()]
      )
    # vae.compile()

  if args.eval:
    print("Evaluation: ")
    dnn.evaluate(x_valid, y_valid)
    # vae.evaluate(x_valid, y_valid)

  if args.tests:


    print("Performing tests...")

    # TEST 1

    # x_test_encoded = vae.encoder.predict(x_valid, batch_size=args.batch)
    # fig, ax = plt.subplots(1,1, figsize=(6, 5))

    # x_test_encoded = np.array(x_test_encoded)
    # print(x_test_encoded.shape)
    # print(x_test_encoded.shape[0])
    # print(x_test_encoded)
    # print(y_valid)
    # print(y_valid.ravel(order='C')[3::5])
    # sp = ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
    # # fig.colorbar(sp)
    # fig.tight_layout()
    # fig.savefig("plots/latentspace.png", transparent=False, bbox_inches='tight')
    # plt.close(fig)

    # TEST 2
    # print(x_train.shape)
    print(x_valid.shape)
    print(np.expand_dims(x_valid[0], axis=0).shape)

    # for j in range(3):
    #   for i in range(len(x_valid[j])):
    #     print(j, i, x_valid[j][i])
    # for j in range(3):
    #   for i in range(len(y_valid[j])):
    #     print(j, i, y_valid[j][i])

    pred = []
    with alive_bar(len(x_valid)) as bar:
      for frame in x_valid:
        bar()
        # print(np.expand_dims(frame, axis=0).shape)
        pred.append(dnn(np.expand_dims(frame, axis=0))[0])
        # pred.append(vae(np.expand_dims(frame, axis=0)))
    pred = np.array(pred)
    print(len(pred), len(y_valid))
    # for i in pred:
    #   print(i.shape)
    #   print(i)
    print(pred.shape)
    
    # Plot the not rescaled outputs
    fig, ax = plt.subplots(1,2,figsize=(11,5))

    # print(y_valid)
    # print(y_valid.ravel(order='C'))
    # print(y_valid.ravel(order='C')[0::3])
    ax[0].scatter(
      [y_valid.ravel(order='C')[0::5]],
      [y_valid.ravel(order='C')[1::5]], 
      label='Truth', s=1)
    # ax[0].scatter([y_valid[i][0] for i in range(len(y_valid))],[y_valid[i][1] for i in range(len(y_valid))], label='Truth', s=2)
    ax[0].scatter(
      [pred.ravel(order='C')[0::5]], 
      [pred.ravel(order='C')[1::5]], 
      label='Prediction', s=1)

    ax[1].scatter(
      [y_valid.ravel(order='C')[2::5]],
      [pred.ravel(order='C')[2::5]], 
      label='Model: '+modelName, c='r', s=1)

    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend(title="Hit position")
    ax[0].grid(b=True, which='major', color='gray', linestyle='--', alpha=0.4)
    ax[1].set_xlabel('Ground truth')
    ax[1].set_ylabel('Prediction')
    ax[1].legend(title="$\sum E_{dep}$")
    ax[1].grid(b=True, which='major', color='gray', linestyle='--', alpha=0.4)
    ax[1].axis(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0)
    fig.tight_layout()
    fig.savefig("plots/predictions.png", transparent=False, bbox_inches='tight')
    plt.close(fig)

    # TEST 3
    
    # Plot the distances
    fig, ax = plt.subplots(1,1,figsize=(6,5))

    ax.hist(
      [
        np.sqrt((y_valid.ravel(order='C')[0::5]-pred.ravel(order='C')[0::5])**2+
        (y_valid.ravel(order='C')[1::5]-pred.ravel(order='C')[1::5])**2)
      ], 
      bins=50,
      # label='Truth'
      )
    # ax[0].scatter([y_valid[i][0] for i in range(len(y_valid))],[y_valid[i][1] for i in range(len(y_valid))], label='Truth', s=2)
    
    ax.set_xlabel('$D_{xy}$')
    ax.set_ylabel('Prob')
    ax.set_yscale('log')
    # ax.legend(title="Hit position")
    ax.grid(b=True, which='major', color='gray', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig("plots/distances.png", transparent=False, bbox_inches='tight')
    plt.close(fig)

    # TEST 4

    print("The first 50 elements of the first 5 frame:")

    for fr in range(5):
      print("Frame: %d" % fr)
      print("#n\tx_val  x_pred\t\ty_val  y_pred\t\tEdep_val Edep_pred\tEsum_val Esum_pred\tBragg_val Bragg_pred")
      for i in range(50):
        print("%d\t%.4f  %.4f\t\t%.4f  %.4f\t\t%.4f  %.4f\t\t%.4f  %.4f\t\t%.1f\t%.3f" % (i, 
        y_valid[fr][i][0],
        pred[fr][i][0],
        y_valid[fr][i][1],
        pred[fr][i][1],
        y_valid[fr][i][2],
        pred[fr][i][2],
        y_valid[fr][i][3],
        pred[fr][i][3],
        y_valid[fr][i][4],
        pred[fr][i][4]
        ))



def acknowledgeMessage():
  print()
  print("``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='````'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_")
  print()
  print("Variational autoencoder for pCT data.")
  print()
  print("The research was supported by the Ministry of Innovation and Technology NRDI Office")
  print("within the framework of the MILAB Artificial Intelligence National Laboratory Program.")
  print()
  print("Author: Gábor Bíró (biro.gabor@wigner.hu)                                             ")
  print()
  print("``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,2021```'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_")
  print()


if __name__ == '__main__':

    # aaa = [[[1,2,3], [4,5,6], [7,8,9]], [['a','b','c'], ['d','e','f'], ['g','h','i']]]
    # aaa = np.array(aaa)
    # print(aaa)
    # print(aaa.ravel(order='C'))
    # print([j for j in aaa.ravel(order='C')[::3]])

    acknowledgeMessage()

    args = argparser.parse_args()
    _main_(args)

    
