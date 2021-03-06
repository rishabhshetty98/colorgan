from cv2 import cv2
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
from keras.models import load_model
from numpy import expand_dims
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def loader():
    # Don't pre-allocate GPU memory; allocate as-needed
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    cust = {'InstanceNormalization': InstanceNormalization}
    model = load_model('models/gfinal_model_graytocolor_072010.h5', cust)
    return model 


@tf.function
def predict(filename, dirname='images', size=(150, 150)):
    # load and resize the image
    pixels = cv2.imread(dirname + "/" + filename)
    pixels = cv2.resize(pixels, dsize=size)
    image_srcnew = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    onechannel = cv2.cvtColor(image_srcnew, cv2.COLOR_RGB2GRAY)
    threechannel = cv2.cvtColor(onechannel, cv2.COLOR_GRAY2RGB)
    # convert to numpy array
    pixels = np.array(threechannel)
    # transform in a sample
    pixels = expand_dims(pixels, 0)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    tar = model(pixels)
    tar = (tar + 1) / 2.0
    out_name = filename + '_result.jpg'
    out_dir = "colored/" + out_name
    return out_name, out_dir, tar


def saver(out_name,out_dir, tar):
    im = tar
    im = im.numpy()
    outname = out_name.numpy()
    outname = outname.decode('UTF-8')
    outdir = out_dir.numpy()
    outdir = outdir.decode('UTF-8')
    mpimg.imsave(outdir, im[0])
    return outname

model = loader()

