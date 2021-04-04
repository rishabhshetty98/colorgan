from cv2 import cv2
import numpy as np
import matplotlib.image as mpimg
from keras.models import load_model
from numpy import expand_dims
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


cust = {'InstanceNormalization': InstanceNormalization}
# model_BtoA = load_model('g_model_colortogray_042090.h5', cust)
model = load_model('models/gfinal_model_graytocolor_072010.h5', cust)


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
    tar = model.predict(pixels)
    tar = (tar + 1) / 2.0
    out_name = filename + '_result.jpg'
    out_dir = "colored/" + out_name
    mpimg.imsave(out_dir, tar[0])
    return out_name


# image_src = load_image('D:/projects/dataset/intel image/seg_pred/seg_pred/182.jpg')


# translate image
# image_tar = model_AtoB.predict(image_src)
# scale from [-1,1] to [0,1]
# image_tar = (image_tar + 1) / 2.0
# convert to img
# print(image_tar.shape)
# plt.imshow(image_tar[0])
# plt.show()
# plt.imsave('your_file.jpeg', image_tar[0])

# new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

# new_arr = ((image_tar +1) * (1/2) * 255)).astype('uint8')
# im = Image.fromarray(new_arr)
# im.save("your_file.jpeg")
