import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"


# import necessary modules
import time
import matplotlib.pyplot as plt
import numpy as np
% matplotlib inline
np.random.seed(2017) 
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
import cv2

# load pre-trained model
model = VGG19(weights='imagenet', include_top=True)
# display model layers
model.summary()
# output:
#Total params: 143,667,240
#Trainable params: 143,667,240
#Non-trainable params: 0

# pre-process the image
img = image.load_img('./data/peacock.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
# predict the output 
preds = model.predict(img)
# decode the prediction
pred_class = decode_predictions(preds, top=3)[0][0]
print "Predicted Class: %s"%pred_class[1]
print "Confidance: %s"%pred_class[2]

# --------------------------
# Pre-trained model as a feature extractor
# --------------------------

# load pre-trained model
base_model = VGG19(weights='imagenet')

# pre-process the image
img = image.load_img('./data/peacock.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# define model from base model for feature extraction from fc2 layer
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
# obtain the outpur of fc2 layer
fc2_features = model.predict(img)
print "Feature vector dimensions: ",fc2_features.shape

# output: Feature vector dimensions:  (1, 4096)
