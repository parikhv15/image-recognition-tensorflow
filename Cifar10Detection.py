import numpy
import sys

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

from utils import *
from Model import Cifar10Model
from keras.datasets import cifar10

reuseModel = True

imageDimensions = (32,32)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar10Model = Cifar10Model(cifar10.load_data(), labels, imageDimensions)

scores = None
if not reuseModel or not cifar10Model.loadExistingModel():
    cifar10Model.createNewModel()
    scores = cifar10Model.trainModel()
    cifar10Model.model.save('models/trained/cifar10.h5')

if not cifar10Model.model:
    print("Model was not loaded. Please check your code or any other errors occurred.")
    sys.exit(1)

images = getImages("test_data/cifar10/")
image_height, image_width = (32,32)
predictions = {}
for imgPath in images:
    img = load_img(imgPath, target_size=(image_height,image_width))
    img = img_to_array(img) / 255
#     plt.imshow(img)
    img = numpy.expand_dims(img, axis = 0)
    predictions[imgPath] = cifar10Model.model.predict(img)

for imgPath, prediction in predictions.items():
    label_index = numpy.argmax(prediction)
#     print("Prediction: {}".format(prediction))
    print("Image {} is predicted as {}".format(imgPath, labels[label_index]))
