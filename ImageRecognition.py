import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

from keras.datasets import cifar10

# Set random seed for purposes of reproducibility
seed = 21

# loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

#  Create a model. Sequential is most common
model = Sequential()

# Adding the convolutional layer with
# 1. 32 Filters/Channels
# 2. Size of filter is (3X3)
# 3. Input shape
# 4. Padding, 'same' in this scenario
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))

# Adding activation function `relu`, most commonly used
model.add(Activation('relu'))

# or you can do the below:
# model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

# Now we will make a dropout layer to prevent overfitting, which functions by randomly eliminating
# some of the connections between the layers (0.2 means it drops 20% of the existing connections)
model.add(Dropout(0.2))

# Batch Normalization normalizes the inputs heading into the next layer,
# ensuring that the network always creates activations with the same distribution
model.add(BatchNormalization())

# Here's the pooling layer, as discussed before this helps make the image classifier
# more robust so it can learn relevant patterns.
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# You can now repeat these layers to give your network more representations to work off
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Flatten the data
model.add(Flatten())
model.add(Dropout(0.2))

# Now we make use of the Dense import and create the first densely connected layer.
# We need to specify the number of neurons in the dense layer. Note that the numbers
# of neurons in succeeding layers decreases, eventually approaching the same number of
# neurons as there are classes in the dataset (in this case 10). The kernel constraint
# can regularize the data as it learns, another thing that helps prevent overfitting.
# This is why we imported maxnorm earlier.
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# In this final layer, we pass in the number of classes for the number of neurons.
# Each neuron represents a class, and the output of this layer will be a 10 neuron
# vector with each neuron storing some probability that the image in question belongs
# to the class it represents.
model.add(Dense(class_num))

# Finally, the softmax activation function selects the neuron with the highest
# probability as its output, voting that the image belongs to that class.
model.add(Activation('softmax'))

# Number of epochs
# The optimizer is what will tune the weights in your network to approach the
# point of lowest loss. The Adam algorithm is one of the most commonly used
# optimizers because it gives great performance
epochs = 25
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

# all we have to do is call the fit() function on the model and pass in the chosen parameters.
numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Save the model
model.save('models/trained/cifar10_04-08-2020.h5')
