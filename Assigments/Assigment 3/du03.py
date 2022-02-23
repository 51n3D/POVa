# To plot graphs.
import numpy as np

import sys

from tools import collage
from tools import readCIFAR
from matplotlib import pyplot as plt

# Example showing how to train and use a Convolutional Neural Network in TensorFlow (keras).
#
# You need OpenCV, TensorFlow and matplotlib
# Install the needed libraries by:
# pip install tensorflow matplotlib opencv-python
# If you have a cuda-capable GPU, install tensorflow-gpu instead of tensorflow to make training
# much faster.
#
# Get the dataset first by:
# cd ./data
# ./downloadCIFAR.sh
#
# Feel free to experiment with the network to reach better accuracy (not on merlin, please).
# (But submit the requested network.)
# It is possible to get ~92% accuracy using larger network of similar arch. with proper regularization and more epochs.
# To compare to state of the art results on CIFAR-10 dataset, look at:
# http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130

# Define the network.
# Input: batch of images 32x32x3
# Use ReLU nonlinearities after each layer with optimized (trained) parameters (except the last one).
# Use "valid" convolutions (no padding).
# Layers:
#   Convolution with 8 3x3 filters
#   Max-pooling with step 2 and pooling area 2x2
#   Convolution with 16 3x3 filters
#   Max-pooling with step 2 and pooling area 2x2
#   Fully connected layer with 256 neurons
#   Dropout with probability 15%
#   Fully connected layer with 256 neurons
#   Dropout with probability 15%
#   Fully connected layer with 10 neurons and softmax activation
# The last layer will produce probabilities for the 10 classes in CIFAR-10 and
# it is the output of the model.
def build_simple_network():
    # These are the layers you need for the network.
    # Documentation is at https://www.tensorflow.org/api_docs/python/tf/keras/layers
    #
    # You can build a sequential model which is simple but restricts the
    # network to single input and single output.
    # This is for people who do not understand neural networks at all :).
    # https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    #
    # Or you can use functional API to build the network which is more
    # flexible and explicitly specifies connections between layers.
    # This is the preferred way.
    # https://www.tensorflow.org/guide/keras/functional
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.models import Model

    img_inputs = Input(shape=(32, 32, 3))
    conv2d_1 = Conv2D(8, (3, 3), padding='valid', activation='relu')
    pool2d_1 = MaxPooling2D(strides=2, pool_size=(2, 2))
    conv2d_2 = Conv2D(16, (3, 3), padding='valid', activation='relu')
    pool2d_2 = MaxPooling2D(strides=2, pool_size=(2, 2))
    flatten = Flatten()
    dense_1 = Dense(256, activation='relu')
    dropout_1 = Dropout(.15)
    dense_2 = Dense(256, activation='relu')
    dropout_2 = Dropout(.15)
    dense_3 = Dense(10, activation='softmax')

    # layer 1
    x = conv2d_1(img_inputs)
    x = pool2d_1(x)
    # layer 2
    x = conv2d_2(x)
    x = pool2d_2(x)
    x = flatten(x)
    # layer 3
    x = dense_1(x)
    x = dropout_1(x)
    # layer 4
    x = dense_2(x)
    x = dropout_2(x)    
    # layer 5
    outputs = dense_3(x)

    model = Model(inputs=img_inputs, outputs=outputs)
    return model


# Get the dataset first by:
# cd ./data
# ./downloadCIFAR.sh
def prepareData(downsample=1):
    # This reads the dataset.
    trnData, tstData, trnLabels, tstLabels = readCIFAR(
        './data/cifar-10-batches-py')
    print('\nDataset tensors')
    print('Training shapes: ', trnData.shape, trnLabels.shape)
    print('Testing shapes: ', tstData.shape, tstLabels.shape)
    print()

    # Convert images from RGB to BGR
    trnData = trnData[::downsample, :, :, ::-1]
    tstData = tstData[::downsample, :, :, ::-1]
    trnLabels = trnLabels[::downsample]
    tstLabels = tstLabels[::downsample]

    # Normalize data
    # This maps all values in trn. and tst. data to range <-0.5,0.5>.
    # Some kind of value normalization is preferable to provide
    # consistent behavior accross different problems and datasets.
    trnData = trnData.astype(np.float32) / 255.0 - 0.5
    tstData = tstData.astype(np.float32) / 255.0 - 0.5
    return trnData, tstData, trnLabels, tstLabels


def main():

    model = build_simple_network()
    print('Model summary:')
    model.summary()

    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras import metrics
    # Use SparseCategoricalCrossentropy loss and Adam optimizer with learning rate 0.001.
    # All the imports you need are provided above.
    model.compile(
        loss=losses.sparse_categorical_crossentropy,
        optimizer=optimizers.Adam(lr=0.001),
        metrics=[metrics.sparse_categorical_accuracy])

    trnData, tstData, trnLabels, tstLabels = prepareData()

    # Show first 144 images from each set.
    trnCollage = collage(trnData[:144] + 0.5)
    tstCollage = collage(tstData[:144] + 0.5)
    plt.imshow(trnCollage)
    plt.title('Training data')
    plt.show()
    plt.imshow(tstCollage)
    plt.title('Testing data')
    plt.show()

    # Train the network for 5 epochs on mini-batches of 64 images.
    model.fit(
        x=trnData, y=trnLabels,
        batch_size=64, epochs=5, verbose=1,
        validation_data=(tstData, tstLabels), shuffle=True)

    # Save the network:
    model.save('model.h5')

    # Compute network predictions for the test set and show results.
    print('Compute model predictions for test images and display the results.')

    dataToTest = tstData[::20]

    # Compute network (model) responses for dataToTest inputs.
    # This should produce a 2D tensor of the 10 class probabilites for each
    # image in dataToTest. The subsequent code displays the predicted classes.
    classProb = model.predict(dataToTest)

    print('Prediction shape:', classProb.shape)

    # These are the classes as defined in CIFAR-10 dataset in the correct order
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

    # Get the most probable class for each test image.
    predicted_classes = np.argmax(classProb, axis=1)
    for i in range(classProb.shape[1]):
        # Get all images assigned to class "i" and show them.
        class_images = dataToTest[predicted_classes == i]
        if class_images.shape[0]:
            class_collage = collage(class_images)
            title = 'Predicted class {} - {}'.format(i, classes[i])
            plt.imshow(class_collage + 0.5)
            plt.title(title)
            plt.show()

    print('Evaluate network error outside of training on test data.')
    loss, acc = model.evaluate(x=tstData, y=tstLabels, batch_size=64)
    print()
    print('Test loss', loss)
    print('Test accuracy', acc)


if __name__ == "__main__":
    main()
