import keras
# Keras - an open-source deep learning framework written in Python
# TensorFlow - an open-source machine learning framework developed by the Google Brain team,
# Others are PyTorch, Deeplearning4j, Apache Spark, Caffee, Chainer, Scikit-learn, Horvoe, Onnx,

from keras.datasets import mnist
# there are many datasets in Keras for learning purposed, and we import one of them, called mnist
# datasets are available at : https://keras.io/api/datasets/


from keras.models import Sequential

from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt


# Activation Function - Softmax, Sigmoid, Relu (Rectified Linear unit), Tanh  -Liner/non-linear


 ## Sigmoid ##: The main reason why we use sigmoid function is because it exists between (0 to 1).
 # Therefore, it is especially used for models where we have to predict the probability as an output.
 # Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice.

 # The softmax function is a more generalized logistic activation function which is used for ##multiclass classification##.

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()


# Preprocessing Data
X_train = X_train.reshape(60000, 784).astype('float32') # features
X_valid = X_valid.reshape(10000, 784).astype('float32') # features

#pixel value is from 0 to 255, o for black and 255 for white
X_train /= 255
X_valid /= 255
#print(X_valid[0])
n_classes = 10;


print(y_valid[6000])
print(y_train[2])



y_train = keras.utils.to_categorical(y_train, n_classes) # labels
y_valid = keras.utils.to_categorical(y_valid, n_classes) # labels

model = Sequential()
#model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax')) # for mutlilabel classification
model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=1, validation_data=(X_valid, y_valid))
