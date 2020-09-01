import pandas as pd
import argparse
import os
import random
import os.path
import numpy as np
import re
import tensorflow as tf
from sklearn.utils import class_weight
import argparse
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt


""" This program assumes that the csv files have 
been divided according to the camera channels or angles and the action
classes 
"""

parser = argparse.ArgumentParser(description='LSTM train/test pipeline')
parser.add_argument('--dir', help='Path to directory to all datasets')
parser.add_argument('--train', help='Path to train set')
parser.add_argument('--test', help='Path to test set')
parser.add_argument('--validation',help='Path to validation set')
parser.add_argument('--name',help="set name of model")
parser.add_argument('--save', help='Path to folder to save model')
args = parser.parse_args()

# Set a seed value
seed_value= 1238 # 1. Set `PYTHONHASHSEED` environment variable at a fixed value before:1232,1235
os.environ['PYTHONHASHSEED']=str(seed_value) # 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value) # 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value) # 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

def process_csv(csv, label):
    temp = csv.values
    data = temp.astype(float)
    x = []
    y = []
    if len(data) < 40:
        print("less than 40 frames")
        pad = create_pad()
        # print("pad",pad)
        data = pad_sequence(data, pad)
        # print("data",data)
        x.append(data)
        # print(np.array(x).shape)
    elif len(data) >= 40:
        # print("greater than 40 frames")
        for i in range(40, len(data)):
            x.append(data[i - 40:i])
    print(np.array(x).shape)
    n = len(x)  # length will be 1 for shorter frames as only 1 sequence of 40 can be made after padding
    print("n", n)
    y = create_labels(n, label)
    return x, y


def create_labels(n, lab):
    Y = []
    if lab == 0:
        for j in range(n):
            Y.append([0])
    elif lab == 1:
        for j in range(n):
            Y.append([1])
    elif lab == 2:
        for j in range(n):
            Y.append([2])
    elif lab == 3:
        for j in range(n):
            Y.append([3])
    elif lab == 4:
        for j in range(n):
            Y.append([4])
    return Y


def pad_sequence(data, pad):
    print("inside pad sequence")
    # print(data)
    # print("pad",pad)
    if len(data) < 40:
        gap = 40 - len(data)
        # print("spread: ",gap)
        for i in range(gap):
            data = np.vstack((data, pad))
    return data


def create_pad():
    pad = []
    for i in range(0, 36):
        pad.append(0.0)
    return pad


def one_hot_encoder(data):
    one_hot_y = tf.one_hot(data.astype(np.int32), depth=5, axis=1, dtype=tf.int32)
    return one_hot_y


def load_data(directory):
    X = []
    Y = []
    for dir_path, dir_names, files in os.walk(directory):
        for file in files:
            if re.search("sit", file):
                label = 0
            elif re.search("stand", file):
                label = 1
            elif re.search("walk", file):
                label = 2
            elif re.search("bend", file):
                label = 3
            elif re.search("fall", file):
                label = 4

            path = dir_path + "/" + file
            print(path)
            print("label", label)
            csv = pd.read_csv(path, header=None, encoding='utf-7')
            csv.drop(csv.index[0], inplace=True)
            if csv.empty:
                print("empty: " + path)
                break
            if len(csv.columns) > 36:
                for i in range(36, len(csv.columns)):
                    csv.drop([i], axis=1, inplace=True)
            is_NaN = csv.isnull()
            if any(is_NaN.any(axis=1)) == True:
                print(path)
            csv.fillna(0.0, inplace=True)
            temp_x, temp_y = process_csv(csv, label)
            X.extend(temp_x)
            Y.extend(temp_y)
    return X, Y


# ch1, ch2, ch3, ch4
# fall, sit, walk, bend, stand

file_names = [args.train, args.test, args.validation]
directory = [args.dir + i for i in file_names]
label = 0
X_train, Y_train = load_data(directory[0])
X_test, Y_test = load_data(directory[1])
X_val, Y_val = load_data(directory[2])

# convert into numpy arrays from list
X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
Y_val = np.array(Y_val)
Y_train_1 = Y_train
Y_test_1 = Y_test
Y_val_1 = Y_val

# for imbalanced class otherwise comment out
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train.reshape(-1))
class_weight = {0: class_weight[0], 1: class_weight[1], 2: class_weight[2], 3: class_weight[3], 4: class_weight[4]}

# To get vector like [0 0 0 1 0] results in Y
Y_train = np.array(one_hot_encoder(Y_train)).reshape(-1, 5).astype(np.int32)
Y_test = np.array(one_hot_encoder(Y_test)).reshape(-1, 5).astype(np.int32)
Y_val = np.array(one_hot_encoder(Y_val)).reshape(-1, 5).astype(np.int32)

print("Train data shape",X_train.shape)
print(Y_train.shape)
print("Test data shape",X_test.shape)
print(Y_test.shape)
print("Validation data shape",X_val.shape)
print(Y_val.shape)

check = [X_train, X_test, X_val]

for i in range(len(check)):
  print(np.isnan(check[i]).any())
  where_are_NaNs = np.isnan(X_train).any()
  print(where_are_NaNs)

# counting number of samples

sit = 0
stand = 0
walk = 0
bend = 0
fall = 0
y_three = [Y_train,Y_test,Y_val]
for i in range(len(y_three)):
    y = y_three[i]
    for i in range(len(y)):
        if y.argmax(1)[i] == 0:
            sit+=1
        elif y.argmax(1)[i] == 1:
            stand+=1
        elif y.argmax(1)[i] == 2:
            walk+=1
        elif y.argmax(1)[i] == 3:
            bend+=1
        elif y.argmax(1)[i] == 4:
            fall+=1


print("sit: ",sit)
print("stand: ",stand)
print("walk: ",walk)
print("bend: ",bend)
print("fall: ",fall)

# LSTM model using tensorflow keras API

# set the values of parameters
epochs = 150
batch_size = 64
time_steps = 40 # a.k.a, length of sequence
num_features = 36
num_output = 5 # output size depends on number of classes
hidden_units = 36 # output dimensionality of each LSTM layer

def build_model(time_steps, n_features, units):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Masking(mask_value=0.,input_shape=(time_steps,n_features)),
            tf.keras.layers.LSTM(units,recurrent_dropout=0.8, return_sequences=True,input_shape=(time_steps, n_features)),
            # tf.keras.layers.RNN(tf.keras.layers.LSTMCell(1,dropout=0.2),return_sequences=True),
            # tf.keras.layers.LSTM(units,recurrent_dropout=0.8, return_sequences=True),
            # tf.keras.layers.LSTM(units,recurrent_dropout=0.8,return_sequences=True),
            # tf.keras.layers.Dense(units, activation='relu'),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(5, activation='softmax')
            tf.keras.layers.LSTM(num_output,activation='softmax')
            # tf.keras.layers.GRU(num_output,activation='softmax')
        ])

    return model

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)
LSTM_model = build_model(time_steps, num_features, hidden_units)
LSTM_model.summary()
LSTM_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.001),
                   loss='categorical_crossentropy',
                   metrics=['categorical_crossentropy',
                     'accuracy'])

history = LSTM_model.fit(X_train, Y_train, batch_size= batch_size, class_weight=class_weight,
                         validation_data=(X_val,Y_val),validation_freq =1,
                         validation_batch_size= 32 ,epochs= epochs,shuffle= True)


# plot the accuracy and loss of the model across epochs


plt.plot(history.epoch, history.history['loss'], linestyle="--")
plt.plot(history.epoch, history.history['accuracy'] )

plt.title('Model training results')
plt.ylabel('loss or accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_loss', 'Train_accuracy'], loc='upper left')
plt.show()

plt.plot(history.epoch, LSTM_model.history.history['val_loss'], linestyle="--")
plt.plot(history.epoch, LSTM_model.history.history['val_accuracy'] )

plt.title('Model training results')
plt.ylabel('loss or accuracy ')
plt.xlabel('Epoch')
plt.legend(['val_loss', 'val_accuracy'], loc='upper left')
plt.show()

plt.plot(history.epoch, history.history['categorical_crossentropy'])
plt.title('Model categorical crossentropy')
plt.ylabel('progress')
plt.xlabel('Epoch')
plt.legend(['binary_crossentropy'],loc = 'upper left')
plt.show()

keras.utils.plot_model(LSTM_model, 'multi_input_and_output_model.png', show_shapes=True)

#Testing LSTM

results = LSTM_model.evaluate(X_test, Y_test, batch_size=1)
print('Test results: ',results)

pred = LSTM_model.predict(X_test)
n_classes = 5

LABELS = [
    "SIT",
    "STAND",
    "WALK",
    "BEND",
    "FALL"
    ]

precision, recall, f_score, support = metrics.precision_recall_fscore_support(Y_test.argmax(1), pred.argmax(1),average="weighted")

print("precision",100 * precision)
print("recall", 100 * recall)
print("f_score",100 * f_score)

print("number of occurrences of each class in test data", support)
print("confusion matrix")
confusion_matrix_basic = metrics.confusion_matrix(Y_test.argmax(1), pred.argmax(1))
print(confusion_matrix_basic)

cm = confusion_matrix(  Y_test.argmax(1), pred.argmax(1) )
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j, i, format(cm[i, j], fmt),
  horizontalalignment="center",
  color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# save model

path = args.save
name = args.name
LSTM_model.save_weights(path)
LSTM_model.save(path + name + '.h5' )

