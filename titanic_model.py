import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

train_new = pd.read_csv("data_for_model.csv")


# Daten vorbereiten

X = train_new.drop('Survived', axis = 1)
y = train_new["Survived"]

# Daten aufteilen

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state= 42)

# Modell erstellen

model = Sequential()

model.add(Dense(140, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(70, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test))