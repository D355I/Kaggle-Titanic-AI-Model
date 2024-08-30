import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd


train_new = pd.read_csv("data_for_model.csv")


# Daten vorbereiten

X = train_new.drop('Survived', axis = 1)
y = train_new["Survived"]

# Daten aufteilen

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state= 42)

# Modell erstellen

model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(450, activation='relu'))
model.add(Dense(450, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation = 'tanh'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=110, batch_size=32,)

print("Genauigkeit Test:")
print(model.evaluate(X_train, y_train))
print("Genauigkeit echt:")
print(model.evaluate(X_test, y_test))

#model.save("model_titanic.h5")
#model.save("model_titanic.keras")