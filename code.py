import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("dataset_1000.csv")   #or "dataset_5000.csv" for larger and more ambiguous cases

X = df.drop("Label", axis=1)   # all feature columns
y = df["Label"]                # target column (0 or 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binary output

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train,epochs=50,batch_size=16,validation_split=0.2,verbose=0)

loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

probabilities = model.predict(X_test)
predictions = (probabilities >= 0.5).astype(int)

np.set_printoptions(precision=6, suppress=True)

print("\nFirst 10 Probabilities:")
print(probabilities[:10])

print("\nFirst 10 Predicted Classes:")
print(predictions[:10])

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))
