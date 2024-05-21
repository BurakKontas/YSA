import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

dataset = pd.read_csv("data.csv")

dataset = dataset.drop(['Salary', 'Salary Currency'], axis=1)

X = dataset.drop('Salary in USD', axis=1)
Y = dataset['Salary in USD']

categorical_features = ['Job Title', 'Employment Type', 'Experience Level', 'Expertise Level', 'Company Location', 'Employee Residence', 'Company Size', 'Year']

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    X[feature] = label_encoders[feature].fit_transform(X[feature])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

x_sc = StandardScaler()
X_train = x_sc.fit_transform(X_train)
X_test = x_sc.transform(X_test)

y_sc = StandardScaler()

Y_train = np.array(Y_train).reshape(-1, 1)
Y_train = y_sc.fit_transform(Y_train)

Y_test = np.array(Y_test).reshape(-1, 1)
Y_test = y_sc.transform(Y_test)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)

def train(learning_rate):

    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.Dense(units=15, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=30, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=50, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=30, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=15, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=1, activation="linear"))

    ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_absolute_error" , metrics=["mean_absolute_error"])

    history = ann.fit(X_train, Y_train, batch_size=512, epochs=5000, validation_data=(X_test, Y_test, ), callbacks=[early_stopping])
    
    return ann, history


def calculate_r_squared(ann):
    y_pred = ann.predict(X_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(Y_test, y_pred)
    return r2


learning_rates = [0.00000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
negative_learning_rates = [-x for x in learning_rates]
learning_rates = negative_learning_rates + learning_rates
learning_rates = learning_rates * 50
learning_rates.sort(reverse=True)

df = pd.DataFrame(columns=["learning_rate", "r_squared", "stopped_epoch", "best_epoch", "best_val_loss", "best_train_loss", "mean_val_loss", "mean_train_loss", "time_taken"])

for lr in learning_rates:
    print(f"Training with learning rate: {lr}, index: {learning_rates.index(lr)}, left: {len(learning_rates) - learning_rates.index(lr)}")
    startTime = time.time()
    ann, history = train(lr)
    timeTaken = time.time() - startTime
    print(f"Training took {timeTaken} seconds")
    r_squared = calculate_r_squared(ann)
    
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = history.history['val_loss'][best_epoch - 1]
    best_train_loss = history.history['loss'][best_epoch - 1]
    
    mean_val_loss = np.mean(history.history['val_loss'])
    mean_train_loss = np.mean(history.history['loss'])
    
    df = pd.concat([df, pd.DataFrame({
        "learning_rate": [format(lr, '.10f')],
        "r_squared": [r_squared],
        "stopped_epoch": [len(history.history['val_loss'])],
        "best_epoch": [best_epoch],
        "best_val_loss": [best_val_loss],
        "best_train_loss": [best_train_loss],
        "mean_val_loss": [mean_val_loss],
        "mean_train_loss": [mean_train_loss],
        "time_taken": [timeTaken]
    })], ignore_index=True)
    
    df.to_csv("r_squared.csv", index=False)    
    print(f"Learning rate: {lr} - R^2: {r_squared}, Stopped epoch: {len(history.history['val_loss'])}, Best epoch: {best_epoch}, Best val loss: {best_val_loss}, Best train loss: {best_train_loss}, Mean val loss: {mean_val_loss}, Mean train loss: {mean_train_loss}")
    
