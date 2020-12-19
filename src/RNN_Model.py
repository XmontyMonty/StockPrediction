import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
"""
Used to edit the original data by taking out the date column
df = pd.read_csv('../Data/KGC_StockData.csv')
df['Date'] = df['Date'].apply(lambda date: date[:-9])
"""

df = pd.read_csv('../Data/KGC_StockData_edited.csv', index_col='Date',
                 parse_dates=True)
df.columns = ["Price_in_USD"]
print(df.head())
test_size = 91
test_ind = len(df) - test_size
train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

length = 90
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length,
                                batch_size=1)
n_features = 1

model = Sequential()
model.add(LSTM(100,  input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5)
validation_generator = TimeseriesGenerator(scaled_test, scaled_test,
                                           length=length, batch_size=1)

model.fit_generator(generator, epochs=20, validation_data=validation_generator,
                    callbacks=[early_stop])

model.save("../Models/Model_v1.1.h5")
