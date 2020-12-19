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

if __name__ == "__main__":
    df = pd.read_csv('../Data/KGC_StockData_edited.csv', index_col='Date',
                     parse_dates=True)

    full_scaler = MinMaxScaler()
    scaled_full_data = full_scaler.fit_transform(df)
    length = 30
    n_features = 1
    generator = TimeseriesGenerator(scaled_full_data, scaled_full_data,
                                    length=length, batch_size=1)
    run_model = input("Run Model Y/N")
    if run_model == "Y":

        model = Sequential()
        model.add(LSTM(100, input_shape=(length, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit_generator(generator, epochs=7)
        model.save("../Models/Model_v2.0.h5")
    else:
        model = load_model("../Models/Model_v2.0.h5")
        forecast = []
        # Replace periods with whatever forecast length you want
        periods = 12

        first_eval_batch = scaled_full_data[-length:]
        current_batch = first_eval_batch.reshape((1, length, n_features))

        for i in range(periods):
            # get prediction 1 time stamp ahead ([0] is for grabbing just the
            # number instead of [array])
            current_pred = model.predict(current_batch)[0]

            # store prediction
            forecast.append(current_pred)

            # update batch to now include prediction and drop first value
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]],
                                      axis=1)
        forecast = full_scaler.inverse_transform(forecast)
