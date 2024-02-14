import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


data = pd.read_csv('ocean2.csv', parse_dates=['year'], dayfirst=True)
data.set_index('year', inplace=True)


data = data.dropna(subset=['pH'])


ph_data = data['pH'].values.astype(float)


scaler = MinMaxScaler(feature_range=(0, 1))
ph_data = scaler.fit_transform(ph_data.reshape(-1, 1))


sequence_length = 10  


X = []
y = []
for i in range(len(ph_data) - sequence_length):
    X.append(ph_data[i:i+sequence_length])
    y.append(ph_data[i+sequence_length])

X = np.array(X)
y = np.array(y)


train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')



history = model.fit(X_train, y_train, epochs=720, batch_size=32, validation_split=0.2)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


last_date = data.index[-1]


future_years = pd.date_range(start=last_date + pd.DateOffset(1), periods=8, freq='Y')
future_predictions = []

input_sequence = ph_data[-sequence_length:].reshape(1, sequence_length, 1)

for _ in range(8):
    predicted_ph = model.predict(input_sequence)
    future_predictions.append(predicted_ph[0, 0])
    input_sequence = np.roll(input_sequence, shift=-1, axis=1)
    input_sequence[0, -1, 0] = predicted_ph[0, 0]


future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

future_ph_df = pd.DataFrame({'year': future_years, 'Predicted pH': future_predictions.flatten()})


original_ph_df = data.copy()

plt.figure(figsize=(12, 6))
plt.plot(original_ph_df.index, original_ph_df['pH'], label='Original pH Data', marker='o')
plt.plot(future_ph_df['year'], future_ph_df['Predicted pH'], label='Predicted pH', marker='x', linestyle='--', color='red')
plt.xlabel('Year')
plt.ylabel('pH Level')
plt.title('Future years pH prediciton based on monthly data')
plt.legend()
plt.grid(True)
plt.show()

print(future_ph_df)