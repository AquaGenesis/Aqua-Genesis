
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset = pd.read_csv('carbon.csv')
x = dataset.drop(columns=['oi'])
y = dataset['oi']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=400)
model.evaluate(x_test, y_test)
print("Enter new data for prediction:")
new_data = []
for feature in x.columns:
    value = float(input(f"Enter {feature}: "))
    new_data.append(value)

new_data = tf.convert_to_tensor([new_data])

prediction = model.predict(new_data)
print(prediction)

if prediction[0] <= 0.5:    
    diagnosis = 'High carbon content detected - release NPK'
else:
    diagnosis = "NPK not required at the present moment"

print(diagnosis)

