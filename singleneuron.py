import numpy as np
import tensorflow as tf

# Creating input (Celsius) and output (Fahrenheit) pairs based on F = C * 1.8 + 32
celsius_features = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit_labels = np.array([-40,  14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

# A Sequential model with a single Dense layer containing exactly 1 neuron
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# We use Mean Squared Error to measure the distance between predictions and actual values.
# The Adam optimizer is used to adjust the weights, with a tuned learning rate of 0.1.
model.compile(loss='mean_squared_error', 
              optimizer=tf.keras.optimizers.Adam(0.1))

# Training for 500 epochs (iterations through the dataset)
print("Training the model (making the single neuron...)")
history = model.fit(celsius_features, fahrenheit_labels, epochs=500, verbose=False)
print("Training complete\n")

# Test the Model
test_value = 100.0
# Pass an unseen value (100 Celsius) to see if it can predict 212 Fahrenheit
predicted_f = model.predict(np.array([test_value]), verbose=False)[0][0]
actual_f = test_value * 1.8 + 32

print(f"Testing with {test_value}°C:")
print(f"Neural Network Prediction: {predicted_f:.2f}°F")
print(f"Actual Formula Output:   {actual_f:.2f}°F\n")

# Inspect the Internal Parameters
# A single neuron calculates: Output = (Input * Weight) + Bias
weights = model.get_weights()
print(f"Learned Weight: {weights[0][0][0]:.3f} (Expected ~1.8)")
print(f"Learned Bias:   {weights[1][0]:.3f} (Expected ~32.0)")
