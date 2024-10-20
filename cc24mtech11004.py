# -*- coding: utf-8 -*-
"""
@author: abhay
        CC24MTECH11004
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score

# Function to generate dataset
def generate_dataset(sample_size):
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(-10, 10, sample_size)
    y = np.random.uniform(-10, 10, sample_size)
    z = 3*x**2 - 4*x*y + 2*y**2 + 5
    data = np.column_stack((x, y, z))  # Combine inputs and output
    return data

# Generate dataset
data = generate_dataset(10000)

# Save to Excel
df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
df.to_excel('nonlinear_dataset.xlsx', index=False)

# Read data from Excel
df = pd.read_excel('nonlinear_dataset.xlsx')
X = df[['X', 'Y']].values  # Input features
y = df['Z'].values  # Output target

# Normalize the data between 0 and 1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def create_model(optimizer='adam'):
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(2,)))  # Input layer
    model.add(keras.layers.Dense(32, activation='relu'))  # Hidden layer
    model.add(keras.layers.Dense(1))  # Output layer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Function to plot true vs predicted values
def plot(y_true, y_pred, title, mse=None, r2=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5, label='Predicted')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linewidth=2, label='True Value')
    ax.set_xlabel('True Outputs')
    ax.set_ylabel('Predicted Outputs')
    ax.set_title(title)
    if mse is not None and r2 is not None:
        ax.text(0.1, 0.8, f'MSE: {mse:.4f}\nR²: {r2:.4f}', transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')
    else:
        ax.text(0.1, 0.8, f'R²: {r2:.4f}', transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')
    ax.legend()

# Train the model using Adam optimizer
model_adam = create_model(optimizer='adam')
history_adam = model_adam.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, verbose=0)

# Calculate Mean Squared Error and R² for training data
y_train_pred = model_adam.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print('MSE (Training): ',train_mse)
print('R² (Training): ',train_r2)

# Predict for testing dataset
y_test_pred = model_adam.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)

print('R² (Testing): ',test_r2)

# Plot results for Adam optimizer
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot(y_train, y_train_pred, 'Adam Optimizer - Training Data', mse=train_mse, r2=train_r2, ax=axes[0])
plot(y_test, y_test_pred, 'Adam Optimizer - Testing Data', r2=test_r2, ax=axes[1])
plt.tight_layout()
plt.show()

# Train the model using RMSProp optimizer
model_rmsprop = create_model(optimizer='rmsprop')
history_rmsprop = model_rmsprop.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, verbose=0)

# Calculate MSE and R² for training data
y_train_pred_rmsprop = model_rmsprop.predict(X_train)
mse_train_rmsprop = mean_squared_error(y_train, y_train_pred_rmsprop)
r2_train_rmsprop = r2_score(y_train, y_train_pred_rmsprop)

# Calculate R² for testing data
y_test_pred_rmsprop = model_rmsprop.predict(X_test)
r2_test_rmsprop = r2_score(y_test, y_test_pred_rmsprop)

# Compare results
print('MSE (Training - RMSProp): ',mse_train_rmsprop)
print('R² (Training - RMSProp): ',r2_train_rmsprop)
print('R² (Testing - RMSProp): ',r2_test_rmsprop)

# Plot results for RMSProp optimizer
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot(y_train, y_train_pred_rmsprop, 'RMSProp Optimizer - Training Data', mse=mse_train_rmsprop, r2=r2_train_rmsprop, ax=axes[0])
plot(y_test, y_test_pred_rmsprop, 'RMSProp Optimizer - Testing Data', r2=r2_test_rmsprop, ax=axes[1])
plt.tight_layout()
plt.show()

# Plotting comparison between Adam and RMSProp optimizers for Training Data
plt.figure(figsize=(14, 6))

# Training Data
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Adam Predicted', color='blue')
plt.scatter(y_train, y_train_pred_rmsprop, alpha=0.5, label='RMSProp Predicted', color='orange')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linewidth=2, label='True Value')
plt.xlabel('True Outputs')
plt.ylabel('Predicted Outputs')
plt.title('Training Data Comparison')
plt.text(0.1, 0.8, f'Adam Train MSE: {train_mse:.4f}\nTrain R²: {train_r2:.4f}',  transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')
plt.text(0.1, 0.7, f'RMSprop Train MSE: {mse_train_rmsprop:.4f}\nTrain R²: {r2_train_rmsprop:.4f}',  transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')
plt.legend()

# Testing Data
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Adam Predicted', color='blue')
plt.scatter(y_test, y_test_pred_rmsprop, alpha=0.5, label='RMSProp Predicted', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='True Value')
plt.xlabel('True Outputs')
plt.ylabel('Predicted Outputs')
plt.title('Testing Data Comparison')
plt.text(0.1, 0.8, f'Adam Test R²  : {test_r2:.4f}', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')
plt.text(0.1, 0.7, f'RMSprop Test R²: {r2_test_rmsprop:.4f}', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')
plt.legend()

plt.tight_layout()
plt.show()

# Function to analyze varying parameters with different sample sizes, hidden layers, nodes, epochs and activation function
# This function takes 30min to fully evaluate all the possibilties and plot those whose R² is greater than 0.60
def varying_parameters(hidden_layers, hidden_nodes, epochs_list, sample_sizes, activation_functions):
    for sample_size in sample_sizes:
        # Generate dataset for the current sample size
        data = generate_dataset(sample_size)
        X = data[:, :2]  # Inputs (x, y)
        y = data[:, 2]   # Output (z)

        # Split dataset
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Normalize the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        for layers in hidden_layers:
            for nodes in hidden_nodes:
                for epochs in epochs_list:
                    for activation in activation_functions:
                        model = keras.Sequential()
                        model.add(keras.layers.Dense(nodes, activation=activation, input_shape=(X_train.shape[1],)))
                        for _ in range(layers - 1):
                            model.add(keras.layers.Dense(nodes, activation=activation))
                        model.add(keras.layers.Dense(1))  # Linear activation for output layer
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        
                        # Fit the model with early stopping
                        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])

                        # Batch predictions
                        y_train_pred = model.predict(X_train, batch_size=32)
                        y_test_pred = model.predict(X_test, batch_size=32)

                        # Calculate R² for training data
                        train_r2 = r2_score(y_train, y_train_pred)

                        # Skip configurations with low R² score
                        if train_r2 < 0.4:
                            continue

                        # Calculate R² for testing data
                        test_r2 = r2_score(y_test, y_test_pred)

                        # Remove results where R² is less than 0.6
                        if test_r2 > 0.6:
                            train_mse = mean_squared_error(y_train, y_train_pred)

                            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                            plot(y_train, y_train_pred, f'Training Data\nSample: {sample_size}, Layers: {layers}, Nodes: {nodes}, Epochs: {epochs}, Activation: {activation}', mse=train_mse, r2=train_r2, ax=axes[0])
                            plot(y_test, y_test_pred, f'Testing Data\nSample: {sample_size}, Layers: {layers}, Nodes: {nodes}, Epochs: {epochs}, Activation: {activation}', r2=test_r2, ax=axes[1])
                            plt.tight_layout()
                            plt.show()

# Example parameters to test
hidden_layers = [1, 2, 3]
hidden_nodes = [16, 32, 64]
epochs_list = [10, 25, 50]
sample_sizes = [1000, 5000, 10000]  # Different sample sizes to test
activation_functions = ['relu', 'tanh', 'leaky_relu']  # Different activation functions

# Call the varying parameters function
varying_parameters(hidden_layers, hidden_nodes, epochs_list, sample_sizes, activation_functions)
