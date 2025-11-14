import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load featured datasets
amazon_featured = pd.read_csv("Amazon_featured.csv")
google_featured = pd.read_csv("GOOG_featured.csv")
netflix_featured = pd.read_csv("NFLX_featured.csv")

# Function to prepare data for LSTM
def prepare_lstm_data(df, target_col='Close', sequence_length=60):
    # Sort by date
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Select features (exclude Date and target if predicting target)
    features = df.drop(columns=['Date', target_col])
    target = df[target_col]

    # Scale features
    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)

    # Scale target
    scaler_target = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(target_scaled[i])

    X, y = np.array(X), np.array(y)

    return X, y, scaler_features, scaler_target

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and evaluate model
def train_evaluate_model(df, name, sequence_length=60):
    print(f"\n--- Training LSTM Model for {name} ---")

    # Prepare data
    X, y, scaler_features, scaler_target = prepare_lstm_data(df, sequence_length=sequence_length)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Train model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # Predict
    predictions = model.predict(X_test)

    # Inverse scale predictions and actual
    predictions_inv = scaler_target.inverse_transform(predictions)
    y_test_inv = scaler_target.inverse_transform(y_test)

    # Evaluate
    mse = mean_squared_error(y_test_inv, predictions_inv)
    mae = mean_absolute_error(y_test_inv, predictions_inv)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save model
    model.save(f'{name.lower()}_lstm_model.h5')

    return model, predictions_inv, y_test_inv

# Train models for each stock
amazon_model, amazon_pred, amazon_actual = train_evaluate_model(amazon_featured, "Amazon")
google_model, google_pred, google_actual = train_evaluate_model(google_featured, "Google")
netflix_model, netflix_pred, netflix_actual = train_evaluate_model(netflix_featured, "Netflix")

print("\nModel training completed for all stocks.")
