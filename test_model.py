import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Function to prepare data for LSTM (same as in model_building.py)
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

# Function to test model
def test_model(df, model_path, name, sequence_length=60):
    print(f"\n--- Testing {name} Model ---")

    # Prepare data
    X, y, scaler_features, scaler_target = prepare_lstm_data(df, sequence_length=sequence_length)

    # Split into train and test (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Load model
    model = load_model(model_path)

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

    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inv, label='Actual Prices', color='blue')
    plt.plot(predictions_inv, label='Predicted Prices', color='red')
    plt.title(f'{name} Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    return predictions_inv, y_test_inv

# Function to validate on unseen data (last 20% of data)
def validate_unseen_data(df, model_path, name, sequence_length=60):
    print(f"\n--- Validating {name} on Unseen Data ---")

    # Prepare data
    X, y, scaler_features, scaler_target = prepare_lstm_data(df, sequence_length=sequence_length)

    # Use last 20% as unseen data
    split_index = int(0.8 * len(X))
    X_unseen = X[split_index:]
    y_unseen = y[split_index:]

    # Load model
    model = load_model(model_path)

    # Predict
    predictions = model.predict(X_unseen)

    # Inverse scale predictions and actual
    predictions_inv = scaler_target.inverse_transform(predictions)
    y_unseen_inv = scaler_target.inverse_transform(y_unseen)

    # Evaluate
    mse = mean_squared_error(y_unseen_inv, predictions_inv)
    mae = mean_absolute_error(y_unseen_inv, predictions_inv)

    print(f"Unseen Data MSE: {mse:.4f}")
    print(f"Unseen Data MAE: {mae:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(y_unseen_inv, label='Actual Prices', color='blue')
    plt.plot(predictions_inv, label='Predicted Prices', color='red')
    plt.title(f'{name} Actual vs Predicted on Unseen Data')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    return predictions_inv, y_unseen_inv

# Load featured datasets
amazon_featured = pd.read_csv("Amazon_featured.csv")
google_featured = pd.read_csv("GOOG_featured.csv")
netflix_featured = pd.read_csv("NFLX_featured.csv")

# Test models on test set
amazon_pred, amazon_actual = test_model(amazon_featured, "amazon_lstm_model.h5", "Amazon")
google_pred, google_actual = test_model(google_featured, "google_lstm_model.h5", "Google")
netflix_pred, netflix_actual = test_model(netflix_featured, "netflix_lstm_model.h5", "Netflix")

# Validate on unseen data
amazon_pred_unseen, amazon_actual_unseen = validate_unseen_data(amazon_featured, "amazon_lstm_model.h5", "Amazon")
google_pred_unseen, google_actual_unseen = validate_unseen_data(google_featured, "google_lstm_model.h5", "Google")
netflix_pred_unseen, netflix_actual_unseen = validate_unseen_data(netflix_featured, "netflix_lstm_model.h5", "Netflix")

print("\nTesting and validation completed.")
