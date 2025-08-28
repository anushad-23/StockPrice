import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, name="Dataset"):
    """
    Preprocess stock data:
    - Ensures 'Date' is datetime
    - Extracts Year, Month, Day, DayOfWeek
    - Normalizes numeric columns using MinMaxScaler
    """
    print(f"\n--- Preprocessing {name} ---")
    
    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Extract new time features
    df['Year'] = df['Date'].dt.year
    print(df['Year'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    
    # Normalize numeric columns
    scaler = MinMaxScaler()
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    available_cols = [col for col in numeric_cols if col in df.columns]
    df[available_cols] = scaler.fit_transform(df[available_cols])
    
    print("Preprocessing done!")
    return df
