import pickle
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from datetime import datetime

# Load the trained model and scaler
with open("rf.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

def fetch_binance_data(symbol="BTCUSDT", interval="1m"):
    """ Fetch the latest price data from Binance API. """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=10"
    response = requests.get(url)
    data = response.json()

    # Use the most recent entry for current data
    current_data = data[-1]
    # Use historical data for calculating shifted features
    historical_data = data[:-1]

    # Extract required fields from current entry
    timestamp, open_, high, low, close, volume, quote_asset_volume, number_of_trades, \
        taker_buy_base_volume, taker_buy_quote_volume, _, _ = current_data

    return {
        "timestamp": timestamp,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
        "quote_asset_volume": float(quote_asset_volume),
        "number_of_trades": int(float(number_of_trades)),
        "taker_buy_base_volume": float(taker_buy_base_volume),
        "taker_buy_quote_volume": float(taker_buy_quote_volume),
        "historical_data": historical_data
    }

def preprocess_data(data):
    """ Convert real-time data into the required format for prediction. """
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Add historical context for shifted features
    historical_df = pd.DataFrame(data['historical_data'], 
                                 columns=['timestamp', 'open', 'high', 'low', 'close', 
                                          'volume', 'close_time', 'quote_asset_volume', 
                                          'number_of_trades', 'taker_buy_base_volume', 
                                          'taker_buy_quote_volume', 'ignore'])
    
    # Convert to numeric
    historical_df = historical_df.apply(pd.to_numeric)

    # Convert timestamp to datetime features
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df.drop(columns=["timestamp"], inplace=True)

    # Calculate rolling and shifted features
    df["MA50"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["MA20"] = df["close"].rolling(window=20, min_periods=1).mean()
    
    # Exponential Moving Averages
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()

    # Shifted features using historical data
    df["close_shift_1"] = historical_df['close'].iloc[-1]
    df["close_shift_2"] = historical_df['close'].iloc[-2]
    df["close_shift_3"] = historical_df['close'].iloc[-3]
    df["close_shift_5"] = historical_df['close'].iloc[-5]
    df["close_shift_10"] = historical_df['close'].iloc[-1]

    # Bollinger Bands
    df["MA20STD"] = df["close"].rolling(window=20, min_periods=1).std()
    df["bollinger_upper"] = df["MA20"] + (df["MA20STD"] * 2)
    df["bollinger_lower"] = df["MA20"] - (df["MA20STD"] * 2)

    # Select required features in the exact order from training
    features = [
        'hour', 'minute', 'open', 'high', 'low', 'close', 'MA50',
        'MA20', 'EMA50', 'EMA26', 'close_shift_1', 'close_shift_2', 
        'close_shift_3', 'close_shift_5', 'close_shift_10', 
        'volume', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 
        'bollinger_upper', 'bollinger_lower'
    ]

    # Prepare final DataFrame
    df_final = df[features].fillna(method="ffill").fillna(method="bfill")

    # Scale the data
    scaled_data = scaler.transform(df_final)

    return scaled_data

@app.route("/predict", methods=["GET"])
def predict():
    """ API Endpoint to fetch live data, process it, and make a prediction. """
    try:
        # Fetch live data
        data = fetch_binance_data()

        # Preprocess data
        processed_data = preprocess_data(data)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        return jsonify({
            "data_gain":data,
            "symbol": "BTCUSDT",
            "prediction": int(prediction),
            "message": "1 indicates price will go up, 0 indicates price will go down."
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)