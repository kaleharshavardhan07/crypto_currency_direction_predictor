import pickle
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from datetime import datetime
def fetch_current_price(symbol="BTCUSDT"):
    """ Fetch the current price of a cryptocurrency from Binance API """
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    return float(data['price'])

print(fetch_current_price())
