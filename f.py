import requests

symbol = "BTCUSDT"  
interval = "1m"  
url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1"

response = requests.get(url)
data = response.json()

# Print the response to understand the data structure
# print(data)
timestamp, open_, high, low, close, volume, quote_asset_volume, number_of_trades, \
taker_buy_base_volume, taker_buy_quote_volume, _, _ = data[0]  # Ignore last two values

print({
    "timestamp": timestamp,
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
    "quote_asset_volume": quote_asset_volume,
    "number_of_trades": number_of_trades,
    "taker_buy_base_volume": taker_buy_base_volume,
    "taker_buy_quote_volume": taker_buy_quote_volume
})
