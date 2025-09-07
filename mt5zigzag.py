import requests
import pandas as pd
import mplfinance as mpf
import argparse
import json
from datetime import datetime, timedelta

def get_trend_color(lbl):
    if lbl in ["HH", "HL"]:
        return 'blue'
    elif lbl in ["LH", "LL"]:
        return 'red'
    else:
        return 'gray'

def get_structure_label(piv_prices, piv_dir, price, dir_):
    lbl = ""
    n = len(piv_prices)
    if n > 1:
        for j in range(n-2, -1, -1):
            if piv_dir[j] == dir_:
                prev_price = piv_prices[j]
                if dir_ == 1:
                    lbl = "HH" if price > prev_price else "LH"
                else:
                    lbl = "HL" if price > prev_price else "LL"
                break
    if not lbl:
        lbl = "H" if dir_ == 1 else "L"
    return lbl

def process_pivot(piv_prices, piv_index, piv_dir, piv_struct, price, index, dir_, backstep, dev):
    n = len(piv_prices)
    if n == 0:
        piv_prices.append(price)
        piv_index.append(index)
        piv_dir.append(dir_)
        lbl = "H" if dir_ == 1 else "L"
        piv_struct.append(lbl)
    else:
        last_price = piv_prices[-1]
        last_index = piv_index[-1]
        last_dir = piv_dir[-1]
        if dir_ == last_dir:
            for j in range(n-1, -1, -1):
                if index - piv_index[j] <= backstep and piv_dir[j] == dir_:
                    more_extreme = (dir_ == 1 and price > piv_prices[j]) or (dir_ == -1 and price < piv_prices[j])
                    if more_extreme:
                        piv_prices[j] = price
                        piv_index[j] = index
                        lbl = get_structure_label(piv_prices, piv_dir, price, dir_)
                        piv_struct[j] = lbl
                    break
        else:
            if abs(price - last_price) >= dev:
                piv_prices.append(price)
                piv_index.append(index)
                piv_dir.append(dir_)
                lbl = get_structure_label(piv_prices, piv_dir, price, dir_)
                piv_struct.append(lbl)

def compute_zigzag(df, depth=12, deviation=5.0, backstep=3):
    piv_prices = []
    piv_index = []
    piv_dir = []
    piv_struct = []
    swing_high = None
    swing_high_index = None
    swing_low = None
    swing_low_index = None
    dev = deviation  # Assuming deviation in price units

    for i in range(len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if swing_high is None or high > swing_high:
            swing_high = high
            swing_high_index = i

        if swing_low is None or low < swing_low:
            swing_low = low
            swing_low_index = i

        if i - swing_high_index > depth and swing_high is not None:
            process_pivot(piv_prices, piv_index, piv_dir, piv_struct, swing_high, swing_high_index, 1, backstep, dev)
            swing_high = None

        if i - swing_low_index > depth and swing_low is not None:
            process_pivot(piv_prices, piv_index, piv_dir, piv_struct, swing_low, swing_low_index, -1, backstep, dev)
            swing_low = None

    return piv_prices, piv_index, piv_struct

def fetch_historical_data(access_token, security_id, exchange_segment, instrument, timeframe, from_date, to_date):
    headers = {
        'access-token': access_token,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    if timeframe.lower() == 'daily':
        url = 'https://api.dhan.co/v2/charts/historical'
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "expiryCode": 0,
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }
    else:
        url = 'https://api.dhan.co/v2/charts/intraday'
        interval = int(timeframe)
        if interval not in [1, 5, 15, 30, 60]:
            raise ValueError("Invalid interval. Supported: 1,5,15,30,60")
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")
    data = response.json()
    
    # Debug: Print the API response to inspect its structure
    print("API Response:", json.dumps(data, indent=2))
    
    # Check if required keys exist
    required_keys = ['open', 'high', 'low', 'close', 'volume']
    if not all(key in data for key in required_keys):
        raise ValueError(f"Missing required keys in API response: {required_keys}")
    
    df = pd.DataFrame({
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })
    
    # Handle timestamp
    timestamp_key = 'start_Time'  # Adjust this based on actual key (e.g., 'startTime', 'timestamp')
    if timestamp_key in data and len(data[timestamp_key]) == len(df):
        df['timestamp'] = pd.to_datetime(data[timestamp_key], errors='coerce')
    else:
        print(f"Warning: '{timestamp_key}' not found or length mismatch. Generating default timestamps.")
        # Fallback: Generate timestamps based on timeframe
        start_date = pd.to_datetime(from_date)
        if timeframe.lower() == 'daily':
            df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq='D')
        else:
            df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq=f'{int(timeframe)}min')
    
    if df['timestamp'].isna().any():
        raise ValueError("Invalid timestamp format in API response")
    df.set_index('timestamp', inplace=True)
    return df
def main():
    parser = argparse.ArgumentParser(description="Plot ZigZag over candlestick chart using Dhan API")
    parser.add_argument('--timeframe', required=True, help="Timeframe (e.g., 'daily', '5', '15')")
    parser.add_argument('--security_id', default='13', help="Security ID (default: 1333 for example)")
    parser.add_argument('--exchange_segment', default='IDX_I', help="Exchange segment (default: NSE_EQ)")
    parser.add_argument('--instrument', default='INDEX', help="Instrument type (default: EQUITY)")
    parser.add_argument('--from_date', default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), help="From date (YYYY-MM-DD)")
    parser.add_argument('--to_date', default=datetime.now().strftime('%Y-%m-%d'), help="To date (YYYY-MM-DD)")
    parser.add_argument('--depth', type=int, default=12, help="Depth (default: 12)")
    parser.add_argument('--deviation', type=float, default=5.0, help="Deviation in points (default: 5.0)")
    parser.add_argument('--backstep', type=int, default=3, help="Backstep (default: 3)")
    args = parser.parse_args()

    access_token = input("Enter your Dhan access token: ")  # Secure input

    df = fetch_historical_data(access_token, args.security_id, args.exchange_segment, args.instrument, args.timeframe, args.from_date, args.to_date)

    piv_prices, piv_index, piv_struct = compute_zigzag(df, args.depth, args.deviation, args.backstep)

    segments = []
    colors = []
    for i in range(1, len(piv_prices)):
        x1 = df.index[piv_index[i-1]]
        y1 = piv_prices[i-1]
        x2 = df.index[piv_index[i]]
        y2 = piv_prices[i]
        segments.append([(x1, y1), (x2, y2)])
        col = get_trend_color(piv_struct[i])
        colors.append(col)

    mpf.plot(df, type='candle', style='yahoo', alines=dict(alines=segments, colors=colors, linewidths=2), volume=True, title=f"ZigZag on {args.security_id} ({args.timeframe})")

if __name__ == "__main__":
    main()
