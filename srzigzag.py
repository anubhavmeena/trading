import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
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
    dev = deviation

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

    return piv_prices, piv_index, piv_struct, piv_dir

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr.iloc[-1]

def merge_levels(levels, atr, tolerance):
    if not levels:
        return []
    sorted_levels = sorted(set(levels))
    merged = []
    i = 0
    while i < len(sorted_levels):
        current = sorted_levels[i]
        width = 1
        j = i + 1
        count = 1
        total = current
        while j < len(sorted_levels) and abs(sorted_levels[j] - current) < atr * tolerance:
            total += sorted_levels[j]
            count += 1
            width += 1
            j += 1
        current = total / count
        merged.append((current, width))
        i = j
    return merged

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
    df = pd.DataFrame({
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })
    df['timestamp'] = data.get('timestamp', [])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    if df['timestamp'].isna().any():
        print("Warning: Some timestamps could not be parsed. Using fallback.")
        start_date = pd.to_datetime(from_date)
        freq = 'D' if timeframe.lower() == 'daily' else f'{int(timeframe)}min'
        df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq=freq)
    df.set_index('timestamp', inplace=True)
    return df

def plot_interactive(df, segments, colors, merged_resistance, merged_support, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=(title, 'Volume'), row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='OHLC'), row=1, col=1)

    # ZigZag lines
    for seg, col in zip(segments, colors):
        x0, y0 = seg[0]
        x1, y1 = seg[1]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                 line=dict(color=col, width=2), name='ZigZag'), row=1, col=1)

    # Support levels
    for level, width in merged_support:
        fig.add_trace(go.Scatter(x=df.index, y=[level] * len(df), mode='lines',
                                 line=dict(color='green', width=width, dash='dash'), name='Support'), row=1, col=1)

    # Resistance levels
    for level, width in merged_resistance:
        fig.add_trace(go.Scatter(x=df.index, y=[level] * len(df), mode='lines',
                                 line=dict(color='red', width=width, dash='dash'), name='Resistance'), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)

    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2_title='Volume',
        title=title,
        template='plotly_dark'  # Optional: Use a dark theme
    )

    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Plot interactive ZigZag with Support/Resistance using Dhan API")
    parser.add_argument('--timeframe', required=True, help="Timeframe (e.g., 'daily', '5', '15')")
    parser.add_argument('--security_id', default='1333', help="Security ID (default: 1333)")
    parser.add_argument('--exchange_segment', default='NSE_EQ', help="Exchange segment (default: NSE_EQ)")
    parser.add_argument('--instrument', default='EQUITY', help="Instrument type (default: EQUITY)")
    parser.add_argument('--from_date', default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), help="From date (YYYY-MM-DD)")
    parser.add_argument('--to_date', default=datetime.now().strftime('%Y-%m-%d'), help="To date (YYYY-MM-DD)")
    parser.add_argument('--depth', type=int, default=12, help="Depth (default: 12)")
    parser.add_argument('--deviation', type=float, default=5.0, help="Deviation in points (default: 5.0)")
    parser.add_argument('--backstep', type=int, default=3, help="Backstep (default: 3)")
    parser.add_argument('--atr_period', type=int, default=14, help="ATR period (default: 14)")
    parser.add_argument('--tolerance', type=float, default=1.0, help="Tolerance factor for merging levels (default: 1.0)")
    args = parser.parse_args()

    access_token = input("Enter your Dhan access token: ")

    df = fetch_historical_data(access_token, args.security_id, args.exchange_segment, args.instrument, args.timeframe, args.from_date, args.to_date)

    piv_prices, piv_index, piv_struct, piv_dir = compute_zigzag(df, args.depth, args.deviation, args.backstep)

    atr = compute_atr(df, args.atr_period)

    high_pivots = [piv_prices[i] for i in range(len(piv_prices)) if piv_dir[i] == 1]
    low_pivots = [piv_prices[i] for i in range(len(piv_prices)) if piv_dir[i] == -1]

    merged_resistance = merge_levels(high_pivots, atr, args.tolerance)
    merged_support = merge_levels(low_pivots, atr, args.tolerance)

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

    plot_interactive(df, segments, colors, merged_resistance, merged_support, f"ZigZag with S/R on {args.security_id} ({args.timeframe})")

if __name__ == "__main__":
    main()
