import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import datetime
import numpy as np
from tensorflow.keras.models import load_model


# Initialize the Binance client
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)

model = load_model('modelo_con_indicador_tecnico.h5')

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Live Bitcoin price forecast with LSTM neural network"),
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every 10 seconds
        n_intervals=0
    ),
    dcc.Graph(id='h-live-graph', animate=True),
    dcc.Interval(
        id='interval-component',
        interval=3600*1000,  # Update every 10 seconds
        n_intervals=0
    ),
],)


# Define a function to calculate RSI
def calculate_rsi2(prices, period=14):
    gains = []
    losses = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calcular_next_secuence(close_prices):
    rsi_20 = calculate_rsi2(close_prices, 20)
    rsi_50 = calculate_rsi2(close_prices, 50)
    rsi_100 = calculate_rsi2(close_prices, 100)
    mv_20 = sum(close_prices[-20:]) / 20
    mv_50 = sum(close_prices[-50:]) / 50
    mv_100 = sum(close_prices[-100:]) / 100
    secuence = [close_prices[-1][0],mv_20[0],mv_50[0],mv_100[0],rsi_20,rsi_50,rsi_100[0]]
    return secuence

def create_sequences(data, time_steps=60):
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i+time_steps])
        labels.append(data[i+time_steps][0])
    return np.array(sequences), np.array(labels)


def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
    
def fetch_live_data():
    # Fetch recent trades from Binance
    klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=200)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

def fetch_live_data_hour():
    # Fetch recent trades from Binance
    klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=200)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

def calculate_future_predictions(data, number_forecast):
    combined_df = data
    # Convert 'close' column to numeric type
    combined_df['close'] = pd.to_numeric(combined_df['close'], errors='coerce')
    
    # Windows sizes to aggregate
    moving_averages_window_size = [20,50,100]
    rsi_window_size = [20,50,100]

    # Calculate RSI for each window size
    for n in rsi_window_size:
        combined_df['rsi_{}'.format(n)] = calculate_rsi(combined_df['close'], n)

    # Calculate the moving average of the 'close' price
    for n in moving_averages_window_size:
        combined_df['moving_average_{}'.format(n)] = combined_df['close'].rolling(window=n).mean()

    data = combined_df[['close', 'moving_average_20', 'moving_average_50', 'moving_average_100', 'rsi_20', 'rsi_50', 'rsi_100']]
    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)


    # Create sequences
    time_steps = 60
    X, y = create_sequences(scaled_data, time_steps)
    future_predictions = []
    current_sequence = X[-1]  
    for _ in range(number_forecast):
        next_prediction = model.predict(current_sequence.reshape(1, time_steps, X.shape[2]))
        future_predictions.append(next_prediction[0])
        close_prices = []
        for n in current_sequence[1:]:
            close_prices.append(n[0])
        close_prices.append(next_prediction[0])
        secuence =calcular_next_secuence(close_prices)
        current_sequence = np.append(current_sequence[1:], [secuence], axis=0)

    scaler2 = MinMaxScaler(feature_range=(0, 1))
    close_data = data[['close']]
    scaled_data = scaler2.fit_transform(close_data)
    
    future_predictions = scaler2.inverse_transform(future_predictions)
    result = []
    for n in future_predictions:
        result.append(n[0])
    return result


# Callback to update the graph
@app.callback(Output('live-graph', 'figure',),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    data = fetch_live_data()
    prediction = calculate_future_predictions(data, 20)
    # Future predicted prices
    future_predicted_prices = [data['close'][-1]]+ prediction
    figure = {
        'data': [go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines+markers',
            name='Real Price'
        ),
        go.Scatter(
            x=[data.index[-1] + datetime.timedelta(minutes=i) for i in range(len(future_predicted_prices))],
            y=future_predicted_prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red')  # Change the color for predicted prices
        )],
        'layout': go.Layout(
            xaxis=dict(range=[min(data.index), max(data.index) + datetime.timedelta(minutes=len(future_predicted_prices))]),  # Extend x-axis for predicted prices
            yaxis=dict(range=[min(data['close']), max(data['close'])]),
            title="Minute Price Prediction"
        )
    }
    return figure

# Callback to update the graph
@app.callback(Output('h-live-graph', 'figure',),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    data = fetch_live_data_hour()
    prediction = calculate_future_predictions(data, 20)
    # Future predicted prices
    future_predicted_prices = [data['close'][-1]]+ prediction
    figure = {
        'data': [go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines+markers',
            name='Real Price'
        ),
        go.Scatter(
            x=[data.index[-1] + datetime.timedelta(hours=i) for i in range(len(future_predicted_prices))],
            y=future_predicted_prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red')  # Change the color for predicted prices
        )],
        'layout': go.Layout(
            xaxis=dict(range=[min(data.index), max(data.index) + datetime.timedelta(hours=len(future_predicted_prices))]),  # Extend x-axis for predicted prices
            yaxis=dict(range=[min(data['close']), max(data['close'])]),
            title="Hourly Price Prediction"
        )
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
