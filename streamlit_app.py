from datetime import date
from datetime import timedelta
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from stocknews import StockNews


def get_data():
   # Authenticate and open the Google Sheet
    credentials_dict = st.secrets['gcp_service_account']
    credentials = service_account.Credentials.from_service_account_info(credentials_dict, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    gc = gspread.Client(auth=credentials)
    wks = gc.open("btc").sheet1
    data = wks.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    return df

def load_data_sheet():
    @st.cache_data()
    def _load_data_sheet():
        price_df = get_data()
        # Convert 'Date' column to the desired format ('2020/01/01')
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.strftime('%Y-%m-%d')

        # Remove commas from the price columns and convert them to numeric
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            price_df[col] = price_df[col].str.replace(',', '').astype(str)

        return price_df

    return _load_data_sheet()

def load_data_yfinance():
    @st.cache_data()
    def _load_data_yfinance():
        today = date.today()
        yesterday = today - timedelta(days=1)
        y_df = yf.download(tickers='BTC-USD', start='2020-01-01', end=yesterday)
        y_df['Date'] = pd.to_datetime(y_df.index)
        y_df['Volume'] = pd.to_numeric(y_df['Volume'])

        return y_df

    return _load_data_yfinance()

def main():
    # DATA LOAD 
    d = load_data_sheet()
    y_finance = load_data_yfinance()
    
    # HEADER
    col1, col2 = st.columns([1, 5])
    # Logo
    with col1:
        st.image(image="img/btc-logo.png", width=100)
    # Title
    with col2:
        st.title("Bitcoin")
    # Description
    st.caption('This is the result of my automated Pipelines retrieving the daily Bitcoin price. With a lambda function on AWS to update the different prices through an API for stocks in a Google Sheets. The historical BTC price is derived from Web Scrapping. This project is intended for training and teaching purposes, not for investment advice.')
    # Sub-title
    st.markdown("#### Metrics from Yesterday:")
    
    # METRICS
    d['Close'] = pd.to_numeric(d['Close'])
    
    col3, col4, col5 = st.columns([3, 3, 6]) 
    
    last_close_price = d['Close'].iloc[-1]  # Get the last value of the 'Close' column
    # Calculate the variation rate from the day before the last one
    variation_rate = round((d['Close'].iloc[-1] - d['Close'].iloc[-2]) / d['Close'].iloc[-2] * 100, 4)
    last_date = d['Date'].iloc[-1]
    last_volume = y_finance['Volume'].iloc[-1]
    formatted_price = "{:,}".format(last_close_price)
    formatted_volume = "{:,}".format(last_volume)
    
    # Display metrics
    col3.metric("Close Price", f"$ {formatted_price}", help=f"Last Close Price {last_date}")
    col4.metric("Variation Rate", f"{variation_rate:.2f}%", delta=variation_rate, delta_color="normal", help="Daily variation rate")
    col5.metric("Volume", f"{formatted_volume} $", help=f"Last Volume Transaction")
    
    # SIDEBAR
    # Convert 'Date' column to datetime type
    d['Date'] = pd.to_datetime(d['Date'])

    # Set minimum and maximum dates
    min_date = d['Date'].min()
    max_date = d['Date'].max()

    # Sidebar for date range filter
    start_date_str = st.sidebar.date_input("Select Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date_str = st.sidebar.date_input("Select End Date", max_date, min_value=min_date, max_value=max_date)

    # Convert string inputs to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Filter data based on selected date range
    mask_date = (d['Date'] >= start_date) & (d['Date'] <= end_date)
    filtered_data = d.loc[mask_date]
    
    # PLOT
    fig = go.Figure(data=[go.Candlestick(x=filtered_data['Date'],
                                     open=filtered_data['Open'],
                                     high=filtered_data['High'],
                                     low=filtered_data['Low'],
                                     close=filtered_data['Close'])])

    fig.update_layout(title='Candlestick BTC- USD ',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      width=800,
                      height=700)
    st.plotly_chart(fig)
    
    # SECTION TABS
    princing_data,volume, news = st.tabs(['Pricing Data', 'Volume', 'Top 10 News'])
    
    with princing_data:
        st.header("Price Movements")
        filtered_data['% Change'] = filtered_data['Close'] / filtered_data['Close'].shift(1) - 1
        st.dataframe(filtered_data, use_container_width=True)
        annual_return = round(filtered_data['% Change'].mean()*252*100, 2)
        st.write('Annual Return is ', annual_return, '%')
        
    with volume:
        st.header('Volume transaction per day')
        
        # Filter data based on selected date range
        mask_date_vol = (y_finance.index >= pd.Timestamp(start_date)) & (y_finance.index <= pd.Timestamp(end_date))
        filtered_data_vol = y_finance[mask_date_vol]

        fig = go.Figure(data=[go.Bar(x=filtered_data_vol.index, y=filtered_data_vol['Volume'])])
        fig.update_layout(title='Daily volume',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        width=800,
                        height=700)
        st.plotly_chart(fig)
        
    with news:
        ticker = 'BTC'
        st.header(f'News of {ticker}')
        sn = StockNews(stocks=ticker, save_news=False)
        df_news = sn.read_rss()
        for _, row in df_news.head(10).iterrows():
            st.subheader(row['title'])
            st.write(row['published'])
            st.write(row['summary'])
            
    
if __name__ == "__main__":
    main()
