#!/usr/bin/env python3
"""
stock_analysis.py

Usage examples:
  python stock_analysis.py AAPL
  python stock_analysis.py TCS.NS --period 6mo --save-fig report.png --save-csv data.csv

Requirements:
  pip install yfinance pandas matplotlib
"""

import sys
import argparse
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def download_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, threads=True)
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {ticker}: {e}")
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{ticker}' with period='{period}'.")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def find_crossovers(df: pd.DataFrame):
    """
    Return indices (timestamps) where MA_20 crosses MA_50.
    A positive "signal" means MA_20 crossed above MA_50 (bullish).
    A negative "signal" means MA_20 crossed below MA_50 (bearish).
    """
    ma_short = df['MA_20']
    ma_long = df['MA_50']
    # Compute sign of difference and find where it changes
    diff = ma_short - ma_long
    sign = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    signal = sign.diff()
    # signal == 2  -> -1 to +1 => bullish cross
    # signal == -2 -> +1 to -1 => bearish cross
    bullish = df.index[(signal == 2)]
    bearish = df.index[(signal == -2)]
    return bullish, bearish

def print_summary(df: pd.DataFrame, ticker: str):
    last = df.iloc[-1]
    print(f"\n=== {ticker} SUMMARY ({df.index[0].date()} to {df.index[-1].date()}) ===")
    print(f"Last Close: {last['Close']:.2f}")
    print(f"20-day MA:  {last['MA_20']:.2f}")
    print(f"50-day MA:  {last['MA_50']:.2f}")
    print("\nBasic statistics for Close:")
    print(df['Close'].describe())
    print("\nDaily return (last 5):")
    print(df['Daily_Return'].dropna().tail(5).round(4))
    print(f"\nAnnualized volatility (stddev * sqrt(252)): "
          f"{df['Daily_Return'].std() * (252**0.5):.4f}")

def plot_data(df: pd.DataFrame, ticker: str, save_fig: str = None, show: bool = True):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label="Close Price", color='tab:blue')
    ax.plot(df.index, df['MA_20'], label="20-day MA", color='tab:orange')
    ax.plot(df.index, df['MA_50'], label="50-day MA", color='tab:green')

    bullish, bearish = find_crossovers(df)
    # Plot markers for the last few crossovers (if any)
    if len(bullish):
        ax.scatter(bullish[-3:], df.loc[bullish[-3:], 'Close'], marker='^', color='green', s=80, label='Bullish Cross')
    if len(bearish):
        ax.scatter(bearish[-3:], df.loc[bearish[-3:], 'Close'], marker='v', color='red', s=80, label='Bearish Cross')

    ax.set_title(f"{ticker} — Close Price with 20/50-day MA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_fig:
        fig.savefig(save_fig, dpi=150)
        print(f"Saved figure to {save_fig}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def main(argv):
    parser = argparse.ArgumentParser(description="Download stock data and show moving averages.")
    parser.add_argument("ticker", help="Ticker symbol (e.g. AAPL, TCS.NS)")
    parser.add_argument("--period", default="1y", help="yfinance period (e.g. 1y, 6mo, 3mo). Default: 1y")
    parser.add_argument("--save-fig", help="Path to save the plot (PNG, PDF, etc).")
    parser.add_argument("--save-csv", help="Path to save the downloaded data as CSV.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot interactively.")
    args = parser.parse_args(argv)

    try:
        df = download_data(args.ticker, period=args.period)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    df = compute_indicators(df)
    print_summary(df, args.ticker)

    if args.save_csv:
        df.to_csv(args.save_csv)
        print(f"Saved data to {args.save_csv}")

    plot_data(df, args.ticker, save_fig=args.save_fig, show=not args.no_show)

if __name__ == "__main__":
    main(sys.argv[1:])
