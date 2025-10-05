import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pywt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12,6)

os.makedirs('plots', exist_ok=True)

# ------------- Parameters ----------------
TICKERS = {
'EURUSD': 'EURUSD=X',
'GBPJPY': 'GBPJPY=X',
'CHFJPY': 'CHFJPY=X'
}
START = '2011-01-01'
END = '2021-01-01' # 10 years exactly as in the report
SCALES = np.arange(1,8) # 1..7
WAVELET = 'mexh'

# ------------- Helper functions ----------------
def download_close(ticker, start, end):
    print(f"Downloading {ticker} from Yahoo Finance...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if 'Close' not in data.columns:
        raise RuntimeError(f"No 'Close' column for {ticker}")
    s = data['Close'].copy()
    s.index = pd.to_datetime(s.index)
    return s

def fill_missing(series):
    # Missing values are erplaced by mean of previous and next day: replace NaN by (prev + next)/2 where both exist.
    s = series.copy()
    n = len(s)
    is_na = s.isna()
    na_idx = np.where(is_na)[0]
    for i in na_idx:
        if i==0 or i==n-1:
            continue
        prev = s.iloc[i-1]
        nxt = s.iloc[i+1]
        if not pd.isna(prev) and not pd.isna(nxt):
            s.iloc[i] = 0.5*(prev+nxt)
    s = s.fillna(method='ffill').fillna(method='bfill')
    return s

def compute_cwt_layers(series, scales,wavelet= 'mexh'):
    """
    Decompose the time series using the Continuous Wavelet Transform
    with the Mexican Hat (Ricker) wavelet.
    """
    data = series.values.ravel()
    data = (data - np.mean(data)) / np.std(data)
    # compute CWT 
    coeffs, freqs = pywt.cwt(data, scales, wavelet)
    # normalize by sqrt(scale) 
    coeffs = coeffs / np.sqrt(scales[:, None])
    # trim edges to reduce boundary artifacts: trim 2% at each end
    edge = int(0.02 * len(data))  
    coeffs = coeffs[:, edge:-edge]
    trimmed_index = series.index[edge:-edge]
    layers = pd.DataFrame(coeffs.T, index=trimmed_index, columns=[f'scale_{s}' for s in scales])
    return layers, freqs

def compress_sum(layers):
    # High frequency: sum of scales 1-3 -> scales[0:3]
    # Low frequency: sum of scales 5-7 -> scales[4:7]
    cols = layers.columns.tolist()
    high = layers[cols[0:3]].sum(axis=1)
    low = layers[cols[4:7]].sum(axis=1)
    return high, low

def compress_svd(layers):
    M = layers.T.values # shape (7, T)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Components: U (7 x 7), s (7,), Vt (7 x T)
    # First right-singular vector times singular value gives principal time-series component
    comp1 = s[0] * Vt[0, :]
    comp2 = s[1] * Vt[1, :]
    idx = layers.index
    vol = pd.Series(comp1, index=idx) # interpret comp1 as VOL (dominant slow component)
    white = pd.Series(comp2, index=idx) # interpret comp2 as white-noise (second component)
    return white, vol

def fit_and_report(series, name):
    t = (series.index - series.index[0]).days.values.reshape(-1,1)
    y = series.values.reshape(-1,1)
    model = LinearRegression()
    model.fit(t, y)
    y_pred = model.predict(t)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    r2 = r2_score(y, y_pred)
    return {'slope': slope, 'intercept': intercept, 'r2': r2, 't': t, 'y_pred': y_pred}

# ------------- Main pipeline ----------------
summary_rows = []
for name, ticker in TICKERS.items():
    s = download_close(ticker, START, END)
    s = fill_missing(s)
    # Trimming first and last 20 values 
    if len(s) > 50:
        s = s.iloc[20:-20]

    plt.figure()
    plt.plot(s.index, s.values)
    plt.title(f'{name} Close Price ({START} to {END})')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.savefig(f'plots/{name}_close_price.png')
    plt.close()

    layers, freqs = compute_cwt_layers(s, SCALES, WAVELET)
    
    # Plot the original series and its 7 decomposed layers
    fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(s.index, s.values, color='black')
    axes[0].set_title(f'{name} Exchange Rate and 7-Layer CWT Decomposition (Mexican Hat)')
    axes[0].set_ylabel('Price')

    for i, col in enumerate(layers.columns):
        axes[i + 1].plot(layers.index, layers[col].values, color='C0')
        axes[i + 1].set_ylabel(f'{col}')
        axes[i + 1].grid(False)
        if i < len(layers.columns) - 1:
            axes[i + 1].set_xticklabels([])
    axes[-1].set_xlabel('Date')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f'{name} - Continuous Wavelet Decomposition (7 scales)', fontsize=14, y=0.995)
    plt.savefig(f'plots/{name}_cwt_layers.png')
    plt.close()

    # Compression - Summation
    high_sum, low_sum = compress_sum(layers)
    plt.figure()
    plt.plot(low_sum.index, low_sum.values, label='Low-frequency (VOL)', color='C1')
    plt.plot(high_sum.index, high_sum.values, label='High-frequency (White noise)', color='C0', alpha=0.7)
    plt.legend()
    plt.title(f'{name} - Summation compression (high vs low)')
    plt.savefig(f'plots/{name}_sum_compressed.png')
    plt.close()

    # Compression - SVD
    white_svd, vol_svd = compress_svd(layers)
    plt.figure()
    plt.plot(vol_svd.index, vol_svd.values, label='VOL (SVD comp1)', color='C1')
    plt.plot(white_svd.index, white_svd.values, label='White noise (SVD comp2)', color='C0', alpha=0.7)
    plt.legend()
    plt.title(f'{name} - SVD compression (comp1=VOL, comp2=White)')
    plt.savefig(f'plots/{name}_svd_compressed.png')
    plt.close()
    
# Regression on both compressed methods
    # Summation method
    sum_high_res = fit_and_report(high_sum, f'{name}_sum_high')
    sum_low_res = fit_and_report(low_sum, f'{name}_sum_low')
    # SVD method
    svd_white_res = fit_and_report(white_svd, f'{name}_svd_white')
    svd_vol_res = fit_and_report(vol_svd, f'{name}_svd_vol')

    # Plot regressions (SVD)
    t = svd_vol_res['t']; y_pred = svd_vol_res['y_pred']
    plt.figure()
    plt.plot(vol_svd.index, vol_svd.values, label='VOL (SVD comp1)', color='C1')
    plt.plot(vol_svd.index, y_pred, color='k', linewidth=2, label=f'Regression slope={svd_vol_res["slope"]:.2e}')
    plt.legend()
    plt.title(f'{name} - SVD VOL regression')
    plt.savefig(f'plots/{name}_svd_vol_regression.png')
    plt.close()

    t = svd_white_res['t']; y_pred = svd_white_res['y_pred']
    plt.figure()
    plt.plot(white_svd.index, white_svd.values, label='White (SVD comp2)', color='C0')
    plt.plot(white_svd.index, y_pred, color='k', linewidth=2, label=f'Regression slope={svd_white_res["slope"]:.2e}')
    plt.legend()
    plt.title(f'{name} - SVD White regression')
    plt.savefig(f'plots/{name}_svd_white_regression.png')
    plt.close()

    # Plot regressions (Summation)
    t = sum_low_res['t']; y_pred = sum_low_res['y_pred']
    plt.figure()
    plt.plot(low_sum.index, low_sum.values, label='VOL (sum)', color='C1')
    plt.plot(low_sum.index, y_pred, color='k', linewidth=2, label=f'Regression slope={sum_low_res["slope"]:.2e}')
    plt.legend()
    plt.title(f'{name} - Summation VOL regression')
    plt.savefig(f'plots/{name}_sum_vol_regression.png')
    plt.close()

    t = sum_high_res['t']; y_pred = sum_high_res['y_pred']
    plt.figure()
    plt.plot(high_sum.index, high_sum.values, label='White (sum)', color='C0')
    plt.plot(high_sum.index, y_pred, color='k', linewidth=2, label=f'Regression slope={sum_high_res["slope"]:.2e}')
    plt.legend()
    plt.title(f'{name} - Summation White regression')
    plt.savefig(f'plots/{name}_sum_white_regression.png')
    plt.close()

    # Summary
    summary_rows.append({
        'pair': name,
        'sum_vol_slope': sum_low_res['slope'],
        'sum_vol_r2': sum_low_res['r2'],
        'sum_white_slope': sum_high_res['slope'],
        'sum_white_r2': sum_high_res['r2'],
        'svd_vol_slope': svd_vol_res['slope'],
        'svd_vol_r2': svd_vol_res['r2'],
        'svd_white_slope': svd_white_res['slope'],
        'svd_white_r2': svd_white_res['r2']
    })

# Saving everything
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('compression_regression_summary.csv', index=False)
print('\nSummary saved to compression_regression_summary.csv')

plt.figure(figsize=(10,2 + 0.5*len(summary_df)))
plt.axis('off')
plt.table(cellText=np.round(summary_df.select_dtypes(include=[np.number]), 6).values,
colLabels=summary_df.select_dtypes(include=[np.number]).columns,
rowLabels=summary_df['pair'], loc='center')
plt.title('Regression slopes and R^2 (numeric columns)')
plt.savefig('plots/summary_table.png', bbox_inches='tight')
plt.close()

print('All plots saved in ./plots/.')