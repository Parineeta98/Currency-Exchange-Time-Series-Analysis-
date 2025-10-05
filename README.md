Currency Exchange Time Series Analysis

Overview
This project analyzes currency exchange rate movements over a 10-year period (2011–2021: in report; 2015-2025 in code) to explore trends in volatility and white noise using time series techniques.  
It combines financial analytics with data engineering methods to uncover whether global currency markets are becoming more stable or more erratic over time.

Project Objective
To evaluate the volatility trends and random fluctuations (white noise) in major currency pairs —  
EUR/USD, GBP/JPY, and CHF/CNY — using time series analysis.

Specifically, the project aims to:
- Understand long-term volatility behavior in exchange rates.
- Separate meaningful market signals from random noise.
- Compare compression techniques to detect subtle financial patterns.

Key Concepts:
Exchange Rate
The price of one currency relative to another.

Volatility
Measures how much and how quickly prices move.  
High volatility → unstable market; Low volatility → stable market.

White Noise
Random, unpredictable movements in data with no underlying pattern.  
Identifying it helps isolate true market signals.

Time Series Analysis
A method to study how values (currency prices) change over time to find patterns, cycles, or randomness.

Technical Approach

1. Data Collection
- Source: Yahoo Finance.  
- Collected **daily closing prices** for each currency pair between 2011–2021 and 2015-2025.

2. Data Cleaning
- Handled missing data by imputing values using the mean of neighboring data points.
- Removed outliers and initial/final irregular entries to maintain trend consistency.

3. Decomposition
- Applied Continuous Wavelet Transform (CWT) using the Mexican Hat wavelet.
- Decomposed data into 7 frequency layers to capture both slow (volatility) and fast (white noise) movements.

4. Compression
- Used two compression methods:
  - Summation: Combining top and bottom frequency layers.
  - SVD (Singular Value Decomposition): For dimensionality reduction and pattern extraction.

5. Regression Analysis
- Applied Linear Regression to model and compare long-term volatility and noise trends.

Results Summary
- Volatility across all currencies showed a declining trend, suggesting increasing market stability for the 2015-2021 period.  
- White noise showed mixed patterns, implying persistent short-term unpredictability.
- SVD compression provided more consistent and reliable results compared to simple summation.
