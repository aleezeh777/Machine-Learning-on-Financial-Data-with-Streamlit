  **StockSavvy Analytics**

An interactive Streamlit app for stock market analysis using machine learning, developed as part of Programming for Finance under the guidance of Dr.Usama Arshad.
App Overview
StockSavvy Analytics is a web-based application that enables users to analyze stock market data through a comprehensive machine learning pipeline. It supports both custom datasets (e.g., CSV files with columns like Date, Netsol_price, Mughal_price, Kohat_price) and live data from Yahoo Finance. With a sleek pastel-themed UI (coral and mint colors), the app guides users through data loading, preprocessing, feature engineering, model training, evaluation, and visualization.
Key Features

*Data Loading*: Upload CSV/Excel files or fetch real-time stock data (e.g., AAPL, TSLA) via Yahoo Finance.
Preprocessing: Handle missing values and clip outliers for clean data.
Feature Engineering: Generate features like moving averages, volatility, and daily returns; visualize correlations and scatter matrices.
Train/Test Split: Split data for model training with customizable test size.
Model Training: Train models like Linear Regression and K-Means Clustering.
Evaluation: Assess model performance with metrics (RMSE, R²) and visualizations (residual plots, histograms).
Visualization: Interactive Plotly charts, including candlestick charts, area plots, rolling averages with confidence bands, and feature importance bars.
User Experience: Progress tracking, downloadable results, and an intuitive interface.

**Tech Stack**

Python: Core programming language.
Streamlit: Web app framework.
Pandas & NumPy: Data manipulation and analysis.
Plotly: Interactive visualizations.
Scikit-learn: Machine learning models.
yfinance: Real-time stock data retrieval.
openpyxl & tenacity: Excel support and robust API calls.


*Contributing*
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature enhancements, or documentation improvements.
Acknowledgments



