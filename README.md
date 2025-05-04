StockSavvy Analytics
An interactive Streamlit app for stock market analysis using machine learning, developed as part of [Your Course Name] under the guidance of [Your Instructor's Name].
App Overview
StockSavvy Analytics is a web-based application that enables users to analyze stock market data through a comprehensive machine learning pipeline. It supports both custom datasets (e.g., CSV files with columns like Date, Netsol_price, Mughal_price, Kohat_price) and live data from Yahoo Finance. With a sleek pastel-themed UI (coral and mint colors), the app guides users through data loading, preprocessing, feature engineering, model training, evaluation, and visualization.
Key Features

Data Loading: Upload CSV/Excel files or fetch real-time stock data (e.g., AAPL, TSLA) via Yahoo Finance.
Preprocessing: Handle missing values and clip outliers for clean data.
Feature Engineering: Generate features like moving averages, volatility, and daily returns; visualize correlations and scatter matrices.
Train/Test Split: Split data for model training with customizable test size.
Model Training: Train models like Linear Regression and K-Means Clustering.
Evaluation: Assess model performance with metrics (RMSE, R²) and visualizations (residual plots, histograms).
Visualization: Interactive Plotly charts, including candlestick charts, area plots, rolling averages with confidence bands, and feature importance bars.
User Experience: Progress tracking, downloadable results, and an intuitive interface.

Tech Stack

Python: Core programming language.
Streamlit: Web app framework.
Pandas & NumPy: Data manipulation and analysis.
Plotly: Interactive visualizations.
Scikit-learn: Machine learning models.
yfinance: Real-time stock data retrieval.
openpyxl & tenacity: Excel support and robust API calls.

Deployment
Explore the live app: [Streamlit URL]
Installation

Clone the repository:
git clone https://github.com/[your-username]/fithub.git
cd fithub


Install dependencies:
pip install -r requirements.txt


Run the app locally:
streamlit run stock_savvy_analytics.py



Requirements

Python 3.8+
Dependencies listed in requirements.txt:streamlit==1.31.0
pandas==2.2.0
numpy==1.26.4
plotly==5.18.0
scikit-learn==1.4.0
openpyxl==3.1.2
yfinance==0.2.36
tenacity==8.2.3



Usage

Launch the app:
https://stocksavvyanalyticsam.streamlit.app/


Follow the pipeline steps:

Load Data: Upload a CSV/Excel file (e.g., with Date, Netsol_price, Mughal_price, Kohat_price) or fetch Yahoo Finance data (e.g., AAPL).
Preprocess: Clean data by filling missing values and clipping outliers.
Feature Engineering: Select features (e.g., Netsol_price, Mughal_price) and target (e.g., Kohat_price), add features like moving averages.
Train/Test Split: Configure test size and random state.
Model Training: Train models like Linear Regression or K-Means.
Evaluation: Review metrics and visualizations.
Results Visualization: Explore area plots, rolling averages, and feature importance.

Download results as CSV files for further analysis.


Troubleshooting
Scatter Matrix Error: If you encounter a "hover_data_0 not in data_frame" error, ensure all selected features and target columns exist in your dataset. The app omits hover_data in the scatter matrix to prevent this issue.
Missing Data: Verify your CSV has required columns (e.g., Date, numeric columns like Close or Kohat_price). Check preprocessing output for missing value counts.
Yahoo Finance Issues: If fetching data fails, try popular tickers (e.g., AAPL, MSFT) and ensure an internet connection.
Streamlit Deployment: If the app fails to load on Streamlit Cloud, check the logs for dependency or file path errors.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature enhancements, or documentation improvements.
Acknowledgments

Dr.Usama Arshad For their invaluable mentorship during Programming for Finance.


