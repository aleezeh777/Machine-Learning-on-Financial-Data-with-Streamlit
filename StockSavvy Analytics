import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
import uuid
import io

# Set page config and apply enhanced pastel theme
st.set_page_config(page_title="StockSavvy Analytics", layout="wide", page_icon="📊")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    .stApp {
        background-color: #f9f7f6;
        color: #333333;
        font-family: 'Poppins', sans-serif;
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    .css-1d391kg {
        background: linear-gradient(135deg, #e1f7ed, #d4f0e5);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #4b6587;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #a9c6e2, #8ba8cc);
        color: #333333;
        border: none;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 5px 0;
        width: 100%;
        text-align: left;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: #8ba8cc;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(139, 168, 204, 0.5);
    }
    .stButton>button:disabled {
        background: #d3d3d3;
        cursor: not-allowed;
        opacity: 0.6;
    }
    
    .completed::before {
        content: '✅';
        margin-right: 10px;
    }
    
    .reset-button>button {
        background: linear-gradient(135deg, #f4a7b9, #e8a8b9);
        color: #333333;
        border: none;
        padding: 12px 20px;
        border-radius: 8px;
        width: 100%;
        text-align: center;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .reset-button>button:hover {
        background: #e8a8b9;
        box-shadow: 0 5px 15px rgba(232, 168, 185, 0.5);
    }
    
    .stDataFrame, .stPlotlyChart {
        background: linear-gradient(135deg, #ffffff, #f0f4f8);
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stExpander, .stSelectbox, .stMultiSelect, .stFileUploader, .stNumberInput, .stSlider {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        border: 1px solid;
        padding: 10px;
    }
    .stInfo, .stSuccess {
        background: #e1f7ed;
        border-color: #b3e0cc;
        color: #2e7d32;
    }
    .stWarning {
        background: #fff5e6;
        border-color: #ffcc80;
        color: #ef6c00;
    }
    .stError {
        background: #fce8e9;
        border-color: #f8bcbc;
        color: #c62828;
    }
    
    .stDownloadButton>button {
        background: #a9c6e2;
        color: #333333;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background: #8ba8cc;
        box-shadow: 0 5px 15px rgba(139, 168, 204, 0.5);
    }
    
    .stSpinner > div > div {
        border: 4px solid #a9c6e2;
        border-top: 4px solid #f4a7b9;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .center-image {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    
    .welcome-hero {
        background: linear-gradient(rgba(249,247,246,0.9), rgba(249,247,246,0.9)), 
                    url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        animation: slideIn 1s ease-out;
    }
    
    @keyframes slideIn {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    .progress-bar {
        background: #e0e0e0;
        border-radius: 5px;
        height: 10px;
        margin: 10px 0;
    }
    .progress-fill {
        background: #a9c6e2;
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
    
    .interpretation {
        background: #e1f7ed;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {
            'current_step': 0, 'data_loaded': False, 'preprocessed': False, 'features_engineered': False,
            'data_split': False, 'model_trained': False, 'model_evaluated': False, 'results_visualized': False,
            'df': None, 'df_processed': None, 'target': None, 'features': None,
            'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
            'models': {}, 'y_preds': {}, 'current_price': None, 'last_symbol': None
        }

# Helper functions
def clean_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert {col} to numeric: {e}")
    return df

def is_continuous(series):
    return pd.api.types.is_numeric_dtype(series) and len(series.unique()) > 10

@st.cache_data
def fetch_yfinance_data(symbol, start_date, end_date, _cache_key=None):
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), 
           retry=retry_if_exception_message(match='Too Many Requests'))
    def fetch():
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                st.error(f"No data for {symbol}. Try AAPL, TSLA, MSFT.")
                return None
            return df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    return fetch()

def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.info.get('regularMarketPrice', stock.info.get('currentPrice'))
        return price
    except Exception as e:
        st.warning(f"Could not fetch price for {symbol}: {e}")
        return None

def plot_config(fig, title, x_title, y_title, width=800, height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title=dict(text=x_title, font=dict(size=16)),
        yaxis_title=dict(text=y_title, font=dict(size=16)),
        width=width, height=height, template='plotly_white',
        font_color='#333333', plot_bgcolor='#f0f4f8', paper_bgcolor='#f9f7f6',
        showlegend=True, legend=dict(font=dict(size=14), x=0.01, y=0.99),
        hovermode='closest', margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(153,153,153,0.2)')

# Pipeline steps
def welcome_step():
    st.markdown("""
        <div class="welcome-hero">
            <h1>📊 StockSavvy Analytics</h1>
            <p style="color: #4b6587; font-size: 18px;">Unleash the power of financial data with interactive ML pipelines!</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="center-image"><img src="https://media3.giphy.com/media/M9USyermmmO8WJG6FK/200w.gif" width="300"></div>', unsafe_allow_html=True)
    if st.button("🚀 Start Analyzing", key="start"):
        st.session_state.pipeline['current_step'] = 1
        st.rerun()

def load_data_step():
    st.header("📊 Step 1: Load Data")
    data_option = st.radio("Data Source", ("Upload CSV/Excel", "Yahoo Finance"), help="Choose to upload a dataset or fetch live stock data.")
    
    if data_option == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload File 📂", type=["csv", "xlsx"], help="Upload a dataset with 'Date' and 'Close' columns.")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                df = clean_numeric_columns(df)
                if not {'Date', 'Close'}.issubset(df.columns):
                    st.warning("Dataset needs 'Date' and 'Close' columns.")
                st.session_state.pipeline.update({'df': df, 'data_loaded': True, 'last_symbol': None, 'current_step': 2})
                st.success("✅ Data Loaded!")
                with st.expander("View Data & Stats"):
                    st.dataframe(df)
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    st.write("Stats:")
                    st.dataframe(df.describe())
                if 'Date' in df.columns and 'Close' in df.columns:
                    fig = px.line(df, x='Date', y='Close', title='Closing Price Trend', 
                                  color_discrete_sequence=['#f4a7b9'], hover_data=['Close'])
                    plot_config(fig, 'Closing Price Trend', 'Date', 'Close')
                    st.plotly_chart(fig)
                    st.markdown("""
                        <div class="interpretation">
                        **Interpretation**: This line plot shows the stock's closing price over time. Upward trends indicate price increases, while downward trends show declines. Hover over points to see exact prices and dates. Look for sharp changes, which may align with market events.
                        </div>
                    """, unsafe_allow_html=True)
                if 'Volume' in df.columns:
                    fig = px.line(df, x='Date', y='Volume', title='Trading Volume Trend', 
                                  color_discrete_sequence=['#a7d8d3'], hover_data=['Volume'])
                    plot_config(fig, 'Trading Volume Trend', 'Date', 'Volume')
                    st.plotly_chart(fig)
                    st.markdown("""
                        <div class="interpretation">
                        **Interpretation**: This plot displays trading volume over time. High volume spikes often coincide with significant price movements (check the closing price trend). Low volume may indicate stable or less active trading periods.
                        </div>
                    """, unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Stock Symbol (e.g., AAPL)", "AAPL", help="Enter a valid stock ticker.")
            start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
            end_date = st.date_input("End Date", datetime.date.today())
        with col2:
            if symbol and start_date < end_date:
                with st.spinner("Fetching..."):
                    df = fetch_yfinance_data(symbol.upper(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 
                                            f"{symbol}_{start_date}_{end_date}")
                if df is not None:
                    price = fetch_current_price(symbol.upper())
                    if price:
                        st.metric(f"Current Price ({symbol.upper()})", f"${price:.2f}")
                    st.session_state.pipeline.update({
                        'df': df, 'data_loaded': True, 'last_symbol': symbol.upper(), 'current_price': price, 'current_step': 2
                    })
                    st.success("✅ Data Fetched!")
                    with st.expander("View Data & Stats"):
                        st.dataframe(df)
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        st.text(buffer.getvalue())
                        st.write("Stats:")
                        st.dataframe(df.describe())
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                        increasing_line_color='#f4a7b9', decreasing_line_color='#a7d8d3'
                    )])
                    plot_config(fig, f'{symbol.upper()} Candlestick Chart', 'Date', 'Price')
                    st.plotly_chart(fig)
                    st.markdown("""
                        <div class="interpretation">
                        **Interpretation**: The candlestick chart shows daily price movements (Open, High, Low, Close). Coral candles indicate price increases, mint candles show decreases. Long candles or gaps suggest volatility. Hover to see detailed price data.
                        </div>
                    """, unsafe_allow_html=True)
                    fig = px.line(df, x='Date', y='Volume', title='Trading Volume Trend', 
                                  color_discrete_sequence=['#a7d8d3'], hover_data=['Volume'])
                    plot_config(fig, 'Trading Volume Trend', 'Date', 'Volume')
                    st.plotly_chart(fig)
                    st.markdown("""
                        <div class="interpretation">
                        **Interpretation**: High volume spikes often signal strong buying or selling activity, potentially driving price changes (check the candlestick chart). Consistent low volume may indicate stable trading.
                        </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.warning("Invalid symbol or date range!")

def preprocessing_step():
    st.header("🛠️ Step 2: Preprocessing")
    if not st.session_state.pipeline['data_loaded']:
        st.warning("Load data first!")
        return
    df = st.session_state.pipeline['df'].copy()
    
    missing_values = df.isnull().sum()
    if missing_values.sum():
        st.dataframe(missing_values[missing_values > 0].to_frame(name="Missing Count"))
        df[df.select_dtypes(np.number).columns] = df.select_dtypes(np.number).fillna(df.mean(numeric_only=True))
        st.success("✅ Missing values filled!")
    else:
        st.success("✅ No missing values!")
    
    numeric_cols = df.select_dtypes(np.number).columns
    if numeric_cols.any():
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        st.success("✅ Outliers clipped!")
    
    st.session_state.pipeline.update({'df_processed': df, 'preprocessed': True})
    with st.expander("Processed Data"):
        st.dataframe(df)
    if st.button("Next ➡️", key="preprocess_next"):
        st.session_state.pipeline['current_step'] = 3
        st.rerun()

def feature_engineering_step():
    st.header("📐 Step 3: Feature Engineering")
    if not st.session_state.pipeline['preprocessed']:
        st.warning("Complete preprocessing first!")
        return
    df = st.session_state.pipeline['df_processed'].copy()
    
    if 'Close' in df.columns:
        window = st.slider("MA Window (days)", 5, 50, 20, help="Select window for moving average and volatility.")
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean().fillna(df['Close'])
        df[f'Volatility_{window}'] = df['Close'].rolling(window=window).std().fillna(df['Close'].std())
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        st.success(f"✅ Added {window}-day MA, Volatility, Daily Return!")
    
    numeric_cols = df.select_dtypes(np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns!")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Target (y)", numeric_cols, index=numeric_cols.index('Close') if 'Close' in numeric_cols else 0, 
                              help="Choose the target variable for prediction.")
    with col2:
        features = st.multiselect("Features (X)", [c for c in numeric_cols if c != target], 
                                  default=[c for c in numeric_cols if c != target][:2], 
                                  help="Select features for modeling.")
    if not features:
        st.warning("Select at least one feature!")
        return
    
    if st.checkbox("Apply Scaling", value=True, help="Standardize features for better model performance."):
        try:
            df[features] = StandardScaler().fit_transform(df[features])
            st.success("✅ Features scaled!")
        except Exception as e:
            st.error(f"❌ Scaling error: {e}")
    
    try:
        corr_matrix = df[features + [target]].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', 
                        title='Feature Correlation Matrix', width=600, height=500)
        plot_config(fig, 'Feature Correlation Matrix', '', '')
        st.plotly_chart(fig)
        st.markdown("""
            <div class="interpretation">
            **Interpretation**: The heatmap shows correlations between features and the target. Values close to 1 or -1 indicate strong relationships, while values near 0 suggest weak relationships. High correlations between features (multicollinearity) may affect model performance.
            </div>
        """, unsafe_allow_html=True)
        
        fig = px.scatter_matrix(df[features], title='Feature Pair Plot', width=800, height=600, 
                                color_discrete_sequence=['#a7d8d3'], hover_data=[target])
        plot_config(fig, 'Feature Pair Plot', 'Features', 'Features')
        st.plotly_chart(fig)
        st.markdown("""
            <div class="interpretation">
            **Interpretation**: The scatter matrix shows pairwise relationships between features. Diagonal plots are histograms of each feature. Strong linear patterns suggest high correlation (check the correlation matrix). Outliers or clusters may influence model training.
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Visualization error: {e}")
    
    st.session_state.pipeline.update({'target': target, 'features': features, 'df_features': df, 'features_engineered': True})
    if st.button("Next ➡️", key="feature_next"):
        st.session_state.pipeline['current_step'] = 4
        st.rerun()

def train_test_split_step():
    st.header("✂️ Step 4: Train/Test Split")
    if not st.session_state.pipeline['features_engineered']:
        st.warning("Complete feature engineering!")
        return
    df, target, features = (st.session_state.pipeline[k] for k in ['df_features', 'target', 'features'])
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, help="Percentage of data for testing.") / 100
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42, help="Seed for reproducibility.")
    
    try:
        X, y = df[features].dropna(), df[target].loc[df[features].dropna().index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        st.session_state.pipeline.update({
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'data_split': True
        })
        
        fig = px.pie(pd.DataFrame({'Set': ['Training', 'Testing'], 'Size': [len(X_train), len(X_test)]}), 
                     names='Set', values='Size', title='Train vs Test Split', width=400, height=400, 
                     color_discrete_sequence=['#f4a7b9', '#a7d8d3'])
        plot_config(fig, 'Train vs Test Split', '', '')
        st.plotly_chart(fig)
        st.markdown("""
            <div class="interpretation">
            **Interpretation**: This pie chart shows the proportion of data used for training vs. testing. A larger training set (coral) provides more data for model learning, while the test set (mint) evaluates performance on unseen data.
            </div>
        """, unsafe_allow_html=True)
        st.success("✅ Data split completed!")
        
        if st.button("Next ➡️", key="split_next"):
            st.session_state.pipeline['current_step'] = 5
            st.rerun()
    except Exception as e:
        st.error(f"❌ Split error: {e}")

def model_training_step():
    st.header("🤖 Step 5: Model Training")
    if not st.session_state.pipeline['data_split']:
        st.warning("Complete train/test split!")
        return
    X_train, y_train = st.session_state.pipeline['X_train'], st.session_state.pipeline['y_train']
    
    model_options = ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
    model_types = st.multiselect("Select Models", model_options, default=["Linear Regression"], 
                                 help="Choose models to train.")
    if not model_types:
        st.warning("Select a model!")
        return
    
    target_is_continuous = is_continuous(y_train)
    if "Linear Regression" in model_types and not target_is_continuous:
        st.warning("⚠️ Linear Regression needs continuous target!")
        return
    if "Logistic Regression" in model_types and target_is_continuous:
        st.warning("⚠️ Logistic Regression needs categorical target!")
        return
    
    models = {}
    if "K-Means Clustering" in model_types:
        n_clusters = st.number_input("Clusters", 2, 10, 3, key="clusters", help="Number of clusters for K-Means.")
        models["K-Means Clustering"] = KMeans(n_clusters=n_clusters, random_state=42)
    if "Linear Regression" in model_types:
        models["Linear Regression"] = LinearRegression()
    if "Logistic Regression" in model_types:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
    
    if st.button("Train Models", key="train"):
        with st.spinner("Training..."):
            try:
                for model_type, model in models.items():
                    model.fit(X_train, y_train if model_type != "K-Means Clustering" else X_train)
                st.session_state.pipeline.update({'models': models, 'model_trained': True})
                st.success("✅ Training completed!")
                
                with st.expander("Model Details"):
                    for model_type, model in models.items():
                        st.write(f"**{model_type}**")
                        if model_type in ["Linear Regression", "Logistic Regression"]:
                            st.dataframe(pd.DataFrame({
                                'Feature': ['Intercept'] + st.session_state.pipeline['features'],
                                'Coefficient': [model.intercept_] + list(model.coef_.flatten())
                            }))
                            st.markdown("""
                                <div class="interpretation">
                                **Interpretation**: Coefficients show the impact of each feature on the target. Positive values increase the target, negative values decrease it. Larger absolute values indicate stronger influence.
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.dataframe(pd.DataFrame(model.cluster_centers_, columns=st.session_state.pipeline['features']))
                            st.markdown("""
                                <div class="interpretation">
                                **Interpretation**: Cluster centers represent the average feature values for each cluster. Compare centers to understand how clusters differ (e.g., high vs. low volatility).
                                </div>
                            """, unsafe_allow_html=True)
                
                if st.button("Next ➡️", key="train_next"):
                    st.session_state.pipeline['current_step'] = 6
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Training error: {e}")

def evaluation_step():
    st.header("📊 Step 6: Model Evaluation")
    if not st.session_state.pipeline['model_trained']:
        st.warning("Train a model first!")
        return
    models, X_test, y_test = (st.session_state.pipeline[k] for k in ['models', 'X_test', 'y_test'])
    
    try:
        y_preds = {mt: m.predict(X_test) for mt, m in models.items()}
        st.session_state.pipeline['y_preds'] = y_preds
        
        metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'R²'])
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                mse = mean_squared_error(y_test, yp)
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Model': [mt], 'RMSE': [np.sqrt(mse)], 'R²': [r2_score(y_test, yp)]
                })], ignore_index=True)
        if not metrics_df.empty:
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df.style.format({'RMSE': '{:.4f}', 'R²': '{:.4f}'}), use_container_width=True)
            st.markdown("""
                <div class="interpretation">
                **Interpretation**: 
                - **RMSE**: Lower values indicate better prediction accuracy (less error).
                - **R²**: Values closer to 1 suggest the model explains more variance in the target. Negative R² indicates poor fit.
                </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure()
            bar_width = 0.35
            x = np.arange(len(metrics_df))
            fig.add_trace(go.Bar(
                x=x - bar_width/2, y=metrics_df['RMSE'], name='RMSE',
                marker_color='#f4a7b9', hovertemplate='RMSE: %{y:.4f}<br>Model: %{text}', text=metrics_df['Model']
            ))
            fig.add_trace(go.Bar(
                x=x + bar_width/2, y=metrics_df['R²'], name='R²',
                marker_color='#a7d8d3', hovertemplate='R²: %{y:.4f}<br>Model: %{text}', text=metrics_df['Model']
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="#999999", annotation_text="Ideal R²")
            plot_config(fig, 'Model Performance Comparison', 'Model', 'Value', 600, 400)
            fig.update_layout(barmode='group', xaxis=dict(tickvals=x, ticktext=metrics_df['Model']))
            st.plotly_chart(fig)
            st.markdown("""
                <div class="interpretation">
                **Interpretation**: The bar chart compares RMSE (coral) and R² (mint) across models. Lower RMSE and higher R² (closer to the dashed line at 1.0) indicate better performance. Hover to see exact values.
                </div>
            """, unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit', line=dict(color='#999999', dash='dash')))
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                fig.add_trace(go.Scatter(x=y_test, y=yp, mode='markers', name=f'{mt} Predictions', 
                                        marker=dict(size=8, opacity=0.7, color='#a7d8d3'),
                                        hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}'))
        plot_config(fig, 'Actual vs Predicted Values', 'Actual', 'Predicted', 600, 400)
        st.plotly_chart(fig)
        st.markdown("""
            <div class="interpretation">
            **Interpretation**: Points close to the dashed line (Ideal Fit) indicate accurate predictions. Scattered points suggest errors. Hover to see exact actual vs. predicted values. Consistent deviations may indicate model bias.
            </div>
        """, unsafe_allow_html=True)
        
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                residual_df = pd.DataFrame({'Predicted': yp, 'Residuals': y_test - yp})
                fig = px.scatter(residual_df, x='Predicted', y='Residuals', title=f'Residual Plot - {mt}', 
                                 width=600, height=400, color_discrete_sequence=['#a7d8d3'], 
                                 hover_data=['Predicted', 'Residuals'])
                fig.add_hline(y=0, line_dash="dash", line_color="#999999")
                plot_config(fig, f'Residual Plot - {mt}', 'Predicted', 'Residuals', 600, 400)
                st.plotly_chart(fig)
                st.markdown("""
                    <div class="interpretation">
                    **Interpretation**: Residuals (prediction errors) should be randomly scattered around zero. Patterns (e.g., curves, clusters) suggest the model misses trends. Large residuals indicate outliers or poor fit.
                    </div>
                """, unsafe_allow_html=True)
                
                fig = px.histogram(residual_df, x='Residuals', title=f'Prediction Error Distribution - {mt}', 
                                   nbins=30, color_discrete_sequence=['#a7d8d3'], opacity=0.7)
                plot_config(fig, f'Prediction Error Distribution - {mt}', 'Residuals', 'Count', 600, 400)
                st.plotly_chart(fig)
                st.markdown("""
                    <div class="interpretation">
                    **Interpretation**: The histogram shows the distribution of prediction errors. A peak near zero suggests an unbiased model. Skewed or wide distributions indicate systematic errors or high variance.
                    </div>
                """, unsafe_allow_html=True)
        
        if "K-Means Clustering" in models:
            st.subheader("K-Means Clustering Results")
            cluster_df = X_test.copy()
            cluster_df['Cluster'] = y_preds["K-Means Clustering"]
            features = st.session_state.pipeline['features']
            if len(features) >= 3:
                fig = px.scatter_3d(cluster_df, x=features[0], y=features[1], z=features[2], 
                                    color='Cluster', title='3D K-Means Clustering', width=600, height=400, 
                                    color_discrete_sequence=px.colors.qualitative.Pastel2, 
                                    hover_data=[features[0], features[1], features[2]])
                plot_config(fig, '3D K-Means Clustering', features[0], features[1])
                st.plotly_chart(fig)
                st.markdown("""
                    <div class="interpretation">
                    **Interpretation**: The 3D plot shows data grouped into clusters. Distinct clusters indicate clear separation based on features. Overlapping clusters suggest similar feature profiles. Hover to see feature values.
                    </div>
                """, unsafe_allow_html=True)
            elif len(features) >= 2:
                fig = px.scatter(cluster_df, x=features[0], y=features[1], color='Cluster', 
                                 title='K-Means Clustering', width=600, height=400, 
                                 color_discrete_sequence=px.colors.qualitative.Pastel2, 
                                 hover_data=[features[0], features[1]])
                plot_config(fig, 'K-Means Clustering', features[0], features[1], 600, 400)
                st.plotly_chart(fig)
                st.markdown("""
                    <div class="interpretation">
                    **Interpretation**: Each point represents a data point, colored by cluster. Well-separated clusters suggest meaningful groupings (e.g., high vs. low volatility periods). Hover to see feature values.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Need ≥2 features for clustering visualization.")
        
        st.session_state.pipeline['model_evaluated'] = True
        if st.button("Next ➡️", key="evaluation_next"):
            st.session_state.pipeline['current_step'] = 7
            st.rerun()
    except Exception as e:
        st.error(f"❌ Evaluation error: {e}")

def results_visualization_step():
    st.header("📈 Step 7: Results Visualization")
    if not st.session_state.pipeline['model_evaluated']:
        st.warning("Complete evaluation first!")
        return
    df, target, features, y_test, y_preds = (st.session_state.pipeline[k] for k in 
                                            ['df', 'target', 'features', 'y_test', 'y_preds'])
    
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        date_col = date_cols[0]
        results_df = pd.DataFrame({
            'Date': st.session_state.pipeline['X_test'].index.map(lambda x: df[date_col].iloc[x] if x < len(df) else df[date_col].iloc[-1]),
            'Actual': y_test.values
        })
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                results_df[f'Predicted_{mt}'] = yp
        
        model_options = [mt for mt in y_preds if mt != "K-Means Clustering"]
        selected_model = st.selectbox("Select Model for Visualization", model_options, index=0, 
                                      help="Choose a model to compare actual vs. predicted prices.")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Actual'], fill='tozeroy', name='Actual', 
            line=dict(color='#f4a7b9'), fillcolor='rgba(244,167,185,0.3)', 
            hovertemplate='Date: %{x}<br>Actual: %{y:.2f}'
        ))
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df[f'Predicted_{selected_model}'], fill='tozeroy', 
            name=f'Predicted ({selected_model})', line=dict(color='#a7d8d3'), fillcolor='rgba(167,216,211,0.3)', 
            hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}'
        ))
        max_idx = results_df['Actual'].idxmax()
        min_idx = results_df['Actual'].idxmin()
        fig.add_annotation(x=results_df['Date'][max_idx], y=results_df['Actual'][max_idx], 
                           text="Max Price", showarrow=True, arrowhead=2, ax=20, ay=-30)
        fig.add_annotation(x=results_df['Date'][min_idx], y=results_df['Actual'][min_idx], 
                           text="Min Price", showarrow=True, arrowhead=2, ax=20, ay=30)
        plot_config(fig, f'Actual vs Predicted {target} ({selected_model})', 'Date', target)
        st.plotly_chart(fig)
        st.markdown("""
            <div class="interpretation">
            **Interpretation**: The area plot compares actual (coral) and predicted (mint) prices over time. Overlapping areas indicate accurate predictions. Gaps show where the model over- or under-predicts. Annotations highlight the highest and lowest actual prices. Use the dropdown to switch models.
            </div>
        """, unsafe_allow_html=True)
        
        window = 20
        results_df['Actual_MA'] = results_df['Actual'].rolling(window=window).mean()
        results_df[f'Predicted_{selected_model}_MA'] = results_df[f'Predicted_{selected_model}'].rolling(window=window).mean()
        pred_std = results_df[f'Predicted_{selected_model}'].rolling(window=window).std()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Actual_MA'], mode='lines', 
            name='Actual (MA)', line=dict(color='#f4a7b9', width=3), 
            hovertemplate='Date: %{x}<br>Actual MA: %{y:.2f}'
        ))
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df[f'Predicted_{selected_model}_MA'], mode='lines', 
            name=f'Predicted ({selected_model}, MA)', line=dict(color='#a7d8d3', width=3), 
            hovertemplate='Date: %{x}<br>Predicted MA: %{y:.2f}'
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([results_df['Date'], results_df['Date'][::-1]]), 
            y=pd.concat([results_df[f'Predicted_{selected_model}_MA'] + pred_std, 
                         results_df[f'Predicted_{selected_model}_MA'][::-1] - pred_std[::-1]]), 
            fill='toself', fillcolor='rgba(167,216,211,0.2)', line=dict(color='rgba(255,255,255,0)'), 
            name='Prediction Confidence', hoverinfo='skip'
        ))
        plot_config(fig, f'Rolling Average ({window}-day) of Actual vs Predicted {target} ({selected_model})', 'Date', target)
        st.plotly_chart(fig)
        st.markdown("""
            <div class="interpretation">
            **Interpretation**: This plot shows a {window}-day moving average of actual (coral) and predicted (mint) prices, smoothing out short-term fluctuations. The shaded band represents prediction uncertainty (standard deviation). Close lines indicate good trend prediction. Use the dropdown to switch models.
            </div>
        """.format(window=window), unsafe_allow_html=True)
    
    for mt, model in st.session_state.pipeline['models'].items():
        if hasattr(model, 'coef_'):
            importance = pd.DataFrame({'Feature': features, 'Importance': model.coef_.flatten()}).sort_values('Importance', ascending=False)
            fig = px.bar(importance, x='Importance', y='Feature', title=f'Feature Importance ({mt})', 
                         color_discrete_sequence=['#a7d8d3'], hover_data=['Importance'])
            plot_config(fig, f'Feature Importance ({mt})', 'Importance', 'Feature')
            st.plotly_chart(fig)
            st.markdown("""
                <div class="interpretation">
                **Interpretation**: Bars show the impact of each feature on predictions. Longer bars (positive or negative) indicate stronger influence. Use this to understand which features drive the model's predictions (e.g., volatility vs. moving average).
                </div>
            """, unsafe_allow_html=True)
    
    hist_df = pd.DataFrame({'Value': y_test, 'Type': ['Actual'] * len(y_test)})
    for mt, yp in y_preds.items():
        if mt != "K-Means Clustering":
            hist_df = pd.concat([hist_df, pd.DataFrame({'Value': yp, 'Type': [f'Predicted ({mt})'] * len(yp)})], ignore_index=True)
    fig = px.histogram(hist_df, x='Value', color='Type', barmode='overlay', nbins=30, 
                       title='Distribution of Predicted vs Actual', 
                       color_discrete_map={'Actual': '#f4a7b9', f'Predicted ({selected_model})': '#a7d8d3'}, 
                       opacity=0.7, hover_data=['Value'])
    plot_config(fig, 'Distribution of Predicted vs Actual', target, 'Count')
    st.plotly_chart(fig)
    st.markdown("""
        <div class="interpretation">
        **Interpretation**: The histogram compares the distribution of actual (coral) and predicted (mint) values. Overlapping distributions suggest accurate predictions. Shifts or different shapes indicate model bias or variance issues.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="center-image"><img src="https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif" width="300"></div>', unsafe_allow_html=True)
    
    st.subheader("Download Results")
    for mt, yp in y_preds.items():
        if mt != "K-Means Clustering":
            results_df = pd.DataFrame({'Actual': y_test, f'Predicted_{mt}': yp, f'Residual_{mt}': y_test - yp})
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {mt} Results CSV 💾", csv, 
                              f'stock_predictions_{mt.lower().replace(" ", "_")}.csv', 'text/csv', 
                              key=f"download_{mt}_{uuid.uuid4()}")
        else:
            cluster_df = st.session_state.pipeline['X_test'].copy()
            cluster_df['Cluster'] = yp
            csv = cluster_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download K-Means Clustering CSV 💾", csv, 'stock_clusters_kmeans.csv', 
                              'text/csv', key=f"download_kmeans_{uuid.uuid4()}")
    
    st.session_state.pipeline['results_visualized'] = True
    st.success("🎉 Pipeline completed!")

# Main app
def main():
    init_session_state()
    
    with st.sidebar:
        st.header("💼 Pipeline Steps")
        steps = [
            ("📋 Welcome", 0, "Start your journey!", None),
            ("📊 Load Data", 1, "Upload/fetch data", None),
            ("🛠️ Preprocessing", 2, "Clean data", 'data_loaded'),
            ("📐 Feature Engineering", 3, "Create features", 'preprocessed'),
            ("✂️ Train/Test Split", 4, "Split data", 'features_engineered'),
            ("🤖 Model Training", 5, "Train models", 'data_split'),
            ("📊 Evaluation", 6, "Evaluate models", 'model_trained'),
            ("📈 Results Visualization", 7, "Visualize results", 'model_evaluated')
        ]
        progress = sum([st.session_state.pipeline.get(c, False) for _, _, _, c in steps if c]) / len([c for _, _, _, c in steps if c]) * 100
        st.markdown(f"<div class='progress-bar'><div class='progress-fill' style='width: {progress}%'></div></div>", unsafe_allow_html=True)
        for name, step, tooltip, condition in steps:
            disabled = False if condition is None else not st.session_state.pipeline.get(condition, False)
            label = f"{name} ✅" if condition and st.session_state.pipeline.get(condition, False) else name
            st.button(label, key=f"step_{step}", disabled=disabled, 
                      on_click=lambda s=step: st.session_state.pipeline.update({'current_step': s}), help=tooltip)
        st.divider()
        st.button("🔄 Reset Pipeline", key="reset", 
                  on_click=lambda: [st.session_state.clear(), init_session_state(), 
                                    st.session_state.pipeline.update({'current_step': 0})], 
                  help="Start over")
    
    step_funcs = [welcome_step, load_data_step, preprocessing_step, feature_engineering_step,
                  train_test_split_step, model_training_step, evaluation_step, results_visualization_step]
    step_funcs[st.session_state.pipeline['current_step']]()

if __name__ == "__main__":
    main()
