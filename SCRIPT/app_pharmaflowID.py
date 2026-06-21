import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
import os
import json
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Darts imports
try:
    from darts import TimeSeries
    from darts.models import TCNModel
    from darts.dataprocessing.transformers import Scaler
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    st.warning("⚠️ Darts library not found. Please install with: pip install darts")

# Page Configuration
st.set_page_config(
    page_title="Forecast Penyaluran Misoprostol", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (combined from both scripts)
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
        background-color: white;
        border-radius: 8px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
    }
    .control-chart-info {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 15px;
        border-bottom: 2px solid #007bff;
        padding-bottom: 5px;
    }
    .stat-item {
        margin: 8px 0;
        padding: 5px 0;
    }
    .stat-label {
        font-weight: 600;
        color: #6c757d;
    }
    .stat-value {
        font-weight: 700;
        color: #495057;
    }
    .anomaly-section {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #dee2e6;
    }
    .anomaly-item {
        margin: 5px 0;
        padding: 3px 0;
    }
    .warning-text {
        color: #856404;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df_data' not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

# =================== FUNGSI UTAMA ===================

@st.cache_data
def load_data():
    """Load the main dataset"""
    try:
        if os.path.exists('data/dfm2021-2024 (1).csv'):
            df = pd.read_csv('data/dfm2021-2024 (1).csv', delimiter = ",")
            # Handle mixed date formats
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='mixed', dayfirst=True, errors='coerce')
            # Remove any rows with invalid dates
            df = df.dropna(subset=['Tanggal'])
            
            # Clean PBF names by removing branch information
            df['Nama PBF'] = df['Nama PBF'].str.replace(r'\s*CABANG.*', '', case=False, regex=True).str.strip()
            
            # Filter only for the two specified PBFs
            target_pbfs = ['MERAPI UTAMA PHARMA', 'PARIT PADANG GLOBAL']
            df = df[df['Nama PBF'].isin(target_pbfs)]
            
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_pbf_model(pbf_name):
    """
    Load model, scaler, dan config berdasarkan nama PBF
    """
    # Mapping nama PBF ke folder model
    pbf_model_mapping = {
        "MERAPI UTAMA PHARMA": "merapi_utama_pharma",
        "PARIT PADANG GLOBAL": "parit_padang_global"
    }
    
    model_folder = pbf_model_mapping.get(pbf_name)
    if not model_folder:
        st.error(f"Model untuk {pbf_name} tidak tersedia!")
        return None, None, None
    
    base_path = f"models/{model_folder}"
    
    try:
        # Load konfigurasi
        config_path = f"{base_path}/model_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load TCN model
        model_path = f"{base_path}/tcn_model.pth"
        if os.path.exists(model_path):
            model = TCNModel.load(model_path)
        else:
            st.warning(f"Model file tidak ditemukan: {model_path}")
            model = None
        
        # Load scaler
        scaler_path = f"{base_path}/scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        st.success(f"✅ Model {pbf_name} berhasil dimuat!")
        return model, scaler, config
        
    except Exception as e:
        st.error(f"❌ Gagal memuat model {pbf_name}: {str(e)}")
        return None, None, None

def forecast_with_pretrained_model(ts_series, pbf_name, n_future=30):
    """
    Melakukan forecasting menggunakan model yang sudah di-train sebelumnya
    """
    # Load model untuk PBF tertentu
    model, scaler, config = load_pbf_model(pbf_name)
    
    if model is None or scaler is None:
        return None, None
    
    try:
        scaled_series = scaler.transform(ts_series)
        
        # Forecasting menggunakan model yang sudah di-train
        forecast = model.predict(n=n_future, num_samples=100)
        
        # Inverse transform untuk mendapatkan nilai asli
        forecast_original = scaler.inverse_transform(forecast)
        
        return forecast, config
        
    except Exception as e:
        st.error(f"Error forecasting dengan model {pbf_name}: {str(e)}")
        return None, None

def train_new_model(ts_series, pbf_name):
    """
    Train model baru jika diperlukan
    """
    try:
        # Normalisasi data
        scaler = Scaler()
        scaled_series = scaler.fit_transform(ts_series)
        
        # Inisialisasi model dengan parameter default
        model = TCNModel(
            input_chunk_length=14,
            output_chunk_length=7,
            n_epochs=50,
            dropout=0.1,
            batch_size=64,
            dilation_base=2,
            kernel_size=3,
            num_filters=32,
            num_layers=3,
            weight_norm=True,
            optimizer_kwargs={'lr': 1e-3},
            random_state=42
        )
        
        # Training
        with st.spinner(f"Training model untuk {pbf_name}..."):
            model.fit(scaled_series, verbose=False)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def preprocess_pbf_data(pbf_df):
    """Preprocess data for a specific PBF"""
    try:
        ts = pbf_df.groupby("Tanggal")["Jumlah"].sum().reset_index()
        ts = ts.set_index("Tanggal").asfreq("D").fillna(0)
        ts['Jumlah_MA'] = ts['Jumlah'].rolling(window=7, min_periods=1).mean().round(0).astype(int)
        return ts
    except Exception as e:
        st.error(f"Error preprocessing: {str(e)}")
        return pd.DataFrame()

def analyze_control_chart(data):
    """Analyze control chart and detect anomalies"""
    mean = data.mean()
    std = data.std()
    ucl = mean + 3 * std  # Upper Control Limit
    lcl = max(0, mean - 3 * std)  # Lower Control Limit
    
    # Detect anomalies
    above_ucl = data[data > ucl]
    below_lcl = data[data < lcl]
    
    # Additional rules for anomaly detection
    # Rule 1: Points beyond control limits (already detected above)
    # Rule 2: 2 out of 3 consecutive points beyond 2-sigma
    warning_upper = mean + 2 * std
    warning_lower = max(0, mean - 2 * std)
    
    analysis = {
        'mean': mean,
        'std': std,
        'ucl': ucl,
        'lcl': lcl,
        'warning_upper': warning_upper,
        'warning_lower': warning_lower,
        'above_ucl': above_ucl,
        'below_lcl': below_lcl,
        'total_points': len(data),
        'anomaly_count': len(above_ucl) + len(below_lcl)
    }
    
    return analysis

def create_control_chart(data, title="Control Chart"):
    """Create an interactive control chart with Plotly"""
    mean = data.mean()
    std = data.std()
    ucl = mean + 3 * std
    lcl = max(0, mean - 3 * std)
    
    # Additional warning limits
    upper_warning = mean + 2 * std
    lower_warning = max(0, mean - 2 * std)
    
    # Create figure
    fig = go.Figure()
    
    # Add the data points
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines+markers',
        name='Data',
        line=dict(color='blue', width=1),
        marker=dict(size=6)
    ))
    
    # Add control limits and center line
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[ucl, ucl],
        mode='lines',
        name='UCL (μ + 3σ)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[lcl, lcl],
        mode='lines',
        name='LCL (μ - 3σ)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[mean, mean],
        mode='lines',
        name='CL (μ)',
        line=dict(color='green', width=2)
    ))
    
    # Add warning limits (optional)
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[upper_warning, upper_warning],
        mode='lines',
        name='Upper Warning (μ + 2σ)',
        line=dict(color='orange', width=1, dash='dot'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[lower_warning, lower_warning],
        mode='lines',
        name='Lower Warning (μ - 2σ)',
        line=dict(color='orange', width=1, dash='dot'),
        opacity=0.7
    ))
    
    # Highlight points outside control limits
    above_ucl = data[data > ucl]
    below_lcl = data[data < lcl]
    
    if not above_ucl.empty:
        fig.add_trace(go.Scatter(
            x=above_ucl.index,
            y=above_ucl.values,
            mode='markers',
            name='Above UCL',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    if not below_lcl.empty:
        fig.add_trace(go.Scatter(
            x=below_lcl.index,
            y=below_lcl.values,
            mode='markers',
            name='Below LCL',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Tanggal',
        yaxis_title='Jumlah Penyaluran',
        hovermode="x unified",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add annotations for control limits
    fig.add_annotation(
        x=data.index[-1],
        y=ucl,
        text=f"UCL: {ucl:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    fig.add_annotation(
        x=data.index[-1],
        y=mean,
        text=f"CL: {mean:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=40
    )
    
    fig.add_annotation(
        x=data.index[-1],
        y=lcl,
        text=f"LCL: {lcl:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=40
    )
    
    return fig

def create_control_chart_info_display(analysis):
    """Create a better formatted information display for control chart interpretation"""
    
    # Use Streamlit's native components instead of raw HTML
    st.markdown("### 📊 Panduan Interpretasi Control Chart")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Statistik Kontrol")
        st.metric("Central Line (CL)", f"{analysis['mean']:.2f}", 
                  help="Rata-rata historis penyaluran")
        st.metric("Upper Control Limit (UCL)", f"{analysis['ucl']:.2f}", 
                  help="μ + 3σ - Batas atas kendali")
        st.metric("Lower Control Limit (LCL)", f"{analysis['lcl']:.2f}", 
                  help="μ - 3σ - Batas bawah kendali")
    
    with col2:
        st.markdown("#### Status Kontrol")
        st.metric("Total Data Points", analysis['total_points'])
        st.metric("Anomali Terdeteksi", analysis['anomaly_count'])
        
        # Status indicator
        if analysis['anomaly_count'] == 0:
            st.success("✅ Proses dalam kendali statistik")
        else:
            st.warning(f"⚠️ {analysis['anomaly_count']} anomali terdeteksi")
    
    # Rules section with better formatting
    st.markdown("#### 🔍 Aturan Deteksi Anomali")
    
    rules_container = st.container()
    with rules_container:
        st.markdown("""
        1. **Titik di atas UCL**: Menandakan potensi penyaluran abnormal tinggi
        2. **Titik di bawah LCL**: Menandakan potensi penurunan penyaluran abnormal
        3. **9 titik berturut-turut di satu sisi**: Indikasi trend yang tidak normal
        """)
    
    # Detailed anomaly information
    if analysis['anomaly_count'] > 0:
        st.markdown("#### 🚨 Detail Anomali")
        
        if len(analysis['above_ucl']) > 0:
            with st.expander(f"📈 {len(analysis['above_ucl'])} titik di atas UCL"):
                anomaly_dates = analysis['above_ucl'].index.strftime('%Y-%m-%d').tolist()
                anomaly_values = analysis['above_ucl'].values.tolist()
                
                anomaly_df = pd.DataFrame({
                    'Tanggal': anomaly_dates,
                    'Nilai': [f"{val:.2f}" for val in anomaly_values],
                    'Status': ['⚠️ Di atas UCL'] * len(anomaly_dates)
                })
                st.dataframe(anomaly_df, use_container_width=True)
        
        if len(analysis['below_lcl']) > 0:
            with st.expander(f"📉 {len(analysis['below_lcl'])} titik di bawah LCL"):
                anomaly_dates = analysis['below_lcl'].index.strftime('%Y-%m-%d').tolist()
                anomaly_values = analysis['below_lcl'].values.tolist()
                
                anomaly_df = pd.DataFrame({
                    'Tanggal': anomaly_dates,
                    'Nilai': [f"{val:.2f}" for val in anomaly_values],
                    'Status': ['⚠️ Di bawah LCL'] * len(anomaly_dates)
                })
                st.dataframe(anomaly_df, use_container_width=True)

def create_enhanced_control_chart_info(analysis):
    """Alternative approach using Streamlit's info boxes and containers"""
    
    # Main info container
    with st.container():
        st.markdown("---")
        st.markdown("### 📊 Analisis Statistical Process Control")
        
        # Key metrics in columns
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Mean (μ)", f"{analysis['mean']:.1f}")
        with metric_col2:
            st.metric("Std Dev (σ)", f"{analysis['std']:.1f}")
        with metric_col3:
            st.metric("UCL", f"{analysis['ucl']:.1f}")
        with metric_col4:
            st.metric("LCL", f"{analysis['lcl']:.1f}")
        
        # Control status
        if analysis['anomaly_count'] == 0:
            st.success("🎯 **Proses Terkendali**: Semua data points berada dalam batas kontrol statistik")
        else:
            st.error(f"🚨 **Anomali Terdeteksi**: {analysis['anomaly_count']} titik berada di luar batas kontrol")
            
            # Show anomaly details in tabs
            if len(analysis['above_ucl']) > 0 or len(analysis['below_lcl']) > 0:
                tab1, tab2 = st.tabs(["📈 Above UCL", "📉 Below LCL"])
                
                with tab1:
                    if len(analysis['above_ucl']) > 0:
                        st.warning(f"**{len(analysis['above_ucl'])} data points** above Upper Control Limit")
                        dates_above = analysis['above_ucl'].index.strftime('%Y-%m-%d')[:5]  # Show first 5
                        st.write("Tanggal:", ", ".join(dates_above))
                        if len(analysis['above_ucl']) > 5:
                            st.write(f"...dan {len(analysis['above_ucl']) - 5} lainnya")
                    else:
                        st.info("Tidak ada data points di atas UCL")
                
                with tab2:
                    if len(analysis['below_lcl']) > 0:
                        st.warning(f"**{len(analysis['below_lcl'])} data points** below Lower Control Limit")
                        dates_below = analysis['below_lcl'].index.strftime('%Y-%m-%d')[:5]  # Show first 5
                        st.write("Tanggal:", ", ".join(dates_below))
                        if len(analysis['below_lcl']) > 5:
                            st.write(f"...dan {len(analysis['below_lcl']) - 5} lainnya")
                    else:
                        st.info("Tidak ada data points di bawah LCL")
        
        # Interpretation guide
        with st.expander("📚 Panduan Interpretasi Control Chart"):
            st.markdown("""
            **Control Chart** adalah tool statistik untuk memantau stabilitas proses:
            
            - **Central Line (CL)**: Rata-rata proses historis
            - **Upper Control Limit (UCL)**: Batas atas (μ + 3σ)
            - **Lower Control Limit (LCL)**: Batas bawah (μ - 3σ)
            
            **Indikator Out of Control:**
            1. Titik di luar batas kontrol (beyond UCL/LCL)
            2. 7+ titik berturut-turut naik/turun
            3. 9+ titik berturut-turut di satu sisi CL
            4. 2/3 titik di zona peringatan (±2σ)
            
            **Tindakan yang Disarankan:**
            - Jika **in control**: Lanjutkan monitoring rutin
            - Jika **out of control**: Investigasi penyebab khusus
            """)

def display_control_chart_analysis(ts_data):
    """Main function to display control chart analysis"""
    analysis = analyze_control_chart(ts_data)
    
    # Choose one of the approaches:
    # Approach 1: Using Streamlit native components
    create_control_chart_info_display(analysis)
    
    # OR Approach 2: Enhanced version with tabs
    # create_enhanced_control_chart_info(analysis)
    
    return analysis

# =================== MAIN APP ===================

# Header
st.markdown('<div class="main-header">📦 Forecast Penyaluran Obat Misoprostol</div>', 
            unsafe_allow_html=True)
st.markdown("""
Aplikasi ini memprediksi penyaluran obat Misoprostol menggunakan model Temporal Convolutional Networks (TCN) 
yang telah di-training khusus untuk setiap PBF.
""")

# Load data
df = load_data()
if df.empty:
    st.warning("⚠️ Data tidak ditemukan. Silakan upload file data atau periksa path file.")
    st.stop()

# Sidebar
st.sidebar.header("Pengaturan")
target_pbf_list = ['MERAPI UTAMA PHARMA', 'PARIT PADANG GLOBAL']
available_pbf = [pbf for pbf in target_pbf_list if pbf in df['Nama PBF'].unique()]
selected_pbf = st.sidebar.selectbox("Pilih Nama PBF", available_pbf, key="pbf_selector")

# Forecast options
forecast_mode = st.sidebar.radio(
    "Mode Forecasting",
    ["Gunakan Model Tersimpan", "Train Model Baru"],
    help="Pilih apakah akan menggunakan model yang sudah di-train atau train model baru"
)

forecast_days = st.sidebar.slider("Periode Forecast (Hari):", 
                                  min_value=7, max_value=90, value=30)

# Model info
with st.sidebar.expander("ℹ️ Info Model"):
    model_info, scaler_info, config_info = load_pbf_model(selected_pbf)
    if config_info:
        st.write(f"**Model:** {config_info.get('model_type', 'TCN')}")
        st.write(f"**Input Length:** {config_info.get('input_chunk_length', 14)}")
        st.write(f"**MA Window:** {config_info.get('ma_window', 7)}")
        st.write(f"**Last Updated:** {config_info.get('saved_date', 'Unknown')}")
    else:
        st.write("Model belum tersedia atau belum di-load")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Data Preview", "➕ Input Data"])

with tab1:
    st.markdown('<div class="section-header">Dashboard Overview</div>', 
                unsafe_allow_html=True)

    # Filter data for selected PBF
    pbf_df = df[df["Nama PBF"] == selected_pbf].copy()
    ts = preprocess_pbf_data(pbf_df)
    
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Sedang memproses forecast..."):
            # Visualisasi Data Historis
            st.markdown('<div class="section-header">Riwayat Penyaluran</div>', 
                        unsafe_allow_html=True)
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=ts.index,
                y=ts['Jumlah'],
                name='Jumlah Aktual',
                line=dict(color='blue', width=1)
            ))
            fig_hist.add_trace(go.Scatter(
                x=ts.index,
                y=ts['Jumlah_MA'],
                name='7-Day MA',
                line=dict(color='red', width=2)
            ))
            fig_hist.update_layout(
                xaxis_title='Tanggal',
                yaxis_title='Jumlah Penyaluran',
                hovermode="x unified"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Forecasting
            ts_series = TimeSeries.from_dataframe(ts, None, 'Jumlah_MA')
            
            if DARTS_AVAILABLE and forecast_mode == "Gunakan Model Tersimpan":
                forecast, config = forecast_with_pretrained_model(ts_series, selected_pbf, forecast_days)
            elif DARTS_AVAILABLE and forecast_mode == "Train Model Baru":
                model, scaler = train_new_model(ts_series, selected_pbf)
                if model is None:
                    st.error("❌ Gagal training model baru.")
                    st.stop()
                
                scaled_series = scaler.transform(ts_series)
                forecast = model.predict(n=forecast_days, num_samples=100)
                config = {"model_type": "TCN (New)", "ma_window": 7}
            else:
                st.warning("Darts library tidak tersedia atau mode forecasting tidak valid. Tidak dapat melakukan forecast.")
                forecast = None

            if forecast is not None:
                # Visualisasi Forecast
                st.markdown('<div class="section-header">Hasil Forecast</div>', 
                            unsafe_allow_html=True)
                
                last_date = ts.index[-1]
                pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
                
                forecast_values = forecast.values().flatten().round(0).astype(int)
                df_pred = pd.DataFrame({
                    "Tanggal": pred_dates,
                    "Prediksi": forecast_values,
                    "Upper Bound": forecast_values * 1.2,
                    "Lower Bound": forecast_values * 0.8
                }).set_index("Tanggal")

                fig_forecast = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_forecast.add_trace(go.Scatter(
                    x=ts.index,
                    y=ts['Jumlah_MA'],
                    name='Data Historis (MA7)',
                    line=dict(color='blue', width=2)
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=df_pred.index,
                    y=df_pred["Prediksi"],
                    name=f'Prediksi {selected_pbf}',
                    line=dict(color='green', width=3)
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=df_pred.index.tolist() + df_pred.index[::-1].tolist(),
                    y=df_pred["Upper Bound"].tolist() + df_pred["Lower Bound"][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    hoverinfo="skip"
                ))
                
                fig_forecast.update_layout(
                    title=f"Forecast Penyaluran {selected_pbf} ({forecast_days} Hari)",
                    xaxis_title='Tanggal',
                    yaxis_title='Jumlah Penyaluran',
                    hovermode="x unified",
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Tabel Forecast
                df_forecast_table = pd.DataFrame({
                    'Tanggal': pred_dates.strftime('%Y-%m-%d'),
                    'Jumlah Prediksi': np.round(forecast_values, 2)
                })
                
                st.dataframe(
                    df_forecast_table.style.format({'Jumlah Prediksi': '{:,.0f}'}),
                    use_container_width=True,
                    height=400
                )
                
                # ---
                # Control Chart
                # ---
                st.markdown('<div class="section-header">📊 Diagram Kontrol</div>', 
                            unsafe_allow_html=True)
                
                control_chart_fig = create_control_chart(
                    ts['Jumlah_MA'], 
                    title=f"Control Chart Penyaluran {selected_pbf}"
                )
                st.plotly_chart(control_chart_fig, use_container_width=True)
                
                # Control Chart Analysis (Panduan Interpretasi Control Chart)
                analysis = analyze_control_chart(ts['Jumlah_MA'])
                display_control_chart_analysis(ts['Jumlah_MA'])

                # Additional improvements for better UX:
                if analysis['anomaly_count'] > 0:
                    st.markdown("#### 💡 Rekomendasi Tindakan")
                    
                    recommendations = []
                    if len(analysis['above_ucl']) > 0:
                        recommendations.append("🔍 **Investigasi penyaluran tinggi**: Periksa faktor penyebab lonjakan permintaan")
                    
                    if len(analysis['below_lcl']) > 0:
                        recommendations.append("📉 **Evaluasi penurunan**: Analisis penyebab rendahnya penyaluran")
                    
                    if analysis['anomaly_count'] > 5:
                        recommendations.append("⚙️ **Review proses**: Pertimbangkan penyesuaian parameter kontrol")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")

                # Optional: Add download button for anomaly report
                if analysis['anomaly_count'] > 0:
                    anomaly_report = []
                    
                    if len(analysis['above_ucl']) > 0:
                        for date, value in analysis['above_ucl'].items():
                            anomaly_report.append({
                                'Tanggal': date.strftime('%Y-%m-%d'),
                                'Nilai': value,
                                'Tipe': 'Above UCL',
                                'UCL': analysis['ucl'],
                                'Selisih': value - analysis['ucl']
                            })
                    
                    if len(analysis['below_lcl']) > 0:
                        for date, value in analysis['below_lcl'].items():
                            anomaly_report.append({
                                'Tanggal': date.strftime('%Y-%m-%d'),
                                'Nilai': value,
                                'Tipe': 'Below LCL',
                                'LCL': analysis['lcl'],
                                'Selisih': analysis['lcl'] - value
                            })
                    
                    if anomaly_report:
                        anomaly_df = pd.DataFrame(anomaly_report)
                        csv = anomaly_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Laporan Anomali",
                            data=csv,
                            file_name=f"anomaly_report_{selected_pbf}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

with tab2:
    st.markdown('<div class="section-header">Data Preview</div>', 
                unsafe_allow_html=True)
    
    pbf_df = df[df["Nama PBF"] == selected_pbf].copy()
    pbf_df['Jumlah'] = pbf_df['Jumlah'].astype(int)
    st.dataframe(
        pbf_df.head(10).style.background_gradient(cmap='Blues', subset=['Jumlah']),
        use_container_width=True,
        height=300
    )
    
    # Top 5 Daerah
    st.markdown('<div class="section-header">Top 5 Daerah Penyaluran</div>', 
                unsafe_allow_html=True)
    
    top_city = pbf_df.groupby("Kabupaten/Kota")["Jumlah"].sum().sort_values(ascending=False).head(5)
    top_city = top_city.round(0).astype(int)
    fig_city = go.Figure(go.Bar(
        x=top_city.values,
        y=top_city.index,
        orientation='h',
        marker_color='skyblue'
    ))
    fig_city.update_layout(
        xaxis_title='Total Jumlah Penyaluran',
        yaxis_title='Kabupaten/Kota',
        height=400
    )
    st.plotly_chart(fig_city, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Input Data Baru</div>', 
                unsafe_allow_html=True)
    
    with st.form("new_data_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            tanggal = st.date_input("Tanggal Penyaluran", datetime.now())
            tujuan_penyaluran = st.text_input("Tujuan Penyaluran")
            alamat_tujuan = st.text_input("Alamat Tujuan") 
            nama_zat_aktif = st.text_input("Nama Zat Aktif", value="Misoprostol")
            nama_obat_jadi = st.text_input("Nama Obat Jadi")
        
        with col2:
            nama_pbf = st.selectbox("Nama PBF", target_pbf_list, 
                                     index=target_pbf_list.index(selected_pbf))
            provinsi = st.text_input("Provinsi")
            kabupaten = st.text_input("Kabupaten/Kota")
            jumlah = st.number_input("Jumlah", min_value=0, step=1)
        
        submit = st.form_submit_button("Tambahkan Data", type="primary")

        if submit:
            required_fields = [tanggal, tujuan_penyaluran, alamat_tujuan, nama_zat_aktif, 
                               nama_obat_jadi, nama_pbf, provinsi, kabupaten, jumlah]
            field_names = ["Tanggal", "Tujuan Penyaluran", "Alamat Tujuan", "Nama Zat Aktif",
                          "Nama Obat Jadi", "Nama PBF", "Provinsi", "Kabupaten/Kota", "Jumlah"]
            
            empty_fields = []
            for i, field in enumerate(required_fields):
                if not field or (isinstance(field, str) and field.strip() == ""):
                    empty_fields.append(field_names[i])
            
            if empty_fields:
                st.warning(f"Harap isi field berikut: {', '.join(empty_fields)}")
            else:
                new_data = {
                    "Tanggal": pd.to_datetime(tanggal).strftime('%Y-%m-%d'),
                    "Tujuan Penyaluran": tujuan_penyaluran.strip(),
                    "Alamat Tujuan": alamat_tujuan.strip(),
                    "Nama Zat Aktif": nama_zat_aktif.strip(),
                    "Nama Obat Jadi": nama_obat_jadi.strip(),
                    "Nama PBF": nama_pbf.strip(),
                    "Provinsi": provinsi.strip(),
                    "Kabupaten/Kota": kabupaten.strip(),
                    "Jumlah": int(jumlah)
                }
                
                try:
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                    df.to_csv('data/dfm2021-2024.csv', index=False)
                    
                    st.success("✅ Data berhasil ditambahkan!")
                    st.balloons()
                    
                    st.write("**Data yang ditambahkan:**")
                    st.dataframe(pd.DataFrame([new_data]), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Gagal menyimpan data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**📊 Forecasting Application | Powered by Darts & Streamlit**")