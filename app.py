import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Pro",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional and stylish dark theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        background-attachment: fixed;
    }
    
    /* Animated Background Pattern */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(233, 69, 96, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(0, 210, 255, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(102, 126, 234, 0.1) 0%, transparent 30%),
            radial-gradient(circle at 60% 70%, rgba(255, 107, 107, 0.1) 0%, transparent 40%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Main Title */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00d2ff 0%, #e94560 25%, #667eea 50%, #0f3460 75%, #00d2ff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
        animation: glow 3s ease-in-out infinite alternate, shimmer 5s linear infinite;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(0, 210, 255, 0.5)); }
        to { filter: drop-shadow(0 0 40px rgba(233, 69, 96, 0.8)); }
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.3rem !important;
        background: linear-gradient(90deg, #a0a0a0, #e0e0e0, #a0a0a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 50%, #1e3a5f 100%) !important;
        border-right: 3px solid rgba(233, 69, 96, 0.5);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
        font-weight: 500;
        padding: 12px 20px;
        border-radius: 12px;
        transition: all 0.3s ease;
        margin: 5px 0;
        border: 1px solid transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: linear-gradient(90deg, rgba(233, 69, 96, 0.3) 0%, rgba(0, 210, 255, 0.3) 100%) !important;
        transform: translateX(8px);
        border: 1px solid rgba(0, 210, 255, 0.5);
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        background: transparent !important;
    }
    
    /* Sidebar Title */
    .sidebar-title {
        background: linear-gradient(90deg, #00d2ff, #e94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid rgba(233, 69, 96, 0.5);
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    /* Page Headers */
    .page-header {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #00d2ff !important;
        padding: 1rem 0;
        border-bottom: 3px solid #00d2ff;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.8);
    }
    
    /* Cards */
    .custom-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #ffffff !important;
    }
    
    .custom-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 210, 255, 0.3);
        border: 1px solid rgba(0, 210, 255, 0.5);
    }
    
    .custom-container * {
        color: #ffffff !important;
    }
    
    .custom-container p {
        color: #ffffff !important;
    }
    
    .custom-container span {
        color: #ffffff !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.2) 0%, rgba(233, 69, 96, 0.1) 100%);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 210, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stMetric"] .stMetricLabel {
        background: linear-gradient(90deg, #00d2ff, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }
    
    [data-testid="stMetric"] .stMetricValue {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.6);
        background: linear-gradient(90deg, #00d2ff 0%, #e94560 100%);
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(90deg, #667eea 0%, #0f3460 100%);
        color: white !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #00d2ff !important;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 0 8px rgba(0, 210, 255, 0.6);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.2) 0%, rgba(102, 126, 234, 0.2) 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00d2ff;
        margin: 1rem 0;
        color: #ffffff !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0, 255, 127, 0.2) 0%, rgba(0, 200, 83, 0.2) 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00ff7f;
        margin: 1rem 0;
        color: #ffffff !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(255, 0, 50, 0.2) 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
        color: #ffffff !important;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(233, 69, 96, 0.2) 0%, rgba(0, 210, 255, 0.2) 100%);
        border-radius: 12px !important;
        font-weight: 600;
        color: #ffffff !important;
    }
    
    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(0, 210, 255, 0.3);
        color: #ffffff;
    }
    
    [data-testid="stSelectbox"] label {
        color: #ffffff !important;
    }
    
    /* Slider */
    div.stSlider > div[data-baseweb="slider"] {
        background: rgba(255,255,255,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        background: linear-gradient(90deg, rgba(0, 210, 255, 0.1), rgba(233, 69, 96, 0.1));
        padding: 2rem 0;
        font-size: 0.9rem;
        border-radius: 15px;
        margin-top: 2rem;
    }
    
    .footer p {
        color: #ffffff !important;
    }
    
    /* Plotly Charts */
    .js-plotly-plot .plotly {
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Text colors for better visibility */
    .stMarkdown p {
        color: #ffffff !important;
    }
    
    .stMarkdown li {
        color: #ffffff !important;
    }
    
    .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Radio button text */
    .stRadio label {
        color: #ffffff !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden}
    .stApp > header {visibility: hidden}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px 4px 0 0;
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e94560, #0f3460);
    }
    
    /* DataFrame text */
    [data-testid="stDataFrame"] td {
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] th {
        background: rgba(0, 210, 255, 0.2) !important;
        color: #ffffff !important;
    }
    
    /* Multiselect */
    [data-testid="stMultiSelect"] label {
        color: #ffffff !important;
    }
    
    /* Number input */
    .stNumberInput > label {
        color: #ffffff !important;
    }
    
    /* All text elements */
    p, li, span, div {
        color: #ffffff !important;
    }
    
    /* All Headings - CRITICAL for visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #00d2ff !important;
        font-weight: 700 !important;
    }
    
    /* Specific heading colors */
    h1 { color: #00d2ff !important; font-size: 2.5rem !important; }
    h2 { color: #00d2ff !important; font-size: 2rem !important; }
    h3 { color: #00d2ff !important; font-size: 1.5rem !important; }
    h4 { color: #e94560 !important; font-size: 1.25rem !important; }
    h5 { color: #667eea !important; font-size: 1.1rem !important; }
    h6 { color: #ffffff !important; font-size: 1rem !important; }
    
    /* Streamlit native headings */
    .stMarkdown h1 { color: #00d2ff !important; }
    .stMarkdown h2 { color: #00d2ff !important; }
    .stMarkdown h3 { color: #00d2ff !important; }
    .stMarkdown h4 { color: #e94560 !important; }
    
    /* Table styling */
    table {
        color: #ffffff !important;
    }
    
    th {
        background: rgba(0, 210, 255, 0.2) !important;
        color: #ffffff !important;
    }
    
    td {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing clustered data
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None

# Title and description
st.markdown("""
<h1 style="text-align: center; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
    <span style="font-size: 3.5rem;">🛍️</span> 
    <span style="background: linear-gradient(90deg, #00d2ff, #e94560); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        Customer Segmentation Pro
    </span>
</h1>
""", unsafe_allow_html=True)

st.markdown('<p class="subtitle">✨ Advanced Analytics & Machine Learning for E-Commerce Customer Insights ✨</p>', unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        if hasattr(file_path, 'read'):
            if hasattr(file_path, 'name'):
                if file_path.name.endswith('.xlsx'):
                    return pd.read_excel(file_path)
                elif file_path.name.endswith('.csv'):
                    return pd.read_csv(file_path)
            file_path.seek(0)
            return pd.read_csv(file_path)
        elif isinstance(file_path, str):
            if file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Process data function
def process_data(df):
    df_processed = df.copy()
    if 'Gender' in df_processed.columns:
        df_processed['Gender'] = df_processed['Gender'].fillna(df_processed['Gender'].mode()[0])
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
    return df_processed

# Calculate elbow method using KneeLocator for accurate detection
def calculate_elbow(X_scaled, max_k=10):
    inertia = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Use KneeLocator for accurate elbow detection
    try:
        kl = KneeLocator(list(K_range), inertia, curve='convex', direction='decreasing')
        elbow_k = kl.elbow
    except Exception:
        # Fallback: find the point where inertia decrease slows down
        diffs = np.diff(inertia)
        diffs2 = np.diff(diffs)
        if len(diffs2) > 0:
            elbow_idx = np.argmax(diffs2) + 1
            elbow_k = list(K_range)[min(elbow_idx, len(K_range)-1)]
        else:
            elbow_k = 3  # Default fallback
    
    # Ensure elbow_k is within valid range
    if elbow_k is None:
        elbow_k = 3
    
    return list(K_range), inertia, silhouette_scores, elbow_k

# Sidebar for navigation
st.sidebar.markdown('<h1 class="sidebar-title">📊 Navigation</h1>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Data Upload & Overview", "Data Exploration", "Clustering Model", "Cluster Analysis"])

# Main data loading
default_file = "ecom customer_data.xlsx"

if st.session_state.df_clustered is not None:
    df = st.session_state.df_clustered
else:
    df = load_data(default_file)
    if df is not None:
        df = process_data(df)

if df is not None:
    # Page 1: Data Upload & Overview
    if page == "Data Upload & Overview":
        st.markdown('<h2 class="page-header">📁 Data Upload & Overview</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("📤 Upload your customer data (CSV or Excel)", 
                                             type=['csv', 'xlsx'])
            
            if uploaded_file is not None:
                df = load_data(uploaded_file)
                df = process_data(df)
                st.session_state.df_clustered = None
                st.markdown('<div class="success-box">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📊 Dataset Overview</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🔍 Dataset Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(min(10, len(df))), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📋 Column Information</h3>', unsafe_allow_html=True)
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(dtype) for dtype in df.dtypes.values],
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values
        })
        st.write(col_info)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📥 Download Clustered Data</h3>', unsafe_allow_html=True)
        if 'Cluster' in df.columns:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Clustered Data (CSV)",
                data=csv,
                file_name="clustered_customers.csv",
                mime="text/csv"
            )
            st.markdown('<div class="success-box">✅ Cluster data is available for download!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ No cluster data available yet. Please go to "Clustering Model" page to perform clustering.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Page 2: Data Exploration
    elif page == "Data Exploration":
        st.markdown('<h2 class="page-header">📈 Data Exploration</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🔍 Filter Data</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            rows_to_show = st.slider("Number of rows to display", 5, 50, 10)
        with col2:
            if 'Gender' in df.columns:
                gender_filter = st.multiselect("Filter by Gender", df['Gender'].unique())
                if gender_filter:
                    df_display = df[df['Gender'].isin(gender_filter)]
                else:
                    df_display = df
            else:
                df_display = df
        
        st.write(df_display.head(min(rows_to_show, len(df_display))))

        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📊 Numerical Statistics</h3>', unsafe_allow_html=True)
        st.write(df.describe().iloc[:10])
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">👥 Gender Distribution</h3>', unsafe_allow_html=True)
            if 'Gender' in df.columns:
                fig = px.pie(df, names='Gender', title='Customer Distribution by Gender',
                            color_discrete_sequence=['#00d2ff', '#667eea', '#764ba2'],
                            hole=0.5)
                fig.update_layout(
                    title_font_size=18, 
                    title_x=0.5,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">📦 Orders Distribution</h3>', unsafe_allow_html=True)
            if 'Orders' in df.columns:
                fig = px.histogram(df, x='Orders', nbins=20, title='Distribution of Orders',
                                  color_discrete_sequence=['#00d2ff'])
                fig.update_layout(
                    title_font_size=18, 
                    title_x=0.5,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False, color='white'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📈 Feature Distributions</h3>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_col = st.selectbox("Select feature to visualize", numeric_cols)
        fig = px.histogram(df, x=selected_col, nbins=30, title=f'Distribution of {selected_col}',
                          color_discrete_sequence=['#667eea'])
        fig.update_layout(
            title_font_size=18, 
            title_x=0.5,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, color='white'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🔥 Correlation Heatmap</h3>', unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax, 
                       annot_kws={'size': 10}, linewidths=0.5, cbar_kws={'shrink': 0.8})
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='600', color='white')
            ax.set_facecolor('none')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            plt.xticks(color='white')
            plt.yticks(color='white')
            fig.patch.set_facecolor('none')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Page 3: Clustering Model
    elif page == "Clustering Model":
        st.markdown('<h2 class="page-header">🎯 K-Means Clustering Model</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚙️ Select Features for Clustering</h3>', unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Cust_ID', 'CustomerID', 'Cluster']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) > 10:
            low_cardinality_cols = []
            for col in feature_cols:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    low_cardinality_cols.append(col)
            
            if low_cardinality_cols:
                feature_cols = low_cardinality_cols
            else:
                feature_cols = feature_cols[:5]
        
        st.write(f"📊 Available features: **{', '.join(feature_cols)}**")
        
        default_features = feature_cols[:2] if len(feature_cols) >= 2 else feature_cols
        
        selected_features = st.multiselect(
            "Select features for clustering:",
            options=feature_cols,
            default=default_features
        )
        
        if len(selected_features) < 2:
            st.markdown('<div class="warning-box">⚠️ Please select at least 2 features for clustering</div>', unsafe_allow_html=True)
        else:
            X = df[selected_features].copy()
            X = X.fillna(X.mean())
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">🔢 Determine Optimal Clusters</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_clusters = st.slider("Select number of clusters", 2, 10, 3)
            
            # Elbow Method
            st.markdown('<h3 class="section-header">📐 Elbow Method & Silhouette Score</h3>', unsafe_allow_html=True)
            
            with st.spinner("⚡ Calculating optimal clusters..."):
                K_range, inertia, silhouette_scores, elbow_k = calculate_elbow(X_scaled, max_k=6)
            
            st.success("✅ Analysis complete!")
            
            # Create beautiful plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Elbow Plot
                fig = go.Figure()
                
                # Add line with gradient effect
                fig.add_trace(go.Scatter(
                    x=list(K_range), 
                    y=inertia,
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='#00d2ff', width=4, shape='spline'),
                    marker=dict(size=12, color='#00d2ff', symbol='diamond')
                ))
                
                # Add elbow point
                elbow_idx = list(K_range).index(elbow_k)
                fig.add_trace(go.Scatter(
                    x=[elbow_k],
                    y=[inertia[elbow_idx]],
                    mode='markers',
                    name=f'Elbow K={elbow_k}',
                    marker=dict(size=25, color='#ff6b6b', symbol='star')
                ))
                
                fig.update_layout(
                    title=dict(text='<b>Elbow Method</b>', font=dict(size=18, color='white'), x=0.5),
                    xaxis_title=dict(text='Number of Clusters (K)', font=dict(color='white')),
                    yaxis_title=dict(text='Inertia', font=dict(color='white')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Silhouette Plot
                fig2 = go.Figure()
                
                best_k_sil = list(K_range)[np.argmax(silhouette_scores)]
                best_sil = max(silhouette_scores)
                
                fig2.add_trace(go.Scatter(
                    x=list(K_range), 
                    y=silhouette_scores,
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='#667eea', width=4, shape='spline'),
                    marker=dict(size=12, color='#667eea', symbol='diamond')
                ))
                
                fig2.add_trace(go.Scatter(
                    x=[best_k_sil],
                    y=[best_sil],
                    mode='markers',
                    name=f'Best K={best_k_sil}',
                    marker=dict(size=25, color='#ffd93d', symbol='star')
                ))
                
                fig2.update_layout(
                    title=dict(text='<b>Silhouette Score</b>', font=dict(size=18, color='white'), x=0.5),
                    xaxis_title=dict(text='Number of Clusters (K)', font=dict(color='white')),
                    yaxis_title=dict(text='Silhouette Score', font=dict(color='white')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Recommendations
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown(f'''
                <div class="success-box">
                    🎯 <b>Recommended K (Elbow):</b> {elbow_k}
                </div>
                ''', unsafe_allow_html=True)
            
            with col_rec2:
                st.markdown(f'''
                <div class="info-box">
                    ⭐ <b>Best K (Silhouette):</b> {best_k_sil} (Score: {best_sil:.4f})
                </div>
                ''', unsafe_allow_html=True)
            
            # Auto update slider
            n_clusters = elbow_k
            st.markdown(f'<div class="info-box">🔄 Using optimal K = {elbow_k} for clustering!</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Train model
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">🚀 Train K-Means Model</h3>', unsafe_allow_html=True)
            
            if st.button("🎯 Run Clustering", type="primary"):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                df['Cluster'] = kmeans.fit_predict(X_scaled)
                
                st.session_state.df_clustered = df.copy()
                
                centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
                centers_df = pd.DataFrame(centers_original, columns=selected_features)
                centers_df['Cluster'] = range(n_clusters)
                
                st.markdown(f'<div class="success-box">✅ Clustering complete! Generated {n_clusters} clusters.</div>', unsafe_allow_html=True)
                
                st.markdown('<h4>🎯 Cluster Centers</h4>', unsafe_allow_html=True)
                st.write(centers_df)
                
                st.markdown('<h4>📊 Cluster Distribution</h4>', unsafe_allow_html=True)
                cluster_counts = df['Cluster'].value_counts().sort_index()
                
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                            labels={'x': 'Cluster', 'y': 'Count'},
                            title='<b>Number of Customers in Each Cluster</b>',
                            color=cluster_counts.index,
                            color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_layout(
                    title_font_size=16, title_x=0.5,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    font=dict(color='white'),
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                df.to_csv("Cluster_data.csv", index=False)
                st.markdown('<div class="success-box">💾 Clustered data saved to Cluster_data.csv</div>', unsafe_allow_html=True)
                
                st.markdown('<h4>📥 Download Clustered Data</h4>', unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Clustered Data (CSV)",
                    data=csv,
                    file_name="clustered_customers.csv",
                    mime="text/csv"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Page 4: Cluster Analysis
    elif page == "Cluster Analysis":
        st.markdown('<h2 class="page-header">🔬 Cluster Analysis</h2>', unsafe_allow_html=True)
        
        if 'Cluster' not in df.columns:
            st.markdown('<div class="warning-box">⚠️ No cluster labels found! Please go to "Clustering Model" page first to perform clustering.</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">Or load existing cluster data:</h3>', unsafe_allow_html=True)
            if st.button("📂 Load Cluster_data.csv"):
                try:
                    df_existing = pd.read_csv("Cluster_data.csv")
                    if 'Cluster' in df_existing.columns:
                        df = df_existing
                        st.session_state.df_clustered = df
                        st.markdown('<div class="success-box">✅ Cluster data loaded successfully!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">❌ The file does not contain cluster data.</div>', unsafe_allow_html=True)
                except FileNotFoundError:
                    st.markdown('<div class="warning-box">❌ Cluster_data.csv not found.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">📋 Cluster Summary</h3>', unsafe_allow_html=True)
            
            cluster_summary = df.groupby('Cluster').agg({
                col: 'mean' for col in df.select_dtypes(include=[np.number]).columns 
                if col not in ['Cust_ID', 'CustomerID', 'Cluster']
            }).round(2)
            
            st.write(cluster_summary.T.iloc[:10])
            
            if 'Gender' in df.columns:
                st.markdown('<h3 class="section-header">👥 Cluster Distribution by Gender</h3>', unsafe_allow_html=True)
                gender_cluster = df.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
                fig = px.bar(gender_cluster, barmode='group', 
                            title='<b>Gender Distribution per Cluster</b>',
                            color_discrete_sequence=['#00d2ff', '#667eea'])
                fig.update_layout(
                    title_font_size=16, title_x=0.5,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    font=dict(color='white'),
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">📈 Feature Comparison</h3>', unsafe_allow_html=True)
            
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in ['Cust_ID', 'CustomerID', 'Cluster']]
            
            selected_feature = st.selectbox("Select feature to compare", numeric_cols)
            
            fig = px.box(df, x='Cluster', y=selected_feature, 
                        title=f'<b>{selected_feature} Distribution by Cluster</b>',
                        color='Cluster',
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(
                title_font_size=16, title_x=0.5,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)',
                font=dict(color='white'),
                xaxis=dict(color='white'),
                yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<h3 class="section-header">🎨 Cluster Visualization</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig = px.scatter(df, x=x_axis, y=y_axis, color='Cluster',
                           title=f'<b>Customer Clusters: {x_axis} vs {y_axis}</b>',
                           color_discrete_sequence=px.colors.qualitative.Bold,
                           hover_data=['Gender'] if 'Gender' in df.columns else None,
                           size_max=10)
            fig.update_layout(
                title_font_size=16, title_x=0.5,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)',
                font=dict(color='white'),
                xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<h3 class="section-header">👤 Cluster Profiles</h3>', unsafe_allow_html=True)
            
            for cluster in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cluster]
                with st.expander(f"📌 Cluster {cluster} Profile"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Customer Count", len(cluster_data))
                    if 'Gender' in cluster_data.columns:
                        with col2:
                            st.metric("Most Common Gender", cluster_data['Gender'].mode()[0])
                    if 'Orders' in cluster_data.columns:
                        with col3:
                            st.metric("Avg Orders", round(cluster_data['Orders'].mean(), 2))
                    
                    st.write("**Average Values:**")
                    st.write(cluster_data[numeric_cols].mean().to_frame('Average').T)
            
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">📥 Download Clustered Data</h3>', unsafe_allow_html=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Clustered Data (CSV)",
                data=csv,
                file_name="clustered_customers.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="warning-box">❌ Could not load the data. Please check the file path.</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🛍️ Customer Segmentation Pro | Built with Streamlit</p>
    <p>Powered by Machine Learning & Data Science</p>
</div>
""", unsafe_allow_html=True)

