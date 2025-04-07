import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="E-commerce Data Analysis",
    page_icon="🛍️",
    layout="wide"
)

# Title and introduction
st.title("🛍️ E-commerce Data Analysis Dashboard")
st.markdown("""
This dashboard analyzes e-commerce data to provide insights about product performance 
and delivery logistics across different regions.
""")

# Function to load data
@st.cache_data
def load_data():
    cumers_df = pd.read_csv("D:\proyek-analisis-data\proyek-analisis-data-main\dashboard\customers_dataset.csv")
    orders_df = pd.read_csv("D:\proyek-analisis-data\proyek-analisis-data-main\dashboard\orders_dataset.csv")
    order_reviews_df = pd.read_csv("D:\proyek-analisis-data\proyek-analisis-data-main\dashboard\order_reviews_dataset.csv")
    sellers_df = pd.read_csv("D:\proyek-analisis-data\proyek-analisis-data-main\dashboard\sellers_dataset.csv")
    order_items_df = pd.read_csv("D:\proyek-analisis-data\proyek-analisis-data-main\dashboard\order_items_dataset.csv")
    products_df = pd.read_csv("D:\proyek-analisis-data\proyek-analisis-data-main\dashboard\products_dataset.csv")
    
    # Convert date columns
    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    
    for col in date_columns:
        if col in orders_df.columns:
            orders_df[col] = pd.to_datetime(orders_df[col])
    
    return customers_df, orders_df, order_reviews_df, sellers_df, order_items_df, products_df

# Load data
try:
    with st.spinner('Loading data...'):
        customers_df, orders_df, order_reviews_df, sellers_df, order_items_df, products_df = load_data()
    st.success('Data loaded successfully!')
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar
st.sidebar.header("Navigation")
analysis_type = st.sidebar.radio(
    "Choose Analysis",
    ["Product Category Performance", "Delivery Logistics", "Customer RFM Analysis"]
)

# Add date range filter in sidebar
st.sidebar.header("Date Filter")
min_date = orders_df['order_purchase_timestamp'].min()
max_date = orders_df['order_purchase_timestamp'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Convert selected dates to datetime
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_orders = orders_df[
        (orders_df['order_purchase_timestamp'].dt.date >= start_date) &
        (orders_df['order_purchase_timestamp'].dt.date <= end_date)
    ]
else:
    filtered_orders = orders_df

# Product Category Performance Analysis
if analysis_type == "Product Category Performance":
    st.header("Product Category Performance Analysis")
    
    # Add price range filter
    price_range = st.slider(
        "Select Price Range ($)",
        float(order_items_df['price'].min()),
        float(order_items_df['price'].max()),
        (float(order_items_df['price'].min()), float(order_items_df['price'].max()))
    )
    
    # Add rating filter
    min_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.5)
    
    # Add category multiselect
    categories = products_df['product_category_name'].unique()
    selected_categories = st.multiselect(
        "Select Product Categories",
        options=categories,
        default=categories[:5]
    )
    
    # Merge and filter data
    merged_df = order_items_df.merge(filtered_orders, on='order_id')
    merged_df = merged_df.merge(order_reviews_df[['order_id', 'review_score']], on='order_id', how='left')
    merged_df = merged_df.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    
    # Apply filters
    merged_df = merged_df[
        (merged_df['price'].between(price_range[0], price_range[1])) &
        (merged_df['review_score'] >= min_rating) &
        (merged_df['product_category_name'].isin(selected_categories))
    ]
    
    # Calculate category performance
    category_performance = merged_df.groupby('product_category_name').agg(
        total_sales=('price', 'sum'),
        average_rating=('review_score', 'mean'),
        order_count=('order_id', 'count')
    ).reset_index()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Categories", len(category_performance))
    with col2:
        st.metric("Average Rating", f"{category_performance['average_rating'].mean():.2f}")
    with col3:
        st.metric("Total Sales", f"${category_performance['total_sales'].sum():,.2f}")
    
    # Visualizations
    chart_type = st.radio("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
    
    if chart_type == "Bar Chart":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=category_performance.sort_values(by='total_sales', ascending=False),
            x='total_sales',
            y='product_category_name',
            palette='viridis'
        )
        plt.title('Categories by Total Sales')
        plt.xlabel('Total Sales ($)')
        plt.ylabel('Product Category')
        st.pyplot(fig)
    
    elif chart_type == "Line Chart":
        # Time series of sales by category
        daily_sales = merged_df.groupby(['product_category_name', merged_df['order_purchase_timestamp'].dt.date])['price'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        for category in selected_categories:
            category_data = daily_sales[daily_sales['product_category_name'] == category]
            plt.plot(category_data['order_purchase_timestamp'], category_data['price'], label=category)
        plt.title('Daily Sales by Category')
        plt.xlabel('Date')
        plt.ylabel('Sales ($)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    else:  # Scatter Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            data=category_performance,
            x='total_sales',
            y='average_rating',
            size='order_count',
            hue='product_category_name'
        )
        plt.title('Sales vs Rating by Category')
        plt.xlabel('Total Sales ($)')
        plt.ylabel('Average Rating')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

# Delivery Logistics Analysis
elif analysis_type == "Delivery Logistics":
    st.header("Delivery Logistics Analysis")
    
    # Add state filter
    states = sellers_df['seller_state'].unique()
    selected_states = st.multiselect(
        "Select States",
        options=states,
        default=states[:5]
    )
    
    # Add delivery time filter
    max_delivery_days = st.slider(
        "Maximum Delivery Time (days)",
        0,
        100,
        50
    )
    
    # Prepare logistics data
    logistics_df = order_items_df.merge(
        filtered_orders[['order_id', 'order_delivered_customer_date', 'order_purchase_timestamp']], 
        on='order_id'
    )
    logistics_df = logistics_df.merge(sellers_df[['seller_id', 'seller_state']], on='seller_id', how='left')
    
    logistics_df['delivery_time_days'] = (
        logistics_df['order_delivered_customer_date'] - 
        logistics_df['order_purchase_timestamp']
    ).dt.days
    
    # Apply filters
    logistics_df = logistics_df[
        (logistics_df['seller_state'].isin(selected_states)) &
        (logistics_df['delivery_time_days'] <= max_delivery_days)
    ]
    
    state_delivery_performance = logistics_df.groupby('seller_state').agg(
        average_delivery_time=('delivery_time_days', 'mean'),
        total_orders=('order_id', 'count')
    ).reset_index()
    
    # Display interactive map or chart based on user selection
    visualization_type = st.radio("Select Visualization", ["Bar Chart", "Delivery Time Distribution"])
    
    if visualization_type == "Bar Chart":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=state_delivery_performance.sort_values(by='average_delivery_time', ascending=False),
            x='seller_state',
            y='average_delivery_time',
            palette='mako'
        )
        plt.title('Average Delivery Time by Seller State')
        plt.xlabel('Seller State')
        plt.ylabel('Average Delivery Time (Days)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=logistics_df, x='delivery_time_days', hue='seller_state', multiple="stack")
        plt.title('Distribution of Delivery Times by State')
        plt.xlabel('Delivery Time (Days)')
        plt.ylabel('Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

# Customer RFM Analysis
else:
    st.header("Customer RFM Analysis")
    
    # Add RFM score filters
    col1, col2, col3 = st.columns(3)
    with col1:
        recency_threshold = st.slider("Max Recency (days)", 0, 365, 180)
    with col2:
        frequency_threshold = st.slider("Min Frequency (orders)", 1, 10, 1)
    with col3:
        monetary_threshold = st.slider("Min Monetary Value ($)", 0, 1000, 100)
    
    # Prepare RFM data
    rfm_df = filtered_orders.merge(order_items_df, on='order_id')
    rfm_df = rfm_df.merge(customers_df, on='customer_id')
    
    reference_date = rfm_df['order_purchase_timestamp'].max() + timedelta(days=1)
    rfm_table = rfm_df.groupby('customer_unique_id').agg(
        Recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
        Frequency=('order_id', 'nunique'),
        Monetary=('price', 'sum')
    ).reset_index()
    
    # Apply filters
    rfm_table = rfm_table[
        (rfm_table['Recency'] <= recency_threshold) &
        (rfm_table['Frequency'] >= frequency_threshold) &
        (rfm_table['Monetary'] >= monetary_threshold)
    ]
    
    # Add customer segmentation
    segmentation_method = st.radio("Select Segmentation Method", ["Quantile-based", "Custom Thresholds"])
    
    if segmentation_method == "Quantile-based":
        try:
            # Handle Recency (lower is better)
            rfm_table['R_Score'] = pd.qcut(rfm_table['Recency'], q=4, labels=['4', '3', '2', '1'], duplicates='drop')
        except ValueError:
            rfm_table['R_Score'] = pd.cut(rfm_table['Recency'], 
                                        bins=[0, 30, 90, 180, float('inf')],
                                        labels=['4', '3', '2', '1'])
            
        try:
            # Handle Frequency (higher is better)
            rfm_table['F_Score'] = pd.qcut(rfm_table['Frequency'], q=4, labels=['1', '2', '3', '4'], duplicates='drop')
        except ValueError:
            # If qcut fails due to duplicates, use custom bins based on data distribution
            freq_bins = [0, 1, 2, 3, float('inf')]
            rfm_table['F_Score'] = pd.cut(rfm_table['Frequency'], 
                                        bins=freq_bins,
                                        labels=['1', '2', '3', '4'])
            
        try:
            # Handle Monetary (higher is better)
            rfm_table['M_Score'] = pd.qcut(rfm_table['Monetary'], q=4, labels=['1', '2', '3', '4'], duplicates='drop')
        except ValueError:
            monetary_median = rfm_table['Monetary'].median()
            monetary_75th = rfm_table['Monetary'].quantile(0.75)
            monetary_25th = rfm_table['Monetary'].quantile(0.25)
            rfm_table['M_Score'] = pd.cut(rfm_table['Monetary'],
                                        bins=[0, monetary_25th, monetary_median, monetary_75th, float('inf')],
                                        labels=['1', '2', '3', '4'])
    else:
        # Custom thresholds for segmentation
        r_threshold = st.slider("Recency Threshold (days)", 0, 365, 90)
        f_threshold = st.slider("Frequency Threshold (orders)", 1, 10, 2)
        m_threshold = st.slider("Monetary Threshold ($)", 0, 1000, 200)
        
        # Define scoring based on custom thresholds
        rfm_table['R_Score'] = np.select([
            rfm_table['Recency'] <= r_threshold/4,
            rfm_table['Recency'] <= r_threshold/2,
            rfm_table['Recency'] <= r_threshold
        ], ['4', '3', '2'], default='1')
        
        rfm_table['F_Score'] = np.select([
            rfm_table['Frequency'] >= f_threshold*4,
            rfm_table['Frequency'] >= f_threshold*2,
            rfm_table['Frequency'] >= f_threshold
        ], ['4', '3', '2'], default='1')
        
        rfm_table['M_Score'] = np.select([
            rfm_table['Monetary'] >= m_threshold*4,
            rfm_table['Monetary'] >= m_threshold*2,
            rfm_table['Monetary'] >= m_threshold
        ], ['4', '3', '2'], default='1')
    
    rfm_table['RFM_Score'] = rfm_table['R_Score'].astype(str) + rfm_table['F_Score'].astype(str) + rfm_table['M_Score'].astype(str)
    
    # Add customer segment labels
    def get_segment(row):
        score = int(row['R_Score']) + int(row['F_Score']) + int(row['M_Score'])
        if score >= 9:
            return 'Champions'
        elif score >= 7:
            return 'Loyal Customers'
        elif score >= 5:
            return 'Potential Loyalists'
        else:
            return 'Need Attention'
    
    rfm_table['Customer_Segment'] = rfm_table.apply(get_segment, axis=1)
    
    # Display segment distribution
    st.subheader("Customer Segments Distribution")
    
    # Allow users to choose visualization type
    viz_type = st.radio("Select Visualization", ["Segments", "RFM Scores", "Detailed Metrics"])
    
    if viz_type == "Segments":
        segment_dist = rfm_table['Customer_Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        segment_dist.plot(kind='bar', color=sns.color_palette("husl", 4))
        plt.title('Distribution of Customer Segments')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    elif viz_type == "RFM Scores":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=rfm_table, x='RFM_Score', hue='Customer_Segment', multiple="stack")
        plt.title('Distribution of RFM Scores by Segment')
        plt.xlabel('RFM Score')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
    else:
        # Show detailed metrics
        st.write("Detailed RFM Metrics by Segment")
        segment_metrics = rfm_table.groupby('Customer_Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'customer_unique_id': 'count'
        }).round(2)
        segment_metrics.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Customer Count']
        st.dataframe(segment_metrics)

# Footer
st.markdown("---")
st.markdown("Created by Sherina Yosephine | Data Analysis Project")
