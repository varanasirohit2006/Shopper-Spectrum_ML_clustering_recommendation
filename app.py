import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gzip

# Set page config with standard defaults
st.set_page_config(
    page_title="Shopper Spectrum",
    layout="centered"
)

# Helper function to load cached model assets
@st.cache_resource
def load_assets():
    workspace = os.path.dirname(os.path.abspath(__file__))
    
    # Load scaler & clustering model
    with open(os.path.join(workspace, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(workspace, "kmeans_model.pkl"), "rb") as f:
        kmeans = pickle.load(f)
    with open(os.path.join(workspace, "cluster_labels.pkl"), "rb") as f:
        cluster_labels_map = pickle.load(f)
        
    # Load recommendation assets
    with open(os.path.join(workspace, "product_names.pkl"), "rb") as f:
        product_names = pickle.load(f)
    with gzip.open(os.path.join(workspace, "item_similarity.pkl.gz"), "rb") as f:
        item_similarity = pickle.load(f)
        
    return scaler, kmeans, cluster_labels_map, product_names, item_similarity

# Load assets
try:
    scaler, kmeans, cluster_labels_map, product_names, item_similarity = load_assets()
except Exception as e:
    st.error(f"Error loading models and data: {e}")
    st.stop()

# Sidebar Navigation (Default Streamlit Radio styling)
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Clustering", "Recommendation"],
    index=0
)

# ----------------------------------------------------
# 1. HOME PAGE
# ----------------------------------------------------
if page == "Home":
    st.title("Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce")
    
    st.subheader("Problem Statement")
    st.write(
        "The global e-commerce industry generates vast amounts of transaction data daily, offering "
        "valuable insights into customer purchasing behaviors. Analyzing this data is essential for "
        "identifying meaningful customer segments and recommending relevant products to enhance "
        "customer experience and drive business growth. This project aims to examine transaction "
        "data from an online retail business to uncover patterns in customer purchase behavior, "
        "segment customers based on Recency, Frequency, and Monetary (RFM) analysis, and "
        "develop a product recommendation system using collaborative filtering techniques."
    )
    
    st.subheader("Real-time Business Use Cases:")
    st.write("- Customer Segmentation for Targeted Marketing Campaigns")
    st.write("- Personalized Product Recommendations on E-Commerce Platforms")
    st.write("- Identifying At-Risk Customers for Retention Programs")
    st.write("- Dynamic Pricing Strategies Based on Purchase Behavior")
    st.write("- Inventory Management and Stock Optimization Based on Customer Demand Patterns")
    
    st.subheader("Dataset Description")
    dataset_schema = [
        {"Column": "InvoiceNo", "Description": "Transaction number"},
        {"Column": "StockCode", "Description": "Unique product/item code"},
        {"Column": "Description", "Description": "Name of the product"},
        {"Column": "Quantity", "Description": "Number of products purchased"},
        {"Column": "InvoiceDate", "Description": "Date and time of transaction (2022–2023)"},
        {"Column": "UnitPrice", "Description": "Price per product"},
        {"Column": "CustomerID", "Description": "Unique identifier for each customer"},
        {"Column": "Country", "Description": "Country where the customer is based"}
    ]
    st.table(pd.DataFrame(dataset_schema))

# ----------------------------------------------------
# 2. CLUSTERING PAGE
# ----------------------------------------------------
elif page == "Clustering":
    st.title("Customer Segmentation")
    
    # Inputs with default values matching PDF
    recency = st.number_input(
        "Recency (days since last purchase)",
        value=325
    )
    
    frequency = st.number_input(
        "Frequency (number of purchases)",
        value=1
    )
    
    monetary = st.number_input(
        "Monetary (total spend)",
        value=765322.00,
        format="%.2f"
    )
    
    # Predict button
    if st.button("Predict Segment"):
        # Transform and predict
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        scaled_input = scaler.transform(input_df)
        cluster_id = kmeans.predict(scaled_input)[0]
        segment_label = cluster_labels_map.get(cluster_id, "Unknown")
        
        # Display output in a simple code box to resemble the small box in PDF
        st.code(str(cluster_id), language="")
        st.write(f"This customer belongs to: {segment_label} Shopper")

# ----------------------------------------------------
# 3. RECOMMENDATION PAGE
# ----------------------------------------------------
elif page == "Recommendation":
    st.title("Product Recommender")
    
    # Text input box for product name matching PDF Page 7
    product_name_input = st.text_input(
        "Enter Product Name",
        value="GREEN VINTAGE SPOT BEAKER"
    )
    
    # Recommend button
    if st.button("Recommend"):
        target_product = product_name_input.strip()
        
        if target_product not in item_similarity.index:
            # Try partial match (case insensitive)
            matches = [p for p in item_similarity.index if target_product.upper() in p.upper()]
            if matches:
                target_product = matches[0]
            else:
                target_product = None
                
        if target_product:
            # Generate recommendations
            scores = item_similarity[target_product].drop(target_product).sort_values(ascending=False)
            top_recos = list(scores.head(5).index)
            
            # Display outputs exactly as vertical text list
            st.write("Recommended Products:")
            for p in top_recos:
                st.write(p)
        else:
            st.error(f"Product '{product_name_input}' not found in the database. Please try another term.")
