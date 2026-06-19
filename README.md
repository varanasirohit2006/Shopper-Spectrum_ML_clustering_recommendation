# Shopper Spectrum: Customer Segmentation and Product Recommendations

Shopper Spectrum is an end-to-end Machine Learning and Collaborative Filtering web application designed to analyze transaction data from an online retail business. It discovers customer purchasing behaviors by segmenting them based on **RFM (Recency, Frequency, Monetary) analysis** and provides personalized **product recommendations** based on item similarity.

---

## 🚀 Live Demo & Screenshots

### 1. Home Dashboard
*Provides problem statement, business use cases, and dataset details.*

### 2. Customer Segmentation (Clustering)
*Accepts RFM values in real-time, scales the inputs, and predicts the shopper segment using a trained K-Means model.*

### 3. Product Recommender
*Accepts a product name and recommends the top 5 most similar products using collaborative filtering.*

---

## 🛠️ Key Features

- **RFM Analysis**: Segmentation based on:
  - **Recency**: Days since the last purchase.
  - **Frequency**: Number of purchases made.
  - **Monetary**: Total amount spent.
- **K-Means Clustering**: Unsupervised classification of shoppers (e.g., VIP, Occasional, At-Risk).
- **Item-based Collaborative Filtering**: Generates product recommendations based on item similarity scores.
- **Interactive UI**: Built with Streamlit for a fast, responsive, and beautiful user experience.
- **Optimized Storage**: Uses Gzipped serialization (`item_similarity.pkl.gz`) to efficiently store and load the 120MB similarity matrix in compliance with GitHub limits.

---

## 📦 Installation & Setup

Ensure you have **Python 3.8+** installed. Follow these steps to run the application locally:

### 1. Clone the Repository
```bash
git clone https://github.com/varanasirohit2006/Shopper-Spectrum_ML_clustering_recommendation.git
cd Shopper-Spectrum_ML_clustering_recommendation
```

### 2. Install Dependencies
Install all required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

Streamlit will automatically open a browser tab at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
├── E_Commerce_Recommendation_Segmentation.ipynb  # Jupyter Notebook containing ML training pipeline
├── Project Title.pdf                             # Documentation and presentation PDF
├── app.py                                        # Streamlit main application logic
├── scaler.pkl                                    # Fitted StandardScaler for RFM inputs
├── kmeans_model.pkl                              # Trained K-Means clustering model
├── cluster_labels.pkl                            # Mapping dictionary for cluster segments
├── product_names.pkl                             # List of valid product names for autocomplete
├── item_similarity.pkl.gz                        # Compressed item-similarity matrix (Gzipped)
├── online_retail.csv                             # Sample transactional dataset used
├── requirements.txt                              # Required packages and dependencies
└── README.md                                     # Project description and manual
```

---

## ⚙️ Technical Details

- **RFM Scaling**: Inputs are normalized using a pre-saved `StandardScaler` to ensure robust predictions.
- **Gzip Compression**: The similarity matrix (`item_similarity.pkl`) is compressed using `gzip` to reduce its footprint from **120MB** to **65.7MB**, ensuring seamless deployment and fast loading times.
- **Fallback Matcher**: The recommendation page includes a case-insensitive, partial-match lookup so users can find products even if they enter a partial name.
