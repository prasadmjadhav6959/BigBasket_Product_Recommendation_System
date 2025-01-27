import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Configure the Streamlit page
st.set_page_config(
    page_title="Big Basket Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# Load and display an image 
image_path = "BigBasket_Logo.png"
image = Image.open(image_path)
st.image(image, use_column_width=True) 

# Load data function with caching
@st.cache_data(ttl=48 * 3600)
def load_data():
    # Load and clean the dataset
    df = pd.read_csv('df_cleaned_1.csv')
    df = df[['product', 'rating', 'sale_price', 'market_price', 'soup']]
    return df[:10000]

df2 = load_data()

# Recommendation function
@st.cache_data(ttl=48 * 3600)
def get_recommendations(product_name, df):
    # Create a CountVectorizer matrix based on 'soup'
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Map product names to indices
    indices = pd.Series(df.index, index=df['product']).drop_duplicates()

    # Check if the product exists in the dataset
    if product_name not in indices:
        return pd.DataFrame()  # Return an empty DataFrame if the product is not found

    # Get the index of the selected product
    idx = indices[product_name]

    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    # Get the indices of the top 10 most similar products
    product_indices = [i[0] for i in sim_scores]

    # Return the recommended products
    return df.iloc[product_indices][['product', 'rating', 'sale_price', 'market_price']]

# CSS for custom styling
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2rem;
        color: #84bd00;
        font-weight: bold;
    }
    .stButton button {
        background-color: #84bd00;
        color: black;
        font-size: 16px;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .stSelectbox > div {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown('<div class="title">Big Basket Product Recommendation System</div>', unsafe_allow_html=True)

# Dropdown for product selection
product_list = df2['product'].values
selected_product = st.selectbox(
    "Type or select a product from the dropdown:",
    product_list,
    index=0
)

# Predict button and display recommendations
if st.button('Recommend'):
    st.subheader(f"Recommended Products for: **{selected_product}**")
    recommended_products = get_recommendations(selected_product, df2)

    # Check if recommendations are available
    if recommended_products.empty:
        st.error("No recommendations found. Please try another product.")
    else:
        st.dataframe(recommended_products.reset_index(drop=True))
