import streamlit as st
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import geopandas as gp
import plotly.figure_factory as ff
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from collections import Counter
from stopwordz import clean_text

# Replace 'your_mapbox_token' with your actual Mapbox token
mapbox_token = 'pk.eyJ1Ijoid2ppYW4iLCJhIjoiY2xvOGMxMmF3MGNjMjJrbzYxOGR1bmZ0ciJ9.01QxWHFIfEsLjddh4wCtgA'
px.set_mapbox_access_token(mapbox_token)
# --------------------------------------------------------------------------------------------

# AB_NYC_2019 shows the data of the Airbnb in NYC

# Load the CSV-clean dataset
df = pd.read_csv('AB_NYC_2019_cleaned.csv')
# st.write(df.head())

# Load the pickled model
with open('nyc_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --------------------------------------------------------------------------------------------
# Set up the Streamlit app
st.set_page_config(page_title="NYC", layout="centered")

st.sidebar.title("NYC Airbnb Price Visualizer & Predictor")
pages = ["Introduction", "Visualize", "Predict"]
page = st.sidebar.radio("Choose a page", pages)
# --------------------------------------------------------------------------------------------

## Introduction Page
if page == 'Introduction':
    st.title('Introduction')
    st.write("This app is to visualize and predict the price of Airbnb in NYC.")
    st.write("The dataset is from the Airbnb in NYC in 2019.")
    st.subheader('Dataset')
    st.write(df)

# --------------------------------------------------------------------------------------------
## Visualize Page
elif page == 'Visualize':
    st.title('Visualize')
    st.write('Location of Airbnb in NYC')
    fig = px.scatter_mapbox(df,
                        lat=df.latitude,
                        lon=df.longitude,
                        hover_data=["location", "area"],
                        color="location",
                        mapbox_style="dark",
                        zoom=10,
                        height=700,
                        width=700)
    st.plotly_chart(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Room Type for Each Location')
    
    fig = px.scatter_mapbox(df,
                        lat=df.latitude,
                        lon=df.longitude,
                        hover_data=["location", "area"],
                        color="room_type",
                        zoom=10,
                        mapbox_style="dark",
                        height=700,
                        width=700)
    st.plotly_chart(fig)
    
    # --------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.countplot(x='room_type', data=df, hue='location', ax=axes[0])
    axes[0].set_title('Room Type')
    
    sns.countplot(x='location', data=df, hue='room_type', ax=axes[1])
    axes[1].set_title('Location')
    
    st.pyplot(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Top 10 Area in NYC')
    top_10_area = df['area'].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='area', data=df, order=top_10_area, palette='Set1')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Average Price for Room Type')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='room_type', y='price', data= df, palette='Set1', errwidth=0)
    st.pyplot(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Average price of the room type of the Neighbourhood Group')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='room_type', y='price', data=df, hue='location' , errwidth=0)
    st.pyplot(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Top 10 Expensive Neighbourhood in NYC and Top 10 Cheap Neighbourhood in NYC')
        
    top_10_expensive_area = df.groupby('area')['price'].mean().sort_values(ascending= False).head(10)
    top_10_cheapest_area = df.groupby('area')['price'].mean().sort_values().head(10)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(x=top_10_expensive_area.index, y=top_10_expensive_area.values, ax=axes[0],palette='Set1')
    axes[0].set_title('Top 10 Expensive Neighbourhood in NYC')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)  # Rotate x-axis labels of the first subplot

    sns.barplot(x=top_10_cheapest_area.index, y=top_10_cheapest_area.values, ax=axes[1],palette='Set1')  
    axes[1].set_title('Top 10 Cheap Neighbourhood in NYC')  
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)  # Rotate x-axis labels of the second subplot

    st.pyplot(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Map the price range with color')
    fig = px.scatter_mapbox(df, 
                            lat="latitude", 
                            lon="longitude", 
                            color="price", 
                            size="price", 
                            color_continuous_scale=px.colors.sequential.Plasma,
                            size_max=50, 
                            zoom=10, 
                            height=700, 
                            width=700
                            )
    st.plotly_chart(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Price Category')
    fig = px.scatter_mapbox(df,
                        lat=df.latitude,
                        lon=df.longitude,
                        hover_data=["location", "area"],
                        color="price_category",
                        mapbox_style="dark",
                        zoom=10,
                        height=700,
                        width=700)
    st.plotly_chart(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Price Category Visualization')
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.scatterplot(x='latitude', y='longitude', hue='price_category', data=df, size=1)
    plt.title('Price Categories')

    plt.subplot(1, 2, 2)
    df['price_category'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Price Categories')

    plt.subplot(2, 2, 3)
    sns.countplot(x='location', data=df, hue='price_category')
    plt.title('Price Categories Group by Location')

    st.pyplot(plt.gcf())
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Availabitity 365 Days')
    fig = px.scatter_mapbox(df,
                        lat=df.latitude,
                        lon=df.longitude,
                        hover_data=["location", "area"],
                        color="availability_category",
                        mapbox_style="dark",
                        zoom=10,
                        height=700,
                        width=700)
    st.plotly_chart(fig)
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Availability 365 Days Visualization')
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='latitude', y='longitude', hue='availability_category', data=df, size=1)
    plt.title('Availability 365')

    plt.subplot(2, 2, 2)
    df['availability_category'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Availability 365')

    st.pyplot(plt.gcf())
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Top 10 Host in NYC')
    
    top_10_host = df.groupby('host_name')['price'].sum().sort_values(ascending= False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_host.index, y=top_10_host.values, palette='Set1')
    
    plt.xticks(rotation=45)
    plt.title('Top 10 Host in NYC')
    st.pyplot(plt.gcf())
    plt.close()

    # --------------------------------------------------------------------------------------------
    st.write('Top 10 Host and their review')
    top_10_host_review = df[df['host_name'].isin(top_10_host.index)]['reviews'].value_counts()
    st.write(top_10_host_review.head(10))
    plt.close()
    
    # --------------------------------------------------------------------------------------------
    st.write('Get Top 10 Popular Area in NYC')
    top_10_full_booked_room = df[df['availability_365'] == 0]['area'].value_counts().head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_full_booked_room.index, y=top_10_full_booked_room.values, palette='Set1')
    plt.title('Top 10 Popular Area in NYC')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.close()
    # --------------------------------------------------------------------------------------------
    
    st.write('Word Cloud of the Description')
    wc = WordCloud(background_color='white').generate(' '.join(df['reviews']))
    plt.imshow(wc)
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()
    # --------------------------------------------------------------------------------------------
    st.write('Top 50 Words in the Description')
    stop_words = set(stopwords.words('english'))
    cleaned_text = df['reviews'].apply(clean_text)
    most_common_words = Counter(" ".join(cleaned_text).split()).most_common(50)
    common_word_dict = dict(most_common_words)
    wc = WordCloud(background_color='white').generate_from_frequencies(common_word_dict)
    
    plt.imshow(wc)
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()
    
    # --------------------------------------------------------------------------------------------
## Predict Page
elif page == 'Predict':
    st.title('Predict')
    st.write('Make Prediction')
    
    # location = 1 : Manhattan, 2 : Brooklyn, 3 : Queens, 4 : Staten Island, 5 : Bronx, 6 : Long Island, 7 : New Jersey, 8 : Connecticut, 9 : Upstate New York
    # room_type = 1 : Entire home/apt, 2 : Private room, 3 : Shared room
    # minimum_nights = No encode
    # availability_365 = No encode
    # price_category = 1 : 1-50$, 2 : 51-100$, 3 : 101-200$, 4 : 201-300$, 5 : >=300$
    # availability_category = 1 : Not Available, 2 : <= 50 days, 3 : 51-100 days, 4 : 101-200 days, 5 : 201-300 days, 6 : > 300 days
    
    location_mapping = {'Manhattan': 1, 'Brooklyn': 2, 'Queens': 3, 'Staten Island': 4, 'Bronx': 5, 'Long Island': 6, 'New Jersey': 7, 'Connecticut': 8, 'Upstate New York': 9}
    room_type_mapping = {'Entire home/apt': 1, 'Private room': 2, 'Shared room': 3}
    price_category_mapping = {'1-50$': 1, '51-100$': 2, '101-200$': 3, '201-300$': 4, '>=300$': 5}
    availability_category_mapping = {'Not Available': 1, '<= 50 days': 2, '51-100 days': 3, '101-200 days': 4, '201-300 days': 5, '> 300 days': 6}

    location = st.selectbox('Location', list(location_mapping.keys()))
    room_type = st.selectbox('Room Type', list(room_type_mapping.keys()))
    minimum_nights = st.number_input('Minimum Nights', min_value=0, max_value=365, value=1)
    availability_365 = st.number_input('Availability 365', min_value=0, max_value=365, value=1)
    # price_category = st.selectbox('Price Category', list(price_category_mapping.keys()))
    # availability_category = st.selectbox('Availability Category', list(availability_category_mapping.keys()))

    if st.button('Predict'):
        input_data = [[location_mapping[location], room_type_mapping[room_type], minimum_nights, availability_365]]
        input_data = np.array(input_data).astype(float)
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: {prediction[0]}$")
