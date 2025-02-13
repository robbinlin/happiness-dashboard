pip install streamlit pandas plotly seaborn statsmodels
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("happiness_data.csv")
    return df

df = load_data()

# Sidebar - User Selection
st.sidebar.title("Happiness Data Dashboard")
selected_page = st.sidebar.radio("Choose a visualization:", [
    "Happiness Trends", 
    "Scatter Plot", 
    "Choropleth Map",
    "Regression Model Insights"
])

# Happiness Trends - Line Chart
if selected_page == "Happiness Trends":
    st.title("Happiness Trends Over Time")
    
    country = st.selectbox("Select a country:", df["Country name"].unique())
    country_data = df[df["Country name"] == country]

    fig = px.line(
        country_data, x="year", y="Life Ladder",
        title=f"Happiness Over Time in {country}",
        markers=True
    )
    st.plotly_chart(fig)

# Scatter Plot - Relationship between variables
elif selected_page == "Scatter Plot":
    st.title("Scatter Plot: Happiness vs. Other Variables")
    
    x_var = st.selectbox("Select X-axis variable:", [
        "Log GDP per capita", "Social support", 
        "Healthy life expectancy at birth", 
        "Freedom to make life choices", "Generosity", 
        "Perceptions of corruption"
    ])
    
    fig = px.scatter(
        df, x=x_var, y="Life Ladder", 
        color="Life Ladder", color_continuous_scale="magma",
        title=f"Happiness vs {x_var}", opacity=0.8
    )
    st.plotly_chart(fig)

# Choropleth Map - Global Happiness
elif selected_page == "Choropleth Map":
    st.title("Global Happiness Map")
    
    year = st.slider("Select Year:", int(df["year"].min()), int(df["year"].max()), 2022)
    df_year = df[df["year"] == year]

    fig = px.choropleth(
        df_year, locations="Country name", locationmode="country names",
        color="Life Ladder", hover_name="Country name",
        color_continuous_scale="spectral",
        title=f"Global Happiness in {year}"
    )
    st.plotly_chart(fig)

# Regression Model Insights - Predicted vs. Actual Happiness
elif selected_page == "Regression Model Insights":
    st.title("Regression Model: Predicted vs. Actual Happiness")

    # Prepare regression model data
    regression_data = df.dropna(subset=[
        "Life Ladder", "Log GDP per capita", "Social support", 
        "Healthy life expectancy at birth", "Freedom to make life choices", 
        "Generosity", "Perceptions of corruption"
    ])
    
    # Fit fixed-effects regression model
    formula = "Q('Life Ladder') ~ Q('Log GDP per capita') + Q('Social support') + Q('Healthy life expectancy at birth') + Q('Freedom to make life choices') + Q('Generosity') + Q('Perceptions of corruption') + C(year) + C(Q('Country name'))"
    model = smf.ols(formula=formula, data=regression_data).fit()

    # User selects country
    country = st.selectbox("Select a country:", df["Country name"].unique())
    country_data = regression_data[regression_data["Country name"] == country].copy()
    country_data["Predicted Happiness"] = model.predict(country_data)

    # Plot actual vs. predicted happiness
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(country_data["year"], country_data["Life Ladder"], marker="o", linestyle="-", label="Actual Happiness", color="blue")
    ax.plot(country_data["year"], country_data["Predicted Happiness"], marker="s", linestyle="--", label="Predicted Happiness", color="red")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Happiness Score")
    ax.set_title(f"Actual vs. Predicted Happiness for {country}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    
    st.pyplot(fig)
