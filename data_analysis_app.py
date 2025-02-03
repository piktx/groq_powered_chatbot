# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import specific modules for language models
from langchain_groq.chat_models import ChatGroq
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe

# Set up Streamlit page
st.set_page_config(page_title="Data Analysis Platform")
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        color: #333333;
    }
    .stButton button {
        background-color: #0072C6;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define functions to load language models
def load_groq_llm():
    return ChatGroq(model_name="mixtral-8x7b-32768", api_key=os.getenv('GROQ_API_KEY'))

def load_openai_llm():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Sidebar for user inputs
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
llm_choice = st.sidebar.selectbox("Select Language Model", ("Groq", "OpenAI"))

# Main application content starts here
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # General Information
    st.subheader("General Information")
    st.write(f"Shape of the dataset: {data.shape}")
    st.write(f"Data Types:\n{data.dtypes}")
    st.write(f"Memory Usage: {data.memory_usage(deep=True).sum()} bytes")

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        st.write(data[categorical_columns].describe())

    # Missing Values Analysis
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    st.write(missing_values)
    # Optional: Visualize missing values here

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    corr_matrix = data.corr().stack().reset_index(name="correlation")
    high_corr = corr_matrix[corr_matrix['correlation'].abs() > 0.5]
    high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
    st.write(high_corr)
    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(10, 7))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Load LLMs
    groq_llm = load_groq_llm()
    openai_llm = load_openai_llm()

    # SmartDataframe setup for language model interaction
    df_groq = SmartDataframe(data, config={'llm': groq_llm})
    df_openai = SmartDataframe(data, config={'llm': openai_llm})

    # User query input for natural language analysis
    query = st.text_input("Enter your query about the data:")
    if query:
        try:
            response = ""
            if llm_choice == "Groq":
                response = df_groq.chat(query)
            elif llm_choice == "OpenAI":
                response = df_openai.chat(query)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Interactive visualization section
    with st.expander("View Interactive Plot"):
        plot_type = st.selectbox("Select Plot Type", ["Scatter", "Line", "Bar"])
        x_col = st.selectbox("Select X-axis Column", data.columns)
        y_col = st.selectbox("Select Y-axis Column", data.columns)

        if plot_type == "Scatter":
            fig = px.scatter(data, x=x_col, y=y_col)
        elif plot_type == "Line":
            fig = px.line(data, x=x_col, y=y_col)
        else:
            fig = px.bar(data, x=x_col, y=y_col)

        st.plotly_chart(fig)
