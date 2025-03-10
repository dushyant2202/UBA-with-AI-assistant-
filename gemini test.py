import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
import streamlit as st
from streamlit_chat import message
from sklearn.preprocessing import LabelEncoder
import random

# Load the dataset
try:
    df = pd.read_csv(r"Z:/10march25/dataset_2000.csv")  # Corrected file path
except FileNotFoundError:
    st.error("Error: dataset_2000.csv not found. Make sure the file is in the specified directory.")
    st.stop()
except OSError as e:
    st.error(f"Error: Could not read dataset_2000.csv.  Please check the file path and permissions.\n{e}")
    st.stop()


# Data Cleaning and Preprocessing
df.columns = df.columns.str.replace(r' \(mins\)', '', regex=True)  # Clean column names
df.rename(columns={'time spent': 'time_spent'}, inplace=True)
df.dropna(inplace=True)  # Drop rows with missing values
df.drop_duplicates(inplace=True)  # Drop duplicate rows

# Convert signup_date to datetime objects
df['signup_date'] = pd.to_datetime(df['signup_date'], format='%m/%d/%Y')

# Encode categorical features
categorical_cols = ['gender', 'device_type', 'location', 'interested_content', 'not_interested_content', 'activity_type', 'account_status', 'interest_2', 'interest_3', 'recommended_type']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# --- Streamlit App ---
st.title("User Engagement Data Analysis")
st.markdown("A comprehensive visualization and analysis of user engagement data, with an AI chatbot to answer your questions.")

# --- Visualizations ---

# 1. Gender Distribution (Pie Chart)
st.header("Gender Distribution")
gender_counts = df['gender'].value_counts()
fig_gender = px.pie(gender_counts, names=gender_counts.index, title='Distribution of Genders')
st.plotly_chart(fig_gender, use_container_width=True)

# 2. Device Type Distribution (Bar Chart)
st.header("Device Type Distribution")
device_counts = df['device_type'].value_counts()
fig_device = px.bar(device_counts, x=device_counts.index, y=device_counts.values, labels={'x': 'Device Type', 'y': 'Count'}, title='Distribution of Device Types')
st.plotly_chart(fig_device, use_container_width=True)

# 3. Age Distribution (Histogram)
st.header("Age Distribution")
fig_age = px.histogram(df, x='age', nbins=30, title='Distribution of User Ages')
st.plotly_chart(fig_age, use_container_width=True)

# 4. Time Spent vs. Age (Scatter Plot)
st.header("Time Spent vs. Age")
fig_time_age = px.scatter(df, x='age', y='time_spent', title='Time Spent (mins) vs. Age')
st.plotly_chart(fig_time_age, use_container_width=True)

# 5. Account Status Distribution (Bar Chart)
st.header("Account Status Distribution")
status_counts = df['account_status'].value_counts()
fig_status = px.bar(status_counts, x=status_counts.index, y=status_counts.values, labels={'x': 'Account Status', 'y': 'Count'}, title='Distribution of Account Statuses')
st.plotly_chart(fig_status, use_container_width=True)

# 6. Clicks vs. Time Spent (Scatter Plot with Trendline)
st.header("Clicks vs. Time Spent")
fig_clicks_time = px.scatter(df, x='time_spent', y='clicks', trendline="ols", title='Clicks vs. Time Spent (mins)')
st.plotly_chart(fig_clicks_time, use_container_width=True)

# 7. 3D Scatter Plot (Age, Time Spent, Clicks)
st.header("3D Scatter Plot of Age, Time Spent, and Clicks")
fig_3d = px.scatter_3d(df, x='age', y='time_spent', z='clicks',
                      color='gender',  # Color by gender for better visualization
                      title='3D Scatter Plot: Age, Time Spent, Clicks')
st.plotly_chart(fig_3d, use_container_width=True)

# 8. Animated Scatter Plot (Time Spent vs. Clicks over Signup Date)
st.header("Animated Time Spent vs. Clicks over Signup Date")
df['signup_month'] = df['signup_date'].dt.to_period('M').astype(str)
fig_animated = px.scatter(df, x="time_spent", y="clicks", animation_frame="signup_month",
                         animation_group="user_id", color="gender", hover_name="username",
                         title="Animated Time Spent vs. Clicks by Signup Month")
st.plotly_chart(fig_animated, use_container_width=True)

# --- Chatbot ---
st.sidebar.header("AI Chatbot")
st.sidebar.info("Ask questions about the dataset and visualizations.")

# Initialize Streamlit Chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def generate_response(prompt):
    # Placeholder for LLM integration - Replace with actual LLM code
    # This is a simplified example; you'd use an actual LLM API like OpenAI's
    if "gender distribution" in prompt.lower():
        return "The gender distribution is visualized using a pie chart."
    elif "device type distribution" in prompt.lower():
        return "The device type distribution is shown using a bar chart."
    elif "age distribution" in prompt.lower():
        return "The age distribution is displayed as a histogram."
    elif "time spent vs age" in prompt.lower():
        return "The relationship between time spent and age is visualized with a scatter plot."
    elif "account status" in prompt.lower():
        return "Account status distribution is represented using a bar chart."
    elif "clicks vs time" in prompt.lower():
        return "The relationship between clicks and time spent is shown on a scatter plot with a trendline."
    elif "3d scatter plot" in prompt.lower():
        return "3D scatter plot shows the relationship between age, time spent, and clicks. The color represents gender."
    elif "animated" in prompt.lower():
        return "Animated graph shows the change of time spent vs clicks over signup months."
    elif "average age" in prompt.lower():
        avg_age = df['age'].mean()
        return f"The average age of users in the dataset is approximately {avg_age:.2f}."
    elif "most popular device" in prompt.lower():
        most_popular = df['device_type'].value_counts().idxmax()
        return f"The most popular device type is {most_popular}"
    elif "active accounts" in prompt.lower():
        active_count = df['account_status'].value_counts()[1]
        return f"There are {active_count} active accounts"
    else:
        return "I'm sorry, I don't have information about that specific query. Please ask about the gender, age, device type, time spent, account status or specific graph visualization."

def get_text():
    input_text = st.text_input("You: ","Ask questions about the dataset and visualizations.", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')