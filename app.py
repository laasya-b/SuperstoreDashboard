import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your data
df = pd.read_csv("superstore.csv")
df['Order.Date'] = pd.to_datetime(df['Order.Date'])

# Aggregate data
sales_by_category = df.groupby('Category')['Sales'].sum().reset_index()
profit_by_region = df.groupby('Region')['Profit'].sum().reset_index()
monthly_sales = df.groupby(df['Order.Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly_sales['YearMonth'] = monthly_sales['Order.Date'].astype(str)
monthly_sales['Month_Num'] = range(len(monthly_sales))

# Linear regression for next month forecast
X = monthly_sales[['Month_Num']]
y = monthly_sales['Sales']
model = LinearRegression()
model.fit(X, y)
next_month = pd.DataFrame({'Month_Num': [len(monthly_sales)]})
predicted_sales = model.predict(next_month)[0]

# Streamlit dashboard
st.title("Superstore Sales Dashboard")
st.metric("Predicted Next Month Sales", f"${predicted_sales:,.2f}")

# Plots
fig_sales_category = px.bar(sales_by_category, x='Category', y='Sales', color='Category', text='Sales')
st.plotly_chart(fig_sales_category)

fig_profit_region = px.bar(profit_by_region, x='Region', y='Profit', color='Region', text='Profit')
st.plotly_chart(fig_profit_region)

fig_monthly_sales = px.line(monthly_sales, x='YearMonth', y='Sales', markers=True, title="Monthly Sales Trend")
st.plotly_chart(fig_monthly_sales)



