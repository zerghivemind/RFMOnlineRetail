import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

import plydata.cat_tools as cat
import plotnine as pn

from xgboost import XGBClassifier , XGBRegressor
from sklearn.model_selection import GridSearchCV , cross_val_score


os.chdir('C:\\Users\\domet\\Documents\\ecommerce')

raw_df = pd.read_excel('online_retail_II.xlsx')
names = ['Customer ID', 'Quantity', 'Price', 'InvoiceDate']

#Cohort Analysis

first_purchase_tbl = raw_df.groupby('Customer ID')[['InvoiceDate', 'Price']].min().reset_index()
# print(first_purchase_tbl['InvoiceDate'].min())
# print(first_purchase_tbl['InvoiceDate'].max())

#plotting the distribution of first purchase dates based on months
first_purchase_tbl['InvoiceMonth'] = first_purchase_tbl['InvoiceDate'].dt.to_period('M')

monthly_counts = first_purchase_tbl['InvoiceMonth'].value_counts().sort_index()

# plt.figure(figsize=(10, 6))
# monthly_counts.plot(kind='line')
# plt.xlabel('Month')
# plt.ylabel('Number of First Purchases')
# plt.title('Distribution of First Purchase Dates')
# plt.xticks(rotation=45)
# plt.show()

#Visualizing Individual Customers

unique_customer_ids = raw_df['Customer ID'].unique()
first_10_customers = unique_customer_ids[:10]

filtered_df = raw_df[raw_df['Customer ID'].isin(first_10_customers)].groupby(['Customer ID', 'InvoiceDate']).sum().reset_index()

#fig, axs = plt.subplots(len(first_10_customers), 1, figsize=(8, 6*len(first_10_customers)))

# for i, customer_id in enumerate(first_10_customers):
#     customer_data = filtered_df[filtered_df['Customer ID'] == customer_id]
#     axs[i].plot(customer_data['InvoiceDate'], customer_data['Price'])
#     axs[i].set_xlabel('Invoice Date')
#     axs[i].set_ylabel('Price')
#     axs[i].set_title(f'Customer ID: {customer_id}')

# plt.tight_layout()
# plt.show()

#Machine learning

n_days = 90
max_date = raw_df['InvoiceDate'].max()
cutoff = max_date - pd.Timedelta(n_days, unit='d')

temporal_in_df = raw_df[raw_df['InvoiceDate'] <= cutoff]
temporal_out_df = raw_df[raw_df['InvoiceDate'] > cutoff]

target_df = temporal_out_df.groupby('Customer ID')['Price'].sum().reset_index().rename(columns={'Price': 'spend_90_total'}).assign(spend_90_flag=1)

# Make Recent Purchase Features
max_date = temporal_in_df['InvoiceDate'].max()
recency_features_df = temporal_in_df.groupby('Customer ID')['InvoiceDate'].max().apply(lambda x: (x - max_date) / pd.to_timedelta(1, "day")).reset_index().rename(columns={'InvoiceDate': 'Recency'})

# Make Frequency Features
frequency_features_df = temporal_in_df.groupby('Customer ID')['InvoiceDate'].count().reset_index().rename(columns={'InvoiceDate': 'Frequency'})

# Make Monetary Features
monetary_features_df = temporal_in_df.groupby('Customer ID')['Price'].sum().reset_index().rename(columns={'Price': 'Monetary'})
monetary_summary_df = monetary_features_df.groupby('Customer ID')['Monetary'].agg(['sum', 'mean']).reset_index()

# Combine Features
features_df = recency_features_df.merge(frequency_features_df, on='Customer ID')
features_df = features_df.merge(monetary_summary_df, on='Customer ID')
features_df = features_df.merge(target_df[['Customer ID', 'spend_90_total']], on='Customer ID')

# ML model
X = features_df[['Recency', 'Frequency', 'sum', 'mean']].fillna(0)

# Next 90 day prediction
y_spend = features_df[['Customer ID', 'spend_90_total']]

xgb_reg_spec = XGBRegressor(
    objective="reg:squarederror", random_state=123)

xgb_reg_model = GridSearchCV(
    estimator=xgb_reg_spec,
    param_grid={
        'learning_rate': [0.01, 0.1, 0.3, 0.5]
    },
    scoring='neg_mean_squared_error',
    refit=True,
    cv=5
)

xgb_reg_model.fit(X, y_spend)
xgb_reg_model.best_score_
xgb_reg_model.best_params_
xgb_reg_model.best_estimator_
prediction_reg = xgb_reg_model.predict(X)


imp_spend_amount_dict = xgb_reg_model.best_estimator_.get_booster().get_score(importance_type='gain')

imp_spend_amount_df = pd.DataFrame.from_dict(imp_spend_amount_dict, orient='index', columns=['importance_score']).reset_index().rename(columns={'index': 'feature'})
# Sort the DataFrame by importance score in descending order
sorted_df = imp_spend_amount_df.sort_values('importance_score', ascending=False)

# # Set up the plot
# plt.figure(figsize=(10, 6))
# plt.bar(sorted_df['feature'], sorted_df['importance_score'])

# # Customize the plot
# plt.xlabel('Feature')
# plt.ylabel('Importance Score')
# plt.title('Feature Importance for Spending Amount')

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=90)

# # Display the plot
# plt.tight_layout()
# plt.show()

prediction_df = pd.concat([
    pd.DataFrame(prediction_reg),
    features_df.reset_index()
])

print(prediction_df)