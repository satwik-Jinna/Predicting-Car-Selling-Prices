#!/usr/bin/env python
# coding: utf-8

# # Problem
# Analyzing car details data for price predictions during selling based on various features from the dataset Car dekho platfrom which is avaliable in Kaggle. link:- https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv
# 
# # Significance
# Understanding factors affecting car prices aids buyers and sellers in making informed decisions. Predictive models provide insights into pricing trends, helping both parties in negotiations.
# 
# 
# # Methodology 
# - Involves data preparation, feature analysis, model building, hyperparameter tuning, and model evaluation. 
# 
# - The aim appears to be predicting the selling price of cars based on various features provided in the dataset. 

# In[4]:


import sqlite3
import pandas as pd
import os
 
# Function to create a connection to the SQLite database
def create_connection(db_file, delete_db=False):
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)
 
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except sqlite3.Error as e:
        print(f"Error connecting to the database: {e}")
 
    return conn
 
# Function to create a table in the SQLite database
def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
 
# Function to insert data into the car details table
def insert_data(conn, insert_data_sql, data):
    try:
        c = conn.cursor()
        c.execute(insert_data_sql, data)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")
 
# Function to select and display rows from the car details table
def select_and_display_rows(conn, select_query, limit=10):
    try:
        c = conn.cursor()
        result = c.execute(select_query).fetchmany(limit)
 
        if result:
            for row in result:
                print(row)
    except sqlite3.Error as e:
        print(f"Error selecting data: {e}")
 
# Database name
db_name = 'car_details.db'
 
# Create a connection to the SQLite database
conn = create_connection(db_name, delete_db=True)
 
# Define the car details table creation SQL statement
create_car_details_table_sql = """
CREATE TABLE IF NOT EXISTS car_details (
    name TEXT,
    year INTEGER,
    selling_price INTEGER,
    km_driven INTEGER,
    fuel TEXT,
    seller_type TEXT,
    transmission TEXT,
    owner TEXT,
    mileage TEXT,
    engine TEXT,
    max_power TEXT,
    torque TEXT,
    seats INTEGER
);
"""
 
# Create the car details table
create_table(conn, create_car_details_table_sql)
 
# Read data from the CSV file
csv_file_path = r'/Users/satwikcrj/Downloads/Car details v3.csv'
df = pd.read_csv(csv_file_path)
 
# Insert data into the car details table
insert_data_sql = """
INSERT INTO car_details (
    name, year, selling_price, km_driven, fuel, seller_type,
    transmission, owner, mileage, engine, max_power, torque, seats
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""
 
for index, row in df.iterrows():
    values = (
        row['name'], row['year'], row['selling_price'], row['km_driven'],
        row['fuel'], row['seller_type'], row['transmission'], row['owner'],
        row['mileage'], row['engine'], row['max_power'], row['torque'], row['seats']
    )
    insert_data(conn, insert_data_sql, values)
 
# Select and display the first 10 rows from the car details table
select_10_rows_query = "SELECT * FROM car_details LIMIT 15"
select_and_display_rows(conn, select_10_rows_query, limit=15)
 
# Close the connection
conn.close()


# # Database management
# Creating SQLite tables (`car_info` and `car_features`) to normalize and store data efficiently.
# 
# - Two tables: `car_info` and `car_features` created using SQLite's `CREATE TABLE` statements.
# - `car_info` contains primary key `Car_Name`, and `car_features` references `Car_Name` as a foreign key.

# In[5]:


# Database name
db_name = 'car_data.db'
def execute_sql_query(conn, sql_query):
    try:
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"Error executing SQL query: {e}")
        return None
 
# Create a connection to the SQLite database
conn = create_connection(db_name, delete_db=True)
 
# Define the car_info table creation SQL statement (if not already created)
create_car_info_table_sql = """
CREATE TABLE IF NOT EXISTS car_info (
    Car_Name TEXT PRIMARY KEY,
    Year INTEGER,
    Selling_Price INTEGER,
    Km_Driven INTEGER,
    Fuel TEXT,
    Seller_Type TEXT,
    Transmission TEXT,
    Owner TEXT
);
"""
 
# Create the car_info table (if not already created)
create_table(conn, create_car_info_table_sql)
 
# Define the car_features table creation SQL statement (if not already created)
create_car_features_table_sql = """
CREATE TABLE IF NOT EXISTS car_features (
    Car_Name TEXT PRIMARY KEY,
    Mileage TEXT,
    Engine TEXT,
    Max_Power TEXT,
    Torque TEXT,
    Seats INTEGER
);
"""
 
# Create the car_features table (if not already created)
create_table(conn, create_car_features_table_sql)
 
# Read data from the CSV file
csv_file_path = r'/Users/satwikcrj/Downloads/Car details v3.csv'
df = pd.read_csv(csv_file_path)
 
# Insert data into the car_info and car_features tables
insert_car_info_sql = """
INSERT INTO car_info (
    Car_Name, Year, Selling_Price, Km_Driven, Fuel, Seller_Type, Transmission, Owner
) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""
 
insert_car_features_sql = """
INSERT INTO car_features (
    Car_Name, Mileage, Engine, Max_Power, Torque, Seats
) VALUES (?, ?, ?, ?, ?, ?);
"""
 
for index, row in df.iterrows():
    car_name = row['name']
 
    # Check if the car_name already exists in the car_info table
    duplicate_check_query = f"SELECT Car_Name FROM car_info WHERE Car_Name = '{car_name}';"
    existing_rows = execute_sql_query(conn, duplicate_check_query)
 
    # If the car_name is not a duplicate, proceed with insertion
    if not existing_rows:
        values_info = (
            car_name, row['year'], row['selling_price'], row['km_driven'],
            row['fuel'], row['seller_type'], row['transmission'], row['owner']
        )
 
        values_features = (
            car_name, row['mileage'], row['engine'], row['max_power'], row['torque'], row['seats']
        )
 
        try:
            c = conn.cursor()
            c.execute(insert_car_info_sql, values_info)
            c.execute(insert_car_features_sql, values_features)
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
 
# Select and display the first 10 rows from the car_info table
select_10_rows_query = "SELECT * FROM car_info LIMIT 10"
result = execute_sql_query(conn, select_10_rows_query)
 
# Display the result
if result:
    for row in result:
        print(row)
 
# Close the connection
conn.close()


# # SQL Join and DataFrame Reconstruction
# 
# - SQL query for joining `car_info` and `car_features`
# - Utilizing SQL join queries to reconstruct combined data and converting it into a Pandas DataFrame.

# In[6]:


# Database name
db_name = 'car_data.db'
 
# Create a connection to the SQLite database
conn = sqlite3.connect(db_name)
conn.execute("PRAGMA foreign_keys = 1")
 
# SQL query to join car_info and car_features tables using only numeric columns
join_query = """
SELECT ci.Car_Name, ci.Year, ci.Selling_Price, ci.Km_Driven,
       cf.Mileage, cf.Engine, cf.Max_Power, cf.Torque, cf.Seats
FROM car_info ci
JOIN car_features cf ON ci.Car_Name = cf.Car_Name;
"""
 
# Execute the join query
result = execute_sql_query(conn, join_query)
 
# Convert the result to a Pandas DataFrame
columns = ['Car_Name', 'Year', 'Selling_Price', 'Km_Driven',
           'Mileage', 'Engine', 'Max_Power', 'Torque', 'Seats']
 
df = pd.DataFrame(result, columns=columns)
 
# Display the DataFrame
print(df)


# # Machine Learning path
# - RandomForestRegressor applied for predicting selling prices based on car features.
# - Features preprocessed for numeric values and split into training and testing datasets.
# - It reduces overfitting in decision trees and helps to improve the accuracy.
# - It is flexible to both classification and regression problems.
# - It works well with both categorical and continuous values.
# - It automates missing values present in the data.

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
 
# Assuming df is the DataFrame from the previous code
# Drop 'Car_Name' as it might not be a useful feature
df = df.drop('Car_Name', axis=1)
 
# Preprocess non-numeric columns
non_numeric_columns = ['Mileage', 'Engine', 'Max_Power', 'Torque']
for column in non_numeric_columns:
    df[column] = pd.to_numeric(df[column].str.extract('(\d+\.\d+|\d+)')[0], errors='coerce')
 
# Drop rows with missing values
df = df.dropna()
 
# Split the data into features and target
features = df.drop('Selling_Price', axis=1)
target = df['Selling_Price']
 
# Split the data into training and testing sets
train_df, test_df, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
 
# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_df, train_target)
 
# Make predictions on the test set
predictions = model.predict(test_df)
 
# Evaluate the model
mse = mean_squared_error(test_target, predictions)
print(f"Mean Squared Error: {mse}")
 
# Feature importances
feature_importances = pd.Series(model.feature_importances_, index=train_df.columns)
print("Feature Importances:")
print(feature_importances)


# # Conclusion
# - If in future we get similer model car we should predict the price in future from this prediction
