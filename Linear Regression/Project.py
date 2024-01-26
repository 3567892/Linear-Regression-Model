import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy,math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('USA_Housing.csv')

df.head()

df.describe

df.isnull().any()

#Plotting area vs price
plt.scatter(x= 'Area Population', y ='Price' , data = df)

#formats my y-axis to disable the scientific notation. 
plt.ticklabel_format(style ='plain',axis = 'both')

#adding commas
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: format(int(x), ',')))


#plots titles and labels
plt.title ('Area Population of House vs. Price of the House')
plt.ylabel('Price')
plt.xlabel('Area Population')
plt.show()

#Average number of prices vs average number of bedrooms
plt.bar(x= df['Avg. Area Number of Bedrooms'], height=df['Price'])
#formats my y-axis to disable the scientific notation. 
plt.ticklabel_format(style ='plain',axis = 'both')

#adding commas
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: format(int(x), ',')))

#plots titles and labels
plt.title ('Area Population of House vs. Price of the House')
plt.ylabel('Price')
plt.xlabel('Avg. Area Number of Bedrooms')
plt.show()

# List of columns to plot against 'Price'
columns_to_plot = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms']

# Number of subplots
num_subplots = len(columns_to_plot)

# Create subplots
fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(8, 6*num_subplots))



# Plot each column against 'Price'
for i, column in enumerate(columns_to_plot):
    axes[i].scatter(x=df[column], y=df['Price'])
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Price')
    axes[i].set_title(f'{column} vs. Price')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#Top 10 Addresses with Highest Average Price
top_addresses = df.groupby('Address')['Price'].mean().nlargest(10).reset_index()

plt.barh(y=top_addresses['Address'], width=top_addresses['Price'])
plt.title('Top 10 Addresses with Highest Average Price')
plt.ylabel('Average Price')
plt.xlabel('Address')
plt.show()

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train.shape: " + str(X_train.shape))
print("X_test.shape: " + str(X_test.shape))
print("y_train.shape: " + str(y_train.shape))
print("y_test.shape: " + str(y_test.shape))

#converting panda dataframes into numoy arrays
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

# Reshape Y_train and Y_test
y_train_np = y_train_np.reshape(-1, 1)
y_test_np = y_test_np.reshape(-1, 1)


#X_train_np = X_train_np.T
#X_test_np = X_test_np.T

X_train_np = X_train_np.astype(np.float64)
y_train_np = y_train_np.astype(np.float64)

#function that makes the prediction
def predict(X, w, b):
    
    
    f = np.dot(X, w) + b
    return f
X_sample = np.random.rand(4000, 5)  # 5 examples with 5 features each
w_sample = np.random.rand(5, 1)  # Weights with shape (5, 1)
b_sample = 3.0  # Bias

prediction = predict(X_sample, w_sample, b_sample)

print("Input data (X):")
print(X_sample)
print("\nWeights (w):")
print(w_sample)
print("\nBias (b):", b_sample)
print("\nPredicted values:")
print(prediction)

def compute_cost(X, y, w, b):
    m, n = X.shape
    cost = 0.0

    # Print shapes before the loop for debugging
    print(f"Initial shapes - X shape: {X.shape}, w shape: {w.shape}, y shape: {y.shape}")

    for i in range(m):
        X_i = X[i].reshape(1, -1)
        f_wb_i = np.dot(X_i, w) + b
        cost += (f_wb_i - y[i, 0]) ** 2  # Access the element at index 0 in y[i]

        # Print shapes for debugging
        print(f"Iteration {i + 1} - X_i shape: {X_i.shape}, w shape: {w.shape}, y shape: {y.shape}")

    cost = cost / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        # Ensure y[i] is treated as a 1D array
        X_i = X[i].reshape(1, -1)
        error = (np.dot(X_i, w) + b) - y[i, 0]
        

        # Update dj_dw using vectorized operation
        dj_dw += np.dot(error, X_i).flatten()
        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent (X, y, w, b, cost_function, gradient_function, learning_rate, number_iterations,print_cost = False):
    
    costs = []
    w = copy.deepcopy(w)
   
    
    for i in range(number_iterations):
        
        dj_dw,dj_db = gradient_function(X, y, w, b)
        
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        if i %  100 ==0 :
            costs.append(cost_function(X, w, y, b))
            
            
        if print_cost and i % 100 == 0:
            print("Cost after interation %i : %f " %(i,cost_function(X, w, y, b)))
                  
    return w, b , costs

#initializing my parameters

initial_w =  np.zeros((X_train_np.shape[1], 1))
initial_b = 0

#hyperparameters
learning_rate = 0.01
iterations = 1000


X_train = X_train_np
y_train = y_train_np
 

# Train the model
w_final, b_final, cost_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                           compute_cost, compute_gradient, learning_rate, iterations, print_cost=True)