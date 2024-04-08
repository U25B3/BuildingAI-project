import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

def load_data(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)
    return data

def pre_process_data(data):
    X = []
    y = []
    i = 0
    while i < len(data):
        entry = data[i]
        if entry["to_year"] is not None:
            duration = entry["to_year"] - entry["from_year"]
            if duration > 15:   # Remove outliers
                data.remove(entry)
                continue
        else:
            data.remove(entry)
            continue
        try:
            votes = float(entry["votes"][:-1])
            rating = float(entry["rating"])
            age_limit = float(entry["age_limit"])
            runtime = float(entry["runtime"])
            start_year = float(entry["from_year"])
        except ValueError:
            data.remove(entry)
            continue  # Skip entries with invalid values
        X.append([votes, rating, age_limit, runtime, start_year])
        y.append(duration)
        i += 1
    
    print("Assessed data inputs: " + str(len(X)))
    X = np.array(X)
    y = np.array(y)

    # Weight based on assumption
    weight_votes = 1
    weight_rating = 0.9
    weight_runtime = 0.7
    weight_age_limit = 0.6
    weight_start_year = 0.4

    # Scale the features
    scaler = StandardScaler()
    X[:, 0] *= weight_votes
    X[:, 1] *= weight_rating
    X[:, 2] *= weight_age_limit  
    X[:, 3] *= weight_runtime  
    X[:, 4] *= weight_start_year
    X = scaler.fit_transform(X)

    return [X, y, data]

def train_test_split_data(X, y, titles):
    X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split(X, y, titles, test_size=0.3, random_state=17)
    return [X_train, X_test, y_train, y_test, titles_train, titles_test]

def train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    return [y_test, y_pred, mse]

def plot_data(y_test, y_pred, titles):
    for i in range(len(y_test)):
        plt.scatter(y_test[i], y_pred[i])
        plt.annotate(titles[i], (y_test[i], y_pred[i]))
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal')
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title('Actual vs. Predicted Duration')
    plt.legend()
    plt.show()

def get_std(y_test, y_pred):
    residuals = y_test - y_pred   
    std = np.std(residuals)
    return std

def get_titles (data):
    titles = [entry["title"] for entry in data]
    return titles


if __name__ == "__main__":
    data = load_data("series_data.json")
    [X,y, processed_data] = pre_process_data(data)
    titles = get_titles(processed_data)
    [X_train, X_test, y_train, y_test, titles_train, titles_test] = train_test_split_data(X,y,titles)
    [y_test, y_pred, mse] = train_model(X_train, y_train,X_test,y_test)
    std = get_std(y_test, y_pred)
    print("Standard Diviation: " + str(std))
    plot_data(y_test, y_pred, titles_test)