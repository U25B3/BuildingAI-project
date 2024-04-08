
# TV Series Duration Prediction 
Final project for the Building AI course

## Summary
This project aims to predict the duration of TV series based on various features such as votes, ratings, age limit,episode runtime and starting year. The duration is calculated as the difference between the end year and start year of the series. The prediction is made using machine learning techniques and neural networks.

## Background
This project aims to predict the duration of TV series based on various features. By analyzing these factors, the goal is to develop a predictive model that can assist in understanding the key determinants of a series' duration. This endeavor is driven by the interest in uncovering patterns in TV series characteristics and their potential impact on viewer engagement and production decisions.

## Dataset

The dataset used in this project consists of information about various TV series including their titles, start and end years, number of episodes, ratings, and other relevant features. The data is stored in a JSON format, with each entry representing a TV series. The series are selected randomly from IMDB top 250 series, and 26 are evaluated.

### Preprocessing

Before training the models, the data undergoes preprocessing to handle missing values, convert features to numeric format, and remove outliers. Features such as votes, ratings, age limit, and runtime are extracted and normalized to ensure consistent scaling.

### Weighting
The following weighting has been implemented after normalizing thee data. 

```python
    weight_votes = 1
    weight_rating = 0.9
    weight_runtime = 0.7
    weight_age_limit = 0.6
    weight_start_year = 0.4
```

## Model Training

### Neural Network Model

The neural network model architecture comprises several dense layers with the ReLU activation function, followed by an output layer for regression. The model is trained using TensorFlow's Keras API.


#### Model Architecture

The neural network architecture is defined as follows:

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

```
## Results

### Mean Squared Error (MSE)
The Mean Squared Error (MSE) measures the average squared difference between predicted and actual values. Here, the MSE of 13.97 indicates that, on average, the model's predictions deviate by approximately 13.97 from the actual TV series durations. Lower MSE values signify better performance.

### Standard Deviation
The Standard Deviation (SD) quantifies the variability of predicted durations around the mean. With a value of approximately 3.81, the model's predictions exhibit moderate variability. Lower SD implies more consistent predictions.

These results suggest that while the model performs reasonably well, there's room for improvement in reducing prediction variability.

![Series_duration](https://github.com/U25B3/BuildingAI-project/assets/80511413/9b667201-d95c-42f8-b41e-6f1f6e51b8b5)

### Results: Votes and rating only
MSE: 20.32

SD: 4.73

![series_duration_only_rating_votes](https://github.com/U25B3/BuildingAI-project/assets/80511413/523e627e-51f1-4100-8676-b9cc39f1f780)

## Improvements
The initial atempt shows a somewhat positive result, but a larger dataset is needed to improve this model. More series parameters are avalible online and the dataset should be improved to include these features.

## Acknowledgments
The series data used to generate this model is fetched from Imdb. 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
