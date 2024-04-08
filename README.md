
# TV Series Duration Prediction
## Building AI course project

This project aims to predict the duration of TV series based on various features such as votes, ratings, age limit,episode runtime and starting year. The duration is calculated as the difference between the end year and start year of the series. The prediction is made using machine learning techniques and neural networks.

## Dataset

The dataset used in this project consists of information about various TV series including their titles, start and end years, number of episodes, ratings, and other relevant features. The data is stored in a JSON format, with each entry representing a TV series. The episodes are selected randomly from IMDB top 250 series.

## Preprocessing

Before training the models, the data undergoes preprocessing to handle missing values, convert features to numeric format, and remove outliers. Features such as votes, ratings, age limit, and runtime are extracted and normalized to ensure consistent scaling.

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

Mean Squared Error: 13.969238234347074

Standard Diviation: 3.8131979046382094

![Series_duration](https://github.com/U25B3/BuildingAI-project/assets/80511413/9b667201-d95c-42f8-b41e-6f1f6e51b8b5)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
