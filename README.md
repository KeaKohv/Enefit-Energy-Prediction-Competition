# Enefit-Energy-Prediction-Competition

Joosep Hook, Kea Kohv

UT Transformers course 2023/2024

This project investigates the application of timeseries transformers to forecasting the energy behavior of prosumers. Prosumers are individuals who both consume and produce energy, in this
case solar energy. The project uses the dataset of a Kaggle competition organized by Enefit, an
energy company in the Baltics. The hourly consumption and production predictions are made
for specific segments in 48h increments. The evaluation is based on the Mean Absolute Error. The features include the forecasted weather data, gas and electricity prices, and prosumerspecific information. The project compares the Vanilla Transformer as well as modified timeseries transformers - the Informer, the Autoformer, the Patch Time-Series Transformer, and
the Temporal Fusion Transformer, against nontransformers. In experiments with NeuralForecast models, the Temporal Fusion Transformer achieves the best performance. We find that
models that support past and future covariates have an advantage over the models that do not
and that optimizing transformers’ hyperparameters, including the dropout rate, learning rate,
and the scaler type, can help improve the accuracy of forecasts but may also risk overfitting
the models.

The folder 'notebooks' contains the experiments with Neuralforecast and Darts models. The Neuralforecast notebooks were run on Google Colab with a 40GB RAM GPU A100. Using a GPU with less RAM on these notebooks may result in cuda out of memory. The Darts notebooks were run on Kaggle with GPU P100. Submissions to the competition are only allowed via offline Kaggle notebooks. The Darts and Neuralforecast packages have to imported via a utility script.

The project raport contains the background, related work, methods, results and discussion of the project.

The dataset and Kaggle competition: https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers 