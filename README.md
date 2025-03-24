# 🚙 Car Prices Prediction 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)   [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io)

## Introduction

Car price prediction plays a vital role in the automotive market, empowering buyers, sellers, and businesses to make well-informed decisions. In this project, data was gathered from Carvago.com, resulting in a rich and comprehensive dataset encompassing various car features and prices. To ensure precision and reliability, the collected data underwent thorough cleaning and preprocessing, laying the foundation for accurate predictions. This effort highlights the value of data-driven approaches in addressing real-world challenges within the automotive industry.

<img src = "https://github.com/suhasmaddali/Images/blob/main/Car%20Prices%20Prediction%20GitHub%20Image%202.jpg" width = 350 height = 200/><img src = "https://github.com/suhasmaddali/Images/blob/main/Car%20Prices%20Prediction%20GitHub%20Image%203.jpg" width = 350 height = 200/>


## Machine Learning and Deep Learning

* __Machine Learning__ and __deep learning__ have gained rapid traction in the recent decade. 
* It would be really helpful if we can predict the prices of a car based on a few sets of features such as __horsepower__, __make__ and __other features__. 
* Imagine if a company wants to set the price of a car based on some of the features such as make, horsepower, and mileage. 
* It could do so with the help of machine learning models that would help it to determine the price of a car. 
* This would ensure that the company sets the right amount so that they get the most profits while setting such a price. 
* Therefore, the machine learning models that we would be working with would ensure that the right price is set for new cars which would save a lot of money for car manufacturers respectively.
* We would be working with the car prices prediction data and looking for the predictions of different kinds of cars. 
* We would be first visualizing the data and understanding some of the information that is very important for predictions. 
* We would be using different regression techniques to get the average price of the car under consideration.

<h2> Data Source</h2>

The data used in this project was collected from the website Carvago.com, comprising approximately 5000 samples with 18 features.
This dataset provides valuable insights into the cars we encounter in our daily lives, helping us understand how they are sold and their average prices.


__Source:__ [Carvago.com/car](https://carvago.com/cars)

## Metrics

Predicting car prices is a __continuous machine learning problem__. Therefore, the following metrics, commonly used for regression problems, were taken into account. Below are the __metrics__ used in the process of predicting car prices:  

* [__Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)  
* [__Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)  
* [__R-squared (R²)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)  

These metrics provide insights into the accuracy and reliability of the predictions made by the model.

## Exploratory Data Analysis (EDA)

**Dashboard**
<img src = "https://github.com/buidinhquan13/CarPricePrediction/blob/main/dashboard.png"/>

<--
### Model Performance

We will now focus our attention on the performance of __various models__ on the test data. Scatterplots can help us determine how much of a spread our predictions are from the actual values. Let us go over the performance of many ML models used in our problem of car price prediction. 

[__Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - After looking at the linear regression plot, it looks like the model is performing quite well. Scatterplots between the predictions and the actual test outputs closely resemble each other. If there are low latency requirements for a deployment setup, linear regression could be used. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/LR%20Plot.png"/>

[__Support Vector Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - Support vector regression (SVR) can be computational. In addition, the results below indicate that the predictions are far off from the actual car prices. Therefore, alternate models can be explored. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/SVR%20Plot.png"/>

[__K Nearest Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - K-Nearest Regressor is doing a good job in predicting the car prices as highlighted in the plot below. There is less spread between the test output labels and the predictions generated by the model. Therefore, there are higher chances that the model gives a low mean absolute error and mean squared error. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/K%20Neighbors%20Regressor.png"/>

[__PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) - This model does a good job overall when it comes to predicting car prices. However, it fails to compare trends and patterns for higher-priced cars well. This is evident due to the fact that there is a lot of spread among higher car price values as shown in the plot. K-Nearest Regressor, on the other hand, also does predictions accurately on higher priced cars. 
 
<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/PLS%20Regressor%20plot.png"/>

[__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - Based on all the models tested, the decision tree regressor was performing the best. As shown below, there is a lot of overlap between the predictions and the actual test values. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Decision%20Tree%20Plot.png"/>

[__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) - The performance of gradient boosted decision regressor is plotted and it shows that it is quite similar to the decision tree. At prices that are extremely high, the model fails to capture the trend in the data. It does a good job overall. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/GBDT%20Plot.png"/>

[__MLP Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) - It does a good job when it comes to predicting car prices. However, there are better models earlier that we can choose as their performance was better than MLP Regressor in this scenario. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/MLP%20Regressor%20plot.png"/>

[__Final Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) - After performing feature engineering and hyperparameter tuning the models, the best model that gave the least mean absolute error (lower is better) was Decision Tree Regressor. Other models such as Support Vector Regressors took a long time to train along with giving less optimum results. Along with good performance, Decision Tree Regressors are highly interpretable and they give a good understanding of how a model gave predictions and which feature was the most important for it to decide car prices. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Final%20MAE.png"/>

[__Final Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - The performance of the Decision Tree Regressor was also the highest when using mean squared error as the output metric. While the Gradient Boosted Regressor came close to the performance of a Decision Tree Regressor, the latter is highly interpretable and easier to deploy in real time. Therefore, we can choose this model for deployment as it is performing consistently across a large variety of metrics. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Final%20MSE.png"/>
-->
## Machine Learning Models 

We have to be using various machine learning models to see which model reduces the __R2__, __mean absolute error (MAE)__ and __mean squared error (MSE)__ on the cross-validation data respectively. Below are the various machine learning models used. 

| __Machine Learning Models__|__R2__| __Mean Absolute Error__| __Mean Squared Error__|
| :-:| :-:| :-:|:-:|
| [__Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)|0.889  |22170.316| 956741143.227|
| [__XGBoost__](https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html)|__0.943__	|__13707.594__|	__492416589.018__|
|	[__Histogram Gradient Boosting__](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)|	0.942|13731.157 |	505920218.740|
|	[__Random Forest__](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html)|	0.939| 13252.834|	528170078.124|
|	[__Decision Tree__](https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeRegressor.html)|	0.898| 16991.575|	881261960.347|

## Outcomes

* The best-performing model in terms of __mean absolute error (MAE)__ and __mean squared error (MSE)__ was __XGBoost Regressor__, which outperformed other models.
* __Scatterplots__ between the __actual prices__ and __predicted prices__ showed an almost __linear__ relationship, particularly for the __XGBoost Regressor__ model.  

## 👉 Run project

### 1. Clone this repo:
<pre>
<code class="copyable">git clone https://github.com/buidinhquan13/CarPricePrediction.git</code>
</pre>

### 2. Install requirements:
<pre>
<code class="copyable">pip install requirements.txt</code>
</pre>

### 3. Run the app:
<pre>
<code class="copyable">streamlit run app.py</code>
</pre>



