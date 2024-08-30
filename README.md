# What drives the price of a car?

**OVERVIEW**

In this application, we will explore a dataset from kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing.  Your goal is to understand what factors make a car more or less expensive.  As a result of this analysis, we will provide clear recommendations to the client -- a used car dealership -- as to what consumers value in a used car.

### Objective

The task is to develop a predictive model to identify the key factors influencing used car prices. This involves analyzing a dataset containing attributes such as year, manufacturer, model, condition, cylinders, fuel type, transmission, VIN, drive type, vehicle size, and vehicle type. Using techniques such as feature selection and regression analysis, we aim to determine the most significant variables that impact the pricing of used cars. The goal is to transform this business objective into a data-driven solution by leveraging statistical and machine learning methods to build an accurate pricing prediction model.

____________________________________________________________________________________________________________________________________________________________________________________________________________________________

<img width="400" alt="image" src="https://github.com/user-attachments/assets/775178a0-d000-48a3-8ba6-4cd9eb8ee3f7">


 Data contains missing values. Here is percentage breakdown: 

 <img width="400" alt="image" src="https://github.com/user-attachments/assets/ffeb670e-b328-41e8-b4c4-15106fad2ad2">

### Summary of Statistics

#### Count:
Price: 3.887450e+05 (or 388,745)

Year: 388745.000000 (or 388,745)

Odometer: 3.887450e+05 (or 388,745)

This represents the total number of entries (vehicles) in the dataset. All three variables have 388,745 entries.

#### Mean:
Price: 80,291

Year: 2011

Odometer: 96,402.18

The mean is the average value of each variable. For example, the average price is $80,291.37, the average vehicle year is approximately 2011, and the average odometer reading is 96,402.18 miles.

#### Standard Deviation (std):
Price: 12,762,720

Year: 8.8

Odometer: 194,525.7

The standard deviation measures the amount of variation or dispersion in the data. A high standard deviation in price (12,762,720) suggests that vehicle prices vary widely. For year, the standard deviation is about 8.86 years, indicating some spread around the average year (2011). The odometer reading also has a high standard deviation, meaning the mileage varies significantly.

#### Minimum (min):
Price: 0

Year: 1900

Odometer: 0

The minimum value shows the lowest recorded value in each category. Some vehicles have a recorded price of $0, which could indicate missing data or a promotional offer. The earliest vehicle year is 1900, and some vehicles have an odometer reading of 0 miles, which might indicate new or unrecorded mileage.

#### 25th Percentile (25%):
Price: 5,995

Year: 2008

Odometer: 37,725

The 25th percentile indicates that 25% of the vehicles have a price below $5,995, a year below 2008, and an odometer reading below 37,725 miles.

#### Median (50%):
Price: 13,999

Year: 2014

Odometer: 85,989

The median is the middle value when all the data points are arranged in order. Half of the vehicles are priced below $13,999, have a year below 2014, and an odometer reading below 85,989 miles.

#### 75th Percentile (75%):
Price: 26,980

Year: 2017

Odometer: 133,500

#### The 75th percentile shows that 75% of the vehicles have a price below $26,980, a year below 2017, and an odometer reading below 133,500 miles.

Maximum (max):
Price: 3,736,929,000

Year: 2022

Odometer: 10,000,000

The maximum value shows the highest recorded value in each category. There’s an exceptionally high price of $3.73 billion, which may indicate an outlier or data entry error. The newest vehicle year is 2022, and the highest odometer reading is 10,000,000 miles, which is likely an error or a very unusual case.

### Target Distribution Before and After applying the Upper and Lower Bound Filters 
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/33241ea4-2c6f-4f09-ad4f-9a3154340286">

### Odometer Distribution Before and After applying the Upper and Lower Bound Filters 

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/32d7caf0-0459-4065-88d9-392b82db30ee">

### Data Preprocessing Steps: 

#### 1. Drop id, VIN columns
#### 2. Reengineer manufacturer column (too many unique values to encode)
To see if price by region differs I will group the list of manufacturers into bins according to their normal average price and name those groups.We can create several price categories (bins) and assign each manufacturer to a bin based on their typical average price.

#### Price Categories (Bins)
Economy: Budget-friendly brands with generally lower average prices.

Mid-Range: Brands with moderate pricing, offering a balance of affordability and quality.

Luxury: High-end brands known for premium pricing and high-quality features.

Ultra-Luxury: Exclusive brands with very high average prices, often associated with luxury and prestige.

Grouping the Manufacturers
Economy:
chevrolet, ford, jeep, nissan, mazda, honda, dodge, buick, hyundai, kia, subaru, mitsubishi, volkswagen, fiat

Mid-Range:
gmc, toyota, ram, chrysler, volvo, lincoln, acura, mini, pontiac

Luxury:
cadillac, lexus, jaguar, audi, infiniti, mercedes-benz, bmw, rover, tesla, land rover

Ultra-Luxury:¶
aston-martin, ferrari, morgan, porsche

Drop Manufacturer column after binning.
#### 3. Drop model column 
#### 4. Drop size, cylinders condition columns (large percentage of missing values)
#### 5. Filter title_status on 'clean' only and then drop title column 
#### 6. Drop nan rows in fuel, odometer, transmission 
#### 7. Investigate average price distribution by class bin by region to see if there are significant variations
#### 8. Investigate average price distribution by class bin by state to see if there are significant variations
#### 9. Drop 402 regions and 50 states
#### 10.Replace missing values in color column with the random mixture of white, black, and grey (3 most popular car colors in the US)
#### 11.Replace missing values in type column with the random mixture of SUV, sedan, and pickup (3 most popular types in the US)
#### 12.Encode categorical variables

## Modeling
### Linear Regression with Standardization and Log Transformed Target
#### MSE
The metrics for the model show a Mean Squared Error (MSE) of 113,213,816.59 and an R^2 Score of 0.30. The MSE indicates the average squared difference between the predicted and actual values, with a higher value suggesting larger discrepancies in predictions. In this case, the relatively high MSE suggests that the model's predictions are not very close to the actual values.

#### R^2
The R^2 Score, which measures the proportion of the variance in the target variable that is predictable from the features, is 0.30. This means that only 30% of the variance in the target variable is explained by the model, indicating a modest fit. A higher R^2 Score would imply that the model is better at capturing the variability in the data.

Overall, these metrics suggest that while the model captures some of the variance in the data, there is significant room for improvement. The model may benefit from additional feature engineering, data preprocessing, or exploring more sophisticated algorithms to improve its predictive performance.

### Ridge Regression Model with Grid Search and Standardization
#### Train MSE (Mean Squared Error): 0.4593
This value represents the average squared difference between the predicted values and the actual values on the training set. A lower value indicates better performance, but it is important to compare it with the test MSE to assess if the model is overfitting.

#### Test MSE: 0.4577
This value indicates the average squared difference between the predicted values and the actual values on the test set. The fact that the test MSE is very close to the train MSE suggests that the model is generalizing well and not overfitting.

#### Best Ridge Alpha: 100
The best alpha parameter found during grid search is 100. This indicates that the optimal amount of regularization for this Ridge regression model is relatively high. A high alpha value suggests that the model is penalizing the size of the coefficients more heavily to avoid overfitting.

#### Summary
The close values of train and test MSEs suggest that the Ridge regression model has a good balance between bias and variance and is generalizing well to unseen data. The high alpha value (100) indicates strong regularization, which is likely helping to prevent overfitting by shrinking the coefficients and simplifying the model.

## Conclusion and Recommendations
#### Price drivers:

#### Manufacturing year and mileage
The modeling process involved applying linear regression with a log-transformed target variable, 'price,' to address the right-skewed distribution. However, this approach yielded a suboptimal performance with a score as low as 0.29. An alternative modeling approach demonstrated improved metrics and highlighted two primary drivers of car prices: the year of manufacture, where newer years correlated with higher prices, and odometer readings, where higher mileage led to lower prices. Features such as color, type, and transmission did not show a significant impact on used car prices.

#### Data Issues
Despite the original dataset containing 426,880 records, many attributes had incomplete or erroneous data, which compromised the overall data quality. After removing incomplete records and normalizing the remaining data, the dataset was reduced to 267,777 records. Due to time constraints, further investigation into the missing data was not possible. To ensure accurate predictions and effective model development, it is essential to have a robust and high-quality dataset.

#### Recommendations

Acquire more newer cars, as the year of production positively influences the price of used cars. For older vehicles, focus on the mileage, as it significantly negatively impacts the car's price.

Invest in obtaining a cleaner dataset or enhancing the quality of the existing data. Conduct thorough research to understand the relationship between attributes to guide data cleaning efforts.

Address anomalies in the dataset, such as prices ranging from 0 to exceeding 5 million, as these indicate data quality issues.Setting in constrains on values upon data entry could be one of the many solutions to help avoid extreme values.

Explore additional modeling techniques or focus on enhancing feature engineering.
