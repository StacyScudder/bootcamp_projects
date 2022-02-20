# Housing Prices
This project was adapted from the Kaggle competition, Housing Prices Advanced Regression Techniques, [found here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)


The dataset for this project contains the train and test set from the competition. There were 80 columns in the original set, but not all correclated with the sale price for the house. I first combined all the columns that contained square footage into one column named TotalSF and deleted the individual SF columns. Then all the columns that had close to 50% null or missing values were dropped, which brought the total number of columns down to 72. All the columns that ended with QC were identical to the columns that ended with Cond, so all the Cond columns got dropped. Without a closer examination, I couldn't drop any more columns. 

After dropping columns, I filled any empty cells with NaN for categorical columns and 0 for numerical columns. Then I wanted to use some graphs to narrow down the columns that correlated most strongly with the price. 

<p float="left">
  <img src="https://github.com/StacyScudder/AWT_bootcamp/blob/main/G_project_2/SF-price.png" width="400" />
  <img src="https://github.com/StacyScudder/AWT_bootcamp/blob/main/G_project_2/Qual-price.png" width="400" /> 
</p>

We encoded our categorical data columns and then used a random forest model to find the 10 most important (closely correlated) to the SalePrice. The unimportant columns were dropped from the dataframe, as well as the SalePrice column since it's the target. The SalePrice distribution is right-skewed, so using a log transformation will make it closer to Normal and easier to use in our model. The last step before creating our model was to drop any outliers from the columns in our clean dataframe.

<img width="480" height="480" src="https://github.com/StacyScudder/AWT_bootcamp/blob/main/G_project_2/corr_matrix.png">
As we can see from the correlation coefficients, the TotalSF and OverallQuality have the strongest correlation. After those, most of the correlations can be considered subgroups of TotalSf and OverallQuality. 

To fit our model, we first need to split the dataset into a training and a test set (70-30 split). We then fit our training data to a second degree polynomial. Looking at the scatterplot of the predicted and actual values, we could visualize how closely our model fit the data. Looking at our metrics, we can see that our model is pretty good with about 88% accuracy. 
