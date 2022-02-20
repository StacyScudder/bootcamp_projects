# World Happiness

![Map of World Happiness](https://github.com/StacyScudder/AWT_bootcamp/blob/main/G_project_1/world_happiness18.png)
![Map of World Happiness](https://github.com/StacyScudder/AWT_bootcamp/blob/main/G_project_1/world_happiness19.png)


This project looked at the world happiness report for 2018 and 2019 and was to help us learn more about data cleaning, EDA, and visualizations. The instructor deleted some of the data in the Healthy_Life_Expectancy (HLE) column.

When importing the data, there were two columns at the beginning of the file that weren't needed (Unnamed: 0 and Unnamed: 0.1), so those were dropped right away. Then, I started getting a feel for the dataset - looking at the info, basic statistics, duplicates, and missing values. There weren't many problems with the data except all the missing values in the HLE column. I couldn't fill the missing values with the mean because that would make the column useless. Dropping the column wouldn't be the best idea since it might be correlated with happiness. Instead, I looked around to see if I could find more data to fill in these values. Of course, the world happiness report is a popular dataset, so it wasn't difficult to find more data to use.

After getting all the data organized and adding the HLE, I wanted to add the ability to look at the world in regions, as well as countries. To add the regions, python has a library called pycountry_convert that will add the regions for each country. THere were some problems because a few of the countries weren't included or were spelled diferently(Trinidad & Tobago, for example). Those countries had to be added manually. 

I divided the dataset into 2018 and 2019 datasets to make it easier to graph those years individually. I could have used groupby, but I hadn't learned that yet. I created various visualizations using the combined dataset and the smaller datasets split by years. I also found the mean happiness score for each continent for both 2018 and 2019. 

Examining the data led to the conclusion that the overall rank, and therefore the happiness score, is correlated with GDP per capita, higher life expectancy, and social supports in that order.
