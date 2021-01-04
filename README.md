# DataScience_Portfolio
This is my portfolio of data science projects. If you find it interesting, Buy me a coffee :)  https://www.buymeacoffee.com/mountaga. I am open to all suggestions.


# 1- Blue Book for Bulldozers:
Predict the auction sale price for a piece of heavy equipment to create a "blue book" for bulldozers.
The goal of the contest is to predict the sale price of a particular piece of heavy equiment at auction based on it's usage, equipment type, and configuaration.  The data is sourced from auction result postings and includes information on usage and equipment configurations.

Fast Iron is creating a "blue book for bull dozers," for customers to value what their heavy equipment fleet is worth at auction.
The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

Sample submission files can be downloaded from the data page. Submission files should be formatted as follows:

Have a header: "SalesID,SalePrice"
Contain two columns
SalesID: SalesID for the validation set in sorted order
SalePrice: Your predicted price of the sale
For this competition, you are predicting the sale price of bulldozers sold at auctions.

The data for this competition is split into three parts:

Train.csv is the training set, which contains data through the end of 2011.
Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
The key fields are in train.csv are:

SalesID: the uniue identifier of the sale
MachineID: the unique identifier of a machine.  A machine can be sold multiple times
saleprice: what the machine sold for at auction (only provided in train.csv)
saledate: the date of the sale
There are several fields towards the end of the file on the different options a machine can have.  The descriptions all start with "machine configuration" in the data dictionary.  Some product types do not have a particular option, so all the records for that option variable will be null for that product type.  Also, some sources do not provide good option and/or hours data.
The machine_appendix.csv file contains the correct year manufactured for a given machine along with the make, model, and product class details. There is one machine id for every machine in all the competition datasets (training, evaluation, etc.).

# 2. Flight Delay Prediction Challenge:
Predict airline delays for Tunisian aviation company, Tunisair

This challenge was designed specifically for the AI Tunisia Hack 2019, which takes place from 20 to 22 September. Welcome to the AI Tunisia Hack participants!

After AI Hack Tunisia, this competition will be re-opened as a Knowledge Challenge to allow others in the Zindi community to learn and test their skills.

Flight delays not only irritate air passengers and disrupt their schedules but also cause :

a decrease in efficiency
an increase in capital costs, reallocation of flight crews and aircraft
an additional crew expenses
As a result, on an aggregate basis, an airline's record of flight delays may have a negative impact on passenger demand.

This competition aims to predict the estimated duration of flight delays per flight

This solution proposes to build a flight delay predictive model using Machine Learning techniques. The accurate prediction of flight delays will help all players in the air travel ecosystem to set up effective action plans to reduce the impact of the delays and avoid loss of time, capital and resources.
The metric used for this challenge is Root Mean Square Error.

Files available for download

Train.csv - the training file
Test.csv - the test file
SampleSubmission.csv - is an example of what your submission file should look like. The order of the rows does not matter, but the names of the IDs must be correct. The column "target" is your prediction.
Variable definitions
DATOP - Date of flight
FLTID - Flight number
DEPSTN - Departure point
ARRSTN - Arrival point
STD - Scheduled Time departure
STA - Scheduled Time arrival
STATUS - Flight status
ETD - Expected Time departure
ETA - Expected Time arrival
ATD - Actual Time of Departure
ATA - Actual Time of arrival
DELAY1 - Delay code 1
DUR1 - delay time 1
DELAY2 - Delay code 2
DUR2 - delay time 2
DELAY3 - Delay code 3
DUR3 - delay time 3
DELAY4 - Delay code 4
DUR4 - delay time 4
AC - Aircraft Code

# 3. Heart Disease Prediction:

Predicting heart disease using machine learning
This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes. If you find that interesting , please vote :).

We're going to take the following approach:

Problem definition
Data
Evaluation
Features
Modelling
Experimentation
-Problem Definition
In a statement,

Given clinical parameters about a patient, can we predict whether or not they have heart
disease?

-Data
The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease

There is also a version of it available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

-Evaluation
If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.
-Features
This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).

Create data dictionary

age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
0: Typical angina: chest pain related decrease blood supply to the heart
1: Atypical angina: chest pain not related to heart
2: Non-anginal pain: typically esophageal spasms (non heart related)
3: Asymptomatic: chest pain not showing signs of disease
trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
chol - serum cholestoral in mg/dl
serum = LDL + HDL + .2 * triglycerides
above 200 is cause for concern
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
'>126' mg/dL signals diabetes
restecg - resting electrocardiographic results
0: Nothing to note
1: ST-T Wave abnormality
can range from mild symptoms to severe problems
signals non-normal heart beat
2: Possible or definite left ventricular hypertrophy
Enlarged heart's main pumping chamber
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
slope - the slope of the peak exercise ST segment
0: Upsloping: better heart rate with excercise (uncommon)
1: Flatsloping: minimal change (typical healthy heart)
2: Downslopins: signs of unhealthy heart
ca - number of major vessels (0-3) colored by flourosopy
colored vessel means the doctor can see the blood passing through
the more blood movement the better (no clots)
thal - thalium stress result
1,3: normal
6: fixed defect: used to be defect but ok now
7: reversable defect: no proper blood movement when excercising
target - have disease or not (1=yes, 0=no) (= the predicted attribute)

# 4. MNIST:

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

Practice Skills
Computer vision fundamentals including simple neural networks

Classification methods such as SVM and K-nearest neighbors

Acknowledgements 
More details about the dataset, including algorithms that have been tried on it and their levels of success, can be found at http://yann.lecun.com/exdb/mnist/index.html. The dataset is made available under a Creative Commons Attribution-Share Alike 3.0 license.
Goal
The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is.
For every in the test set, you should predict the correct label.

Metric
This competition is evaluated on the categorization accuracy of your predictions (the percentage of images you get correct).
The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

# 5. Malaria:
This is a competition page that hosts a repository of segmented cells from the thin blood smear slide images from the Malaria Screener research activity. The smartphone’s built-in camera acquired images of slides for each microscopic field of view. The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research Unit in Bangkok, Thailand. The de-identified images and annotations are archived at NLM

The dataset contains a total of 25,000 cell images with instances of parasitized and uninfected cells.
EVALUATION Metric
The Scoring Metric for this competition is categorization accuracy.

Submission
The submission format is given in the data. There should be two columns in the CSV file titled 'file' and 'class' as given in the sample submission file.
File descriptions - train.zip - the training set of images - test.zip - the test set of images - sample-submission.csv - a sample submission file in the correct format

# 6.Tunisian Fraud Detection Challenge:
Detect tax fraud using the Ministry of Finance of Tunisia's data
This challenge was designed specifically for the AI Tunisia Hack 2019, which takes place from 20 to 22 September. Welcome to the AI Tunisia Hack participants!

After AI Hack Tunisia, this competition will be re-opened as a Knowledge Challenge to allow others in the Zindi community to learn and test their skills.

Tax fraud is the intentional act of lying on a tax return form with the intent to lower one’s tax liability. Under-reporting is one of the most common types of tax frauds. It consists of filing a tax return form with a lesser tax base. As a result of this act, fiscal revenues are reduced, undermining public investment in much-needed services.

The objective of the challenge is to detect tax fraud. This is one of the main priorities of local tax authorities which are required to develop cost-efficient strategies to tackle this problem.

Using historical data, a supervised machine learning technique that detects potential fraudulent taxpayers will increase the operational efficiency of the tax supervision process.

The evaluation metric for this challenge is Root Mean Square Error.

The dataset provided by the Tunisian Ministry of Finance includes variables about tax analysis, taxpayer inspection, and VAT returns. The training dataset provided here is a subset of over 25,000 samples aggregated by year. You are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals the amount of the tax liability that the taxpayer has to adjust.

Files available for download

SUPCOM_Train.csv - this is the file you will use to train your model.
SUPCOM_Test.csv - this is the file you will use to test your model.
SUPCOM_SampleSubmission.csv - is an example of what your submission file should look like. The order of the rows does not matter, but the names of the IDs must be correct. The column "target" is your prediction.
VariableDescription.csv - descriptions of the variables.


# 7.ZindiWeekendz Learning: Urban Air Pollution Challenge by #ZindiWeekendz:
part of Urban Air Pollution Challenge
Can you predict air quality in cities around the world using satellite data?

This challenge was designed specifically as a #ZindiWeekendz hackathon (Urban Air Pollution Challenge). We are re-opening the hackathon as a Knowledge Challenge, to allow the Zindi community to learn and test their skills. To help you all out, we’ve created a new Tutorials tab with helpful resources from the community. We encourage Zindians to share their code on the discussion board so that everyone in our community can learn from and support one another.

You may have seen recent news articles stating that air quality has improved due to COVID-19. This is true for some locations, but as always the truth is a little more complicated. In parts of many African cities, air quality seems to be getting worse as more people stay at home. For this challenge we’ll be digging deeper into the data, finding ways to track air quality and how it is changing, even in places without ground-based sensors. This information will be especially useful in the face of the current crisis, since poor air quality makes a respiratory disease like COVID-19 more dangerous.

We’ve collected weather data and daily observations from the Sentinel 5P satellite tracking various pollutants in the atmosphere. Your goal is to use this information to predict PM2.5 particulate matter concentration (a common measure of air quality that normally requires ground-based sensors to measure) every day for each city. The data covers the last three months, spanning hundreds of cities across the globe.

About #ZindiWeekendz

The Zindi community is joining the fight against COVID-19! #ZindiWeekendz are virtual weekend hackathons hosted by Zindi. This series of #ZindiWeekendz throughout April and May 2020 focuses specifically on COVID-19.

In a time of lockdowns, remote work, and general uncertainty, #ZindiWeekendz offer data scientists the opportunity to continue to develop their skills while contributing to practical, open-source AI solutions to help in the battle against COVID-19.

All winning solutions will be shared as a public good on GitHub. We are committed to supporting partners implement these solutions and encourage anyone who is interested to reach out to us at zindi@zindi.africa.

About World Air Quality Index

The World Air Quality Index project is a non-profit project started in 2007. Its mission is to promote air pollution awareness for citizens and provide a unified and world-wide air quality information. The project is providing transparent air quality information for more than 100 countries, covering more than 12,000 stations in 1000 major cities, via those two websites: aqicn.org and waqi.info

The error metric for this competition is the Root Mean Squared Error

Submissions should follow the sample submission format, with ‘Place_ID X Date’ in one column and predictions for ‘target’ in the other.

The objective of this challenge is to predict PM2.5 particulate matter concentration in the air every day for each city. PM2.5 refers to atmospheric particulate matter that have a diameter of less than 2.5 micrometers and is one of the most harmful air pollutants. PM2.5 is a common measure of air quality that normally requires ground-based sensors to measure. The data covers the last three months, spanning hundreds of cities across the globe.

The data comes from three main sources:

Ground-based air quality sensors. These measure the target variable (PM2.5 particle concentration). In addition to the target column (which is the daily mean concentration) there are also columns for minimum and maximum readings on that day, the variance of the readings and the total number (count) of sensor readings used to compute the target value. This data is only provided for the train set - you must predict the target variable for the test set.
The Global Forecast System (GFS) for weather data. Humidity, temperature and wind speed, which can be used as inputs for your model.
The Sentinel 5P satellite. This satellite monitors various pollutants in the atmosphere. For each pollutant, we queried the offline Level 3 (L3) datasets available in Google Earth Engine (you can read more about the individual products here: https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p). For a given pollutant, for example NO2, we provide all data from the Sentinel 5P dataset for that pollutant. This includes the key measurements like NO2_column_number_density (a measure of NO2 concentration) as well as metadata like the satellite altitude. We recommend that you focus on the key measurements, either the column_number_density or the tropospheric_X_column_number_density (which measures density closer to Earth’s surface).
Unfortunately, this data is not 100% complete. Some locations have no sensor readings for a particular day, and so those rows have been excluded. There are also gaps in the input data, particularly the satellite data for CH4.

Variable Definitions: Read about the datasets at the following pages:

Weather Data: https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25
Sentinel 5P data: https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p - all columns begin with the dataset name (eg L3_NO2 corresponds to https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2) - look at the corresponding dataset on GEE for detailed descriptions of the image bands - band names should match the second half of the column titles.
Files available for download:

Train.csv - contains the target and supporting data for 349 locations. This is the dataset that you will use to train your model.
Test.csv- resembles Train.csv but without the target-related columns, and covers 179 different locations.This is the dataset on which you will apply your model to.
SampleSubmission.csv - shows the submission format for this competition, with the ‘Place_ID X Date’ column mirroring that of Test.csv and the ‘target’ column containing your predictions. The order of the rows does not matter, but the names of the Place_ID X Date must be correct.
Competitions.

# 8.Anomaly detection in 4G cellular networks:
Explore ML solutions for the detection of abnormal behaviour of eNB

-Introduction

The purpose of this homework is to solve a classification problem proposed as a competition in the Kaggle InClass platform, where each team of two members will try to get the maximum score. You can apply any of the concepts and techniques studied in class for exploratory data analysis, feature selection and classification.

-Problem description

Context:
Traditionally, the design of a cellular network focuses on the optimization of energy and resources that guarantees a smooth operation even during peak hours (i.e. periods with higher traffic load). However, this implies that cells are most of the time overprovisioned of radio resources. Next generation cellular networks ask for a dynamic management and configuration in order to adapt to the varying user demands in the most efficient way with regards to energy savings and utilization of frequency resources. If the network operator were capable of anticipating to those variations in the users’ traffic demands, a more efficient management of the scarce (and expensive) network resources would be possible.
Current research in mobile networks looks upon Machine Learning (ML) techniques to help manage those resources. In this case, you will explore the possibilities of ML to detect abnormal behaviors in the utilization of the network that would motivate a change in the configuration of the base station.

Goal:
The objective of the network optimization team is to analyze traces of past activity, which will be used to train an ML system capable of classifying samples of current activity as:
• 0 (normal): current activity corresponds to normal behavior of any working day and. Therefore, no re-configuration or redistribution of resources is needed.
• 1 (unusual): current activity slightly differs from the behavior usually observed for that time of the day (e.g. due to a strike, demonstration, sports event, etc.), which should trigger a reconfiguration of the base station.

Content:
The dataset has been obtained from a real LTE deployment. During two weeks, different metrics were gathered from a set of 10 base stations, each having a different number of cells, every 15 minutes. The dataset is provided in the form of a csv file, where each row corresponds to a sample obtained from one particular cell at a certain time. Each data example contains the following features:

• Time : hour of the day (in the format hh:mm) when the sample was generated.
• CellName1: text string used to uniquely identify the cell that generated the current sample. CellName is in the form xαLTE, where x identifies the base station, and α the cell within that base station (see the example in the right figure).
• PRBUsageUL and PRBUsageDL: level of resource utilization in that cell measured as the portion of Physical Radio Blocks (PRB) that were in use (%) in the previous 15 minutes. Uplink (UL) and downlink (DL) are measured separately.
• meanThrDL and meanThrUL: average carried traffic (in Mbps) during the past 15 minutes. Uplink (UL) and downlink (DL) are measured separately.
• maxThrDL and maxThrUL: maximum carried traffic (in Mbps) measured in the last 15 minutes. Uplink (UL) and downlink (DL) are measured separately.
• meanUEDL and meanUEUL: average number of user equipment (UE) devices that were simultaneously active during the last 15 minutes. Uplink (UL) and downlink (DL) are measured separately.
• maxUEDL and maxUEUL: maximum number of user equipment (UE) devices that were simultaneously active during the last 15 minutes. Uplink (UL) and downlink (DL) are measured separately.
• maxUE_UL+DL: maximum number of user equipment (UE) devices that were active simultaneously in the last 15 minutes, regardless of UL and DL.

• Unusual: labels for supervised learning. A value of 0 determines that the sample corresponds to normal operation, a value of 1 identifies unusual behavior.
