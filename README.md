# Amazon Book Review Sentiment Prediction

### INTRODUCTION 

Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP) that identifies the emotional tone behind a body of text. This is a popular way for organizations to determine and categorize opinions about a product, service, or idea. It involves the use of data mining, machine learning (ML) and artificial intelligence (AI) to mine text for sentiment and subjective information.

Sentiment analysis systems help organizations gather insights from unorganized and unstructured text that comes from online sources such as emails, blog posts, support tickets, web chats, social media channels, forums and comments. Algorithms replace manual data processing by implementing rule-based, automatic or hybrid methods. Rule-based systems perform sentiment analysis based on predefined, lexicon-based rules while automatic systems learn from data with machine learning techniques. A hybrid sentiment analysis combines both approaches.

In addition to identifying sentiment, opinion mining can extract the polarity (or the amount of positivity and negativity), subject and opinion holder within the text. Furthermore, sentiment analysis can be applied to varying scopes such as document, paragraph, sentence and sub-sentence levels.

Vendors that offer sentiment analysis platforms or SaaS products include Brandwatch, Hootsuite, Lexalytics, NetBase, Sprout Social, Sysomos and Zoho. Businesses that use these tools can review customer feedback more regularly and proactively respond to changes of opinion within the market.

### Features
• unique_id
 
• asin

• product_name

• product_type

• helpful

• rating

• title

• date 

• reviewer 

• reviewer_location 

• review_text

## Exploratory Data Analysis

1)First we get acquainted with Data, we found that the there are 42,000 reviews to be analysed. 

2)Out of the given number of columns (11), only one column have the data, which contains the review_text, which needs to be analysed and based upon the analysis, need to be segmented into positive, negative or neutral sentiment.

3)First step in the EDA part is to convert the review text in Lower casing. The idea is to convert the input text into same casing format so that 'text', 'Text' and 'TEXT' are treated the same way.

4)Next step would be the removal of punctuation. This is again a text standardization process that will help to treat 'hurray' and 'hurray!' in the same way.

5)Next step is the removal of stopwords. Stopwords are commonly occuring words in a language like 'the', 'a' and so on. They can be removed from the text most of the times, as they don't provide valuable information for downstream analysis.

6)Next step is the removal of frequent words and removal of rare words. We might also have some frequent words which are of not so much importance to us.

7)Next step is stemming. Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form. For example, if there are two words in the corpus walks and walking, then stemming will stem the suffix to make them walk.

8)Next step is lemmatization. Lemmatization is similar to stemming in reducing inflected words to their word stem but differs in the way that it makes sure the root word (also called as lemma) belongs to the language.

9)Since there are no emojis and emoticons in the data, next step is the removal of urls.  For example, if we are doing a twitter analysis, then there is a good chance that the tweet will have some URL in it. Probably we might need to remove them for our further analysis.

10)Next step is the removal of html tags. This is especially useful, if we scrap the data from different websites. We might end up having html strings as part of our text.

11)Next step is the spelling correction.  Typos are common in text data and we might want to correct those spelling mistakes before we do our analysis.

##Calculating Polarity 

We have installed textblob and using it, we have got the polarity value and subjectivity value of the modified version of the review_text column (which is final_text in this case) and stored the subjectivity as a new column i.e. subjectivity and stored the polarity as a new column i.e. Analysis_labels. 

##SUPPORT VECTOR MACHINE 

From sklearn, we have imported the svm and from sklearn.metrics, we have imported the classification_report. We are using TfidVectorizer, whih we have imported from sklearn.feature_extraction.text

##Spliting the dataset

Firstly we scaled our entire Dataset using MinMax Scaling in the range 0,1.

As we are working on Timeseries Data, we cant split our data randomly.a sequential manner must be followed.
So first 80% data is taken as Training data and rest as testing data. 

## Creating Input Arrays for Model     

our main aim is to predict the polarity of the Amazon book text review.

So ,it is clear that we are dealing only with the final_text column.
For grouping the reviews into positive or negative sentiment, we need final_text (this will be added to X_train ), the polarity value and the sentiment will be predicted (this column Analysis_labels will be added to Y_train). 
Similar grouping is done for test dataset.

After this reshaping of train and test data set is done to convert it into 3d array.

## Modelling
### SVM

Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.

The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. 

Summary of model is given below :


## Accuracy
After predicting the output for test dataset.
we calculated R-score which resulted in 85.33%

## Predicting Output for Analysis_label

For the text review we have in hand, we can calculate the sentiment whether it is positive, negative or neutral with 85.33% accuracy.


In such a way we can get the sentiment of the input text review.

## Deployment

We have used Flask framework to create a webapp .
The input taken from the user will be :
* the text review 

We have run and tested the webapp on the command prompt. 
Last step is to deploy it on the heroku platform, which we will complete shortly 



## Conclusion and FutureWork

Our Model has significantly performed well on train as well as test data set.Also after analysis of the review and sentiment prediction ,we have crosschecked the sentiment value as per the data which came out to be nearly same.
So concluding that Model is properly trained and tested.
