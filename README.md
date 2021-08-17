# Twitter Public Opinion Analysis

This project was developed as the final project on the Ironhack Data Analytics Bootcamp.

The objective was to build a algorithm capable of analysing the public perception from Twitter tweets. For the analysis, Joe Biden, Bernie Sanders, Ted Cruz and Mike Pence were the politicians used as dummy examples.


## Brief Wiki Descripition

Twitter is an American microblogging and social networking service on which users post and interact with messages known as "tweets". Registered users can post, like, and retweet tweets, but unregistered users can only read them. Users access Twitter through its website interface or its mobile-device application software ("app"), though the service could also be accessed via SMS before April 2020. The service is provided by Twitter, Inc., a corporation based in San Francisco, California, and has more than 25 offices around the world. Tweets were originally restricted to 140 characters, but the limit was doubled to 280 for non-CJK languages in November 2017. Audio and video tweets remain limited to 140 seconds for most accounts.

Public opinion is the collective opinion on a specific topic or voting intention relevant to a society. Democracy requires public opinion because it derives authority from the public.


## How to run the code

1. Either fork or clone the repository into your device.
2. Install the requirements.
3. For this project you need to have a Twitter Developer Account, and have the right keys to make the connections with the API.
4. Run the "maindevelop.ipynb" notebook.
5. This project was done using some dummie examples for the targets, if you want some different analysis you would need to change some parts of the code.

# Data Pipeline:

![final_project_landscape](https://user-images.githubusercontent.com/83870535/127627948-122ee655-def3-436f-8f25-a7f0286cd163.png)

# Snapshot from the final table:

![image](https://user-images.githubusercontent.com/83870535/127628798-aa4b382b-f7f8-4f98-b923-40adb0e0f9b2.png)

## Inside of the 'maindevelop' notebook you can find:

1. Tweepy connection
2. Twitter cursor scrapping 
3. NLTK Text Cleaning, tokenization, lemmatization and stop-words removal
4. NLTK SentimentAnalyzer
5. Matplotlib Data Visualition
6. Gensim semantic meaning embedding
7. Sci-kit Learn for features modeling
8. Gensim for context analysis


## 1. Tweepy Connection

Tweepy a package that provides a very convenient way to use the Twitter API.

![image](https://user-images.githubusercontent.com/83870535/129705828-c37b0a11-e160-4799-bd18-99dddb2c548b.png)

## 2. Twitter cursor scrapping

The Twitter API gives developers access to most of Twitter’s functionality. You can use the API to read and write information related to Twitter entities such as tweets, users, and trends.

In this section we are scrapping a lot of content that we are in the end not using:

![image](https://user-images.githubusercontent.com/83870535/129706309-edd43d92-96ac-46ec-aba1-692ff2e490e4.png)

## 3. NLTK Text Cleaning, tokenization, lemmatization and stop-words removal

The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania. NLTK includes graphical demonstrations and sample data. It is accompanied by a book that explains the underlying concepts behind the language processing tasks supported by the toolkit, plus a cookbook. [Wiki](https://en.wikipedia.org/wiki/Natural_Language_Toolkit).

Tokenization is the process by which a large quantity of text is divided into smaller parts called tokens. These tokens are very useful for finding patterns and are considered as a base step for stemming and lemmatization. Tokenization also helps to substitute sensitive data elements with non-sensitive data elements. [guru99](https://www.guru99.com/tokenize-words-sentences-nltk.html).

Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. [Wiki](https://en.wikipedia.org/wiki/Lemmatisation).

The words which are generally filtered out before processing a natural language are called stop words. These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. Examples of a few stop words in English are “the”, “a”, “an”, “so”, “what”. [towardscience](https://towardsdatascience.com/text-pre-processing-stop-words-removal-using-different-libraries-f20bac19929a)

#### This first step of text cleaning:
![image](https://user-images.githubusercontent.com/83870535/129707345-163938fe-04da-45ea-9f79-d721a7168db6.png)

#### Tokenization:
![image](https://user-images.githubusercontent.com/83870535/129707419-459f07fd-91c9-4c8a-a297-e4cfb9c1d37c.png)

#### Lemmatization:
![image](https://user-images.githubusercontent.com/83870535/129707503-8945addb-2cb1-498e-aa0a-ef28f56d088a.png)

#### Stop-words removal:
![image](https://user-images.githubusercontent.com/83870535/129707571-207354ae-b6fe-4c5a-bace-8bf1e079d6a5.png)

## 4. NLTK Sentiment Analyzer

Sentiment analysis can help you determine the ratio of positive to negative engagements about a specific topic. You can analyze bodies of text, such as comments, tweets, and product reviews, to obtain insights from your audience [realpython](https://realpython.com/python-nltk-sentiment-analysis/).

![image](https://user-images.githubusercontent.com/83870535/129708054-f6966400-9c27-4acd-87e3-16533f05f359.png)

I also looked at the intensity of the Tweets using the same tool.

![image](https://user-images.githubusercontent.com/83870535/129708147-0fa493ad-5a79-45b4-83d9-4a3f95083b54.png)


## 5. MatplotLib Data Visualization

For visualizing the data I used Matplotlib. Furthermore this readme will have other plots.

#### Word cloud
![download](https://user-images.githubusercontent.com/83870535/129708248-14183feb-3904-4c45-b991-f4c6eaa6cb29.png)

## 6. Gensim semantic meaning embedding

Another thing that we need was to capture the semantic meaning of each tweet, transforming it into a vector shape:

![image](https://user-images.githubusercontent.com/83870535/129708965-cd638e22-ea3f-4cdd-9ec9-d97cef298c37.png)

Our table now looked like this:

![image](https://user-images.githubusercontent.com/83870535/129709023-2b69b43e-d2de-42c0-a80b-7b704b994b5e.png)

## 7. Sci-kit Learn for features modeling

I standardized the data because the sentiment and intensity had different scales as the semantic vectors:

![image](https://user-images.githubusercontent.com/83870535/129709478-3c2b1905-f0ee-40cd-9720-1bd9853ed2c0.png)

Now we needed to reduce this features, because the information was really sparsed out into many columns.

For that I used the PCA.

The principal components of a collection of points in a real coordinate space are a sequence of p unit vectors, where the i-th vector is the direction of a line that best fits the data while being orthogonal to the first i-1 vectors. Here, a best-fitting line is defined as one that minimizes the average squared distance from the points to the line. These directions constitute an orthonormal basis in which different individual dimensions of the data are linearly uncorrelated. Principal component analysis (PCA) is the process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest. [wiki](https://en.wikipedia.org/wiki/Principal_component_analysis)

This was the distribution of the 5 features:

![image](https://user-images.githubusercontent.com/83870535/129709594-2b3c48d4-c2a8-432f-9b0f-21ae0794822e.png)

Then, for clustering KMEANS was used:

![image](https://user-images.githubusercontent.com/83870535/129709736-e7cb231e-3fc4-40a4-b450-4598dc62518a.png)

The hyperparametrization, to see the best amount of clusters was done witht he elbow method:

![image](https://user-images.githubusercontent.com/83870535/129709808-0f134f7d-a028-4682-914f-1180db4e1335.png)

The affinity for each topic was important to analyze our targets, so for that I used the centroid distance. Here tweet is a row, and each column is the distance for the centroid:

![image](https://user-images.githubusercontent.com/83870535/129709940-69e1dd84-bc26-4739-9752-d10a42463867.png)

## 8. Gensim for context analysis

After that, we needed to see what where these clusters talking about. As it was a unsupervised clustering we don't know how the algorithm decided to divide the tweets. I used the LDA Multicore tool by Gensim to come up with this topic extraction:

This is just the example of the first cluster, I did that for every single cluster:

![image](https://user-images.githubusercontent.com/83870535/129710602-96f978cb-2d05-48d2-a055-dc4ee6bb5453.png)

![image](https://user-images.githubusercontent.com/83870535/129710642-c6b852bb-4bbf-4bb1-9f99-387abee7c1f3.png)






