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

## DISCLAIMER!

This code was runned from 01-07-2021 to 31-07-2021. If you try it again in another timeframe we may have different results.

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

```python


# importing libraries for the API connection

import tweepy
from tweepy.auth import OAuthHandler
import pandas as pd
import numpy as np

# generating a dict with the keys for the API connection

secrets_dict={}
secrets_file = open('tweepy-keys.txt')
for line in secrets_file:
  (key,value) = line.split(':')
  secrets_dict[key] = value[:-1]
  
 
 # creating the API cursor

auth = tweepy.OAuthHandler(secrets_dict['API Key'], secrets_dict['API secret'])
auth.set_access_token(secrets_dict['Access token'], secrets_dict['Access secret'])
api = tweepy.API(auth)

```

## 2. Twitter cursor scrapping

The Twitter API gives developers access to most of Twitter’s functionality. You can use the API to read and write information related to Twitter entities such as tweets, users, and trends.

In this section we are scrapping a lot of content that we are in the end not using:

```python

def scrape(words, date_since, numtweet):

# Creating DataFrame using pandas
  db = pd.DataFrame(columns=['username', 'description', 'location', 'following',
              'followers', 'totaltweets', 'retweetcount', 'text', 'hashtags'])

# We are using .Cursor() to search through twitter for the required tweets.
# The number of tweets can be restricted using .items(number of tweets)
  tweets = tweepy.Cursor(api.search, q=words, lang="en",
            since=date_since, tweet_mode='extended').items(numtweet)

# .Cursor() returns an iterable object. Each item in
# the iterator has various attributes that you can access to
# get information about each tweet
  list_tweets = [tweet for tweet in tweets]

# Counter to maintain Tweet Count
  i = 1

# we will iterate over each tweet in the list for extracting information about each tweet
  for tweet in list_tweets:
    username = tweet.user.screen_name
    description = tweet.user.description
    location = tweet.user.location
    following = tweet.user.friends_count
    followers = tweet.user.followers_count
    totaltweets = tweet.user.statuses_count
    retweetcount = tweet.retweet_count
    hashtags = tweet.entities['hashtags']
  
# Retweets can be distinguished by a retweeted_status attribute,
# in case it is an invalid reference, except block will be executed
    try:
      text = tweet.retweeted_status.full_text
    except AttributeError:
      text = tweet.full_text
    hashtext = list()
    for j in range(0, len(hashtags)):
      hashtext.append(hashtags[j]['text'])

    # Here we are appending all the extracted information in the DataFrame
    ith_tweet = [username, description, location, following,
          followers, totaltweets, retweetcount, text, hashtext]
    db.loc[len(db)] = ith_tweet

# we will save our database as a CSV file.
  return db

```

## 3. NLTK Text Cleaning, tokenization, lemmatization and stop-words removal

The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania. NLTK includes graphical demonstrations and sample data. It is accompanied by a book that explains the underlying concepts behind the language processing tasks supported by the toolkit, plus a cookbook. [Wiki](https://en.wikipedia.org/wiki/Natural_Language_Toolkit).

Tokenization is the process by which a large quantity of text is divided into smaller parts called tokens. These tokens are very useful for finding patterns and are considered as a base step for stemming and lemmatization. Tokenization also helps to substitute sensitive data elements with non-sensitive data elements. [guru99](https://www.guru99.com/tokenize-words-sentences-nltk.html).

Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. [Wiki](https://en.wikipedia.org/wiki/Lemmatisation).

The words which are generally filtered out before processing a natural language are called stop words. These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. Examples of a few stop words in English are “the”, “a”, “an”, “so”, “what”. [towardscience](https://towardsdatascience.com/text-pre-processing-stop-words-removal-using-different-libraries-f20bac19929a)

#### This first step of text cleaning:
```python

# the main cleaning step

def clean_up(s):
    element1 = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', s).lower() # remove links
    element2 = re.sub('[^a-zA-Z0-9]', ' ', element1) # remove non character symbols
    element3 = re.sub('amp', '', element2) # twitter has &amp as a special character
    element4 = re.sub('joe biden', 'joebiden', element3)
    element5 = re.sub('bernie sanders', 'berniesanders', element4)
    element6 = re.sub('ted cruz', 'tedcruz', element5)
    element7 = re.sub('mike pence', 'mikepence', element6)
    element8 = re.sub('joebiden|berniesanders|tedcruz|mikepence','', element7)
    return (re.sub('\d+',' ',element8)) # remove any digits and lowercase everything

```

#### Tokenization:
```python

# tokenize the text

def tokenize(s):
    return word_tokenize(s)
    
# categorize function to help the next function that is lemmatize

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper() # gets first letter of POS categorization
    tag_dict = {"J": wordnet.ADJ, 
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) # get returns second argument if first key does not exist

```

#### Lemmatization:

```python

def lemmatize(l):
  
    lem = WordNetLemmatizer()
    lemmatized = [lem.lemmatize(w,get_wordnet_pos(w)) for w in l]
    
    return lemmatized

```

#### Stop-words removal:

```python

def remove_stopwords(l):
    
    filtered_sentence = []
    
    for w in l:
        if len(w) > 1:
            if w not in stopwords.words('english'):
                filtered_sentence.append(w)
    
    return filtered_sentence
    

# remove elements with 1 len

def remove_1len(row):

    lista = row

    for s in lista:
        if len(s) < 2:
            lista.remove(s)
            
    return lista
 
```

## 4. NLTK Sentiment Analyzer

Sentiment analysis can help you determine the ratio of positive to negative engagements about a specific topic. You can analyze bodies of text, such as comments, tweets, and product reviews, to obtain insights from your audience [realpython](https://realpython.com/python-nltk-sentiment-analysis/).

```python

# now lets start working on our features
# the first feature that we are interested in is the sentiment
# is the tweet negative, neutral or positive?


from nltk.sentiment import SentimentIntensityAnalyzer # this is the package that does the trick
sia = SentimentIntensityAnalyzer()

def is_positive(tweet):
    if sia.polarity_scores(tweet)["compound"] > 0: # when the compound score is greater than 0 the tweet is positive
        return 1 # 1 for positive
    return 0 # 0 for negative or neutral

features['sentiment'] = features['text_preprocessed'].apply(is_positive)
 
```

I also looked at the intensity of the Tweets using the same tool.

```python

# other thing that we want to capture is the actual intensity of the tweet
# some words are more intense than others
# the package is the same, but now we will use the actual compound score

def intensity(tweet):
    return (abs(sia.polarity_scores(tweet)["compound"])+1)**2 # to give more importance the intense tweets we are going to square it

features['intensity'] = features['text_preprocessed'].apply(intensity)
 
```

## 5. MatplotLib Data Visualization

For visualizing the data I used Matplotlib. Furthermore this readme will have other plots.

#### Word cloud
![download](https://user-images.githubusercontent.com/83870535/129708248-14183feb-3904-4c45-b991-f4c6eaa6cb29.png)

## 6. Gensim semantic meaning embedding

Another thing that we need was to capture the semantic meaning of each tweet, transforming it into a vector shape:

```python


# another feature that is really important is the context of the tweet
# to capture that we agre going to use the package gensim with the doc2vec model

from tqdm import tqdm 
tqdm.pandas(desc="progress-bar") 
from gensim.models.doc2vec import LabeledSentence
import gensim

# create a function to label the tweets

def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output

labeled_tweets = add_label(features['text_processed']) # label all the tweets


# initialize the model

model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model
                                  dm_mean=1, # dm_mean = 1 for using mean of the context word vectors
                                  vector_size=50, # no. of desired features
                                  window=5, # width of the context window                                  
                                  negative=7, # if > 0 then negative sampling will be used
                                  min_count=2, # Ignores all words with total frequency lower than 2.                                  
                                  workers=32, # no. of cores                                  
                                  alpha=0.1, # learning rate                                  
                                  seed = 23, # for reproducibility
                                 ) 

# build the vocab with the labeled tweets

model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])

model_d2v.train(labeled_tweets, total_examples= len(features['text_preprocessed']), epochs=5)
 
```

Our table now looked like this:

![image](https://user-images.githubusercontent.com/83870535/129709023-2b69b43e-d2de-42c0-a80b-7b704b994b5e.png)

## 7. Sci-kit Learn for features modeling

I standardized the data because the sentiment and intensity had different scales as the semantic vectors:

```python

# we want to standardize the features for improving the model

from sklearn.preprocessing import StandardScaler

X = target_features_df.drop(columns=['index','target'])

# create object

scaler = StandardScaler()

# fit

scaler.fit(X)

# transform 

X_scaled = scaler.transform(X)
 
```

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

```python

# then we going to use a lda model to extract the topic

theme_0 = list(tweet_info[tweet_info['labels'] == 0]['text_processed'])

theme_0

dictionary = gensim.corpora.Dictionary(theme_0)
bow_corpus = [dictionary.doc2bow(doc) for doc in theme_0]

lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 3, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 3)
                                   
# and this are the topics

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
    
```

```
Topic: 0 
Words: 0.012*"covid" + 0.010*"biden" + 0.010*"trump" + 0.007*"confirm" + 0.007*"hall" + 0.007*"like" + 0.006*"nothing" + 0.006*"news" + 0.006*"town" + 0.006*"diplomat"


Topic: 1 
Words: 0.019*"trump" + 0.012*"get" + 0.012*"covid" + 0.009*"biden" + 0.007*"america" + 0.007*"make" + 0.006*"democrat" + 0.006*"news" + 0.005*"donaldtrump" + 0.005*"vote"


Topic: 2 
Words: 0.013*"want" + 0.008*"president" + 0.008*"work" + 0.008*"go" + 0.007*"texas" + 0.007*"biden" + 0.007*"american" + 0.006*"get" + 0.006*"would" + 0.006*"gun"
```



