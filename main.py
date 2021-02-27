#importing the required modules
import numpy as np
import pandas as pd #to read the csv files
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer #vectorizes text as an array
#from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.utils import shuffle #Shuffles the data
import string #we use punctuations
import nltk
from nltk.corpus import stopwords
from nltk import tokenize

#Reading from the data set:-
fake = pd.read_csv(r"C:\Users\Pranav\Desktop\SpaceJam\Testing\Fake\Fake.csv")
true = pd.read_csv(r"C:\Users\Pranav\Desktop\SpaceJam\Testing\True\True.csv")
#Flagging the data as real or fake
fake['target'] = 'fake'
true['target'] = 'true'
#Concatenate the real and fake news and shuffling it
data = pd.concat([fake, true]).reset_index(drop = True)
data = shuffle(data)
data = data.reset_index(drop=True)

#Now we remove the parts we don't require
data.drop(["date"],axis=1,inplace=True) #date
data.drop(["title"],axis=1,inplace=True) #title
data['text'] = data['text'].apply(lambda x: x.lower())

#Removing punctuations and stop words
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
data['text'] = data['text'].apply(punctuation_removal)

#nltk.download('stopwords')
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

#Plotting graph of no. of articles per subject
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()
#Plotting a graph on the no. of fake and real articles
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()

from wordcloud import WordCloud
fake_data = data[data["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,max_font_size = 110,collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

from wordcloud import WordCloud
real_data = data[data["target"] == "true"]
all_words = " ".join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500, max_font_size = 110,collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Frequent words occuring
token_space = tokenize.WhitespaceTokenizer()
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()), "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue') #Seaborn is used
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()

counter(data[data["target"] == "fake"], "text", 20)
counter(data[data["target"] == "true"], "text", 20)

