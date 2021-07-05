#content based filtering
from moviedock import app, db
from moviedock.models import User, Movie, Reviews, Recommendations
import pandas as pd
import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
bag_of_words = pd.read_csv('bag_of_words.csv')
# instantiating and generating the count matrix
count = CountVectorizer()
#creates sparce matrix commpressed
count_matrix = count.fit_transform(bag_of_words['bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def content_based_recommendations(movie_id):

    # gettin the index of the movie that matches the title
    #idx = indices[indices==movie_title].index.values.astype(int)[0]
    idx = movie_id-1
    # creating a Series with the similarity scores in descending order
    score_series_Desc = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    score_series_Asce = pd.Series(cosine_sim[idx]).sort_values(ascending = True)
    # getting the indexes of the 10 most similar movies
    starting_index = int(score_series_Desc.size/3)
    ending_index = starting_index*2
    top_10_indexes = list(score_series_Desc.iloc[1:20].index)
    middle_10_indexes = list(score_series_Desc.iloc[starting_index:starting_index+20].index)
    last_10_indexes = list(score_series_Asce.iloc[1:20].index)
    top_10_indexes=[x+1 for x in top_10_indexes]
    middle_10_indexes=[x+1 for x in middle_10_indexes]
    last_10_indexes=[x+1 for x in last_10_indexes]

    return top_10_indexes, middle_10_indexes, last_10_indexes
#collaborative filtering

#this code is for text pre cleaning
#important imports for data cleaning
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import unicodedata#import contractions
from contractions import contractions_dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
#functions for data cleaning
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


# Define function to expand contractions
def expand_contractions(text, contraction_mapping=contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                    flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction


    try:
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
    except:
        return text
    return expanded_text


# special_characters removal
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

stopword_list= stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas
#this is the main function which calls all above to perform text cleaning
def normalize_and_lemmaize(input):
    sample = denoise_text(input)
    sample = expand_contractions(sample)
    sample = remove_special_characters(sample)
    words = nltk.word_tokenize(sample)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)
#importing for our model predictions
from keras.models import load_model
model_test = load_model('checkpoint-0.673.h5') #loading sentiment model
MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes = ["negative", "neutral", "positive"]


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_a = np.nan_to_num(norm_a)
    norm_b = np.nan_to_num(norm_b)
    return dot_product / (norm_a * norm_b)

def get_sentiment_persentage(text):
    text = normalize_and_lemmaize(text)
    sequences_test = tokenizer.texts_to_sequences([text])
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    y_prob = model_test.predict(data_test)
    return y_prob

def find_similar_users(user1, movie_row, user1_review):
    movie_reviews = movie_row.reviews
    similarity_measure = {}
    similar_users = []
    user1_sentiment_percentage = [user1_review.negative, user1_review.neutral, user1_review.positive]
    for user2_review in movie_reviews:
        user2_sentiment_percentage = [user2_review.negative, user2_review.neutral, user2_review.positive]
        similarity_measure[user2_review.user_id] = cos_sim(user1_sentiment_percentage, user2_sentiment_percentage)


    sorted_users = sorted(similarity_measure.items(), key=operator.itemgetter(1), reverse=True)

    if(len(sorted_users) < 10 and len(sorted_users) > 1):
        similar_users = sorted_users[1:]
    else:
        similar_users = sorted_users[1:10]

    return similar_users

# one approtch for collaborative filtering
def colaborative_filtering(user1, movie_row, user1_review):
    similar_users = find_similar_users(user1, movie_row, user1_review)
    recommended_movies = []
    for user2 in similar_users:
        reviewed_movies = Reviews.query.filter(Reviews.user_id==user2[0]).filter(Reviews.movie_id != movie_row.id).all()

        for reviewed_movie in reviewed_movies:
            if(reviewed_movie.positive > reviewed_movie.negative and reviewed_movie.positive > reviewed_movie.neutral):
                recommended_movies.append(reviewed_movie.movie_id)
    return recommended_movies
# second approtch for collaborative filtering
#Utility functions
def fill_Matrix(movies, user_reviews):
    user_sentiments = np.zeros((len(movies),3))
    for user_review in user_reviews:
        movie_index = movies.index(user_review.movie_id)
        user_sentiments[movie_index][0] = user_review.negative
        user_sentiments[movie_index][1] = user_review.neutral
        user_sentiments[movie_index][2] = user_review.positive
    return user_sentiments

def colaborative_filtering1(user1, movie_row, user1_review):
    movie_reviews = movie_row.reviews
    all_movies_reviews = []
    users = []
    movies = []
    for movie_review in movie_reviews:
        users.append(movie_review.user_id)
        user_reviews = Reviews.query.filter(Reviews.user_id==movie_review.user_id).all()
        for movie_review1 in user_reviews:
            movies.append(movie_review1.movie_id)
        all_movies_reviews.append(user_reviews)
    movies = list(dict.fromkeys(movies))
    total_movies = len(movies)
    user1_Matrix = fill_Matrix(movies, user1.reviews)
    similarity_measure = {}

    for user2_reviews in all_movies_reviews:
        user2_Matrix = fill_Matrix(movies, user2_reviews)
        similarity_vector = []
        for i in range(total_movies):
            similarity_vector.append(cos_sim(user1_Matrix[i],user2_Matrix[i]))
        similarity_vector = np.nan_to_num(similarity_vector)
        similarity_measure[user2_reviews[0].user_id] = np.linalg.norm(similarity_vector)

    sorted_users = sorted(similarity_measure.items(), key=operator.itemgetter(1), reverse=True)

    if(len(sorted_users) < 10 and len(sorted_users) > 1):
        similar_users = sorted_users[1:]
    else:
        similar_users = sorted_users[1:10]

    recommended_movies = []
    for user2 in similar_users:
        reviewed_movies = Reviews.query.filter(Reviews.user_id==user2[0]).filter(Reviews.movie_id != movie_row.id).all()

        for reviewed_movie in reviewed_movies:
            check = True
            user1_reviews = user1.reviews
            for user1_review in user1_reviews:
                if (user1_review.movie_id == reviewed_movie.movie_id):
                    check = False
                    break
            if(check):
                if(reviewed_movie.positive > reviewed_movie.negative and reviewed_movie.positive > reviewed_movie.neutral):
                    recommended_movies.append(reviewed_movie.movie_id)
    return recommended_movies
