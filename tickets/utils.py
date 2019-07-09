from bs4 import BeautifulSoup, NavigableString
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import re


def get_soup_strings(soup, ignore_tags={'code'}):
    texts = []
    for tag in soup.descendants:
        if isinstance(tag, NavigableString):
            if tag.parent.name not in ignore_tags:
                if str(tag).strip():
                    texts.append(str(tag).strip())
    return texts


def clean(text):
    """
    Cleans the passed english text by
    -- removing HTML tags
    -- removing all the special characters except common linguistic punctuations
    -- convert to lowercase
    :param text: text to be cleaned
    :return: cleaned text
    """
    soup = BeautifulSoup(text, "lxml")
    text = " ".join(get_soup_strings(soup))
    # remove multiple spaces
    re.sub(r"\s+", " ", text, flags=re.I)
    # remove special characters and digits, keep commas and fullstops
    # !&,.:;?
    text = re.sub("(\\d|[^!&,.:;?a-zA-Z])+", " ", text)
    text = text.lower()
    return text


def get_tf_idf_vector_for_text(text, count_vectorizer, tfidf_transformer):
    """
    Generates TF-IDF vector for the provided free text according to the trained TF-IDF model.
    Unseen words are ignored, hence do not impact the tfidf scores.
    :param text: Free text for which you want to generate a tfidf vector.
    :param count_vectorizer: Traind BoW model.
    :param tfidf_transformer: Traind TF-IDF transformer.
    :return: the TF-IDF vector.
    """
    tf_idf_vector = tfidf_transformer.transform(count_vectorizer.transform([text]))
    return tf_idf_vector


def fit_bow_tfidf(text_dataset_array):
    """
    Fits a BoW model and uses a TF-IDF transformer on it to compute TF-IDF scores.
    :param text_dataset_array: numpy array of strings, or pandas series. This means you can pass all values of a
    pandas data frame like this df["column_name"]
    :return: tuple(word count matrix, TF-IDF matrix, BoW model, TF-IDF transformer)
    """
    # Extracting features from text files
    count_vectorizer = CountVectorizer(min_df=5, stop_words=ENGLISH_STOP_WORDS)

    X_train_counts = count_vectorizer.fit_transform(text_dataset_array)

    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_counts, X_train_tfidf, count_vectorizer, tfidf_transformer
