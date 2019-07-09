import pandas as pd
import random
from tickets.utils import get_tf_idf_vector_for_text, clean, fit_bow_tfidf


def show_stack_overflow_ticket_tfidf(ticket_df_row, count_vectorizer, tfidf_transformer):
    """
    Prints a stack overflow ticket.
    :param ticket_df_row: Row of the pandas dataframe for the ticket
    :param count_vectorizer: BoW model
    :param tfidf_transformer: TF-IDF transformer
    :return: None
    """
    tf_idf_vector = get_tf_idf_vector_for_text(ticket_df_row["clean_title_body"], count_vectorizer, tfidf_transformer)
    keywords_df = pd.DataFrame(tf_idf_vector.T.todense(), index=count_vectorizer.get_feature_names(), columns=["tfidf"])
    keywords_df = keywords_df.sort_values(by=["tfidf"], ascending=False)

    # now print the results
    print("\n=====Title=====")
    print(ticket_df_row['title'])
    print("\n=====Body=====")
    print(ticket_df_row['body'])
    print("\n===Tags===\n")
    print(ticket_df_row['tags'])

    print("\n===Keywords===")
    for index, score in keywords_df[:20].iterrows():
        print(index + ":", '{0:.{1}f}'.format(score['tfidf'], 3))


def load_and_preprocess_so_tickets_data(filename, tqdmnb=False):
    """
    Loads the stack overflow tickets dataset and cleans it up.
    :param filename: filename
    :param tqdmnb: set to true if you're using it from a jupyter notebook and want to see progress bars
    :return: a pandas dataframe
    """
    df = pd.read_json(filename, lines=True)
    print("Schema:\n\n", df.dtypes)
    print("Number of questions,columns=", df.shape)
    if tqdmnb:
        from tqdm import tqdm_notebook
        tqdm_notebook().pandas()

    # clean-up
    if tqdmnb:
        print("Cleaning body texts: ")
    df['clean_body'] = df['body'].progress_apply(lambda x: clean(x)) if tqdmnb else df['body'].apply(lambda x: clean(x))
    if tqdmnb:
        print("Cleaning title texts: ")
    df['clean_title'] = df['title'].progress_apply(lambda x: clean(x)) if tqdmnb else df['title'].apply(lambda x: clean(x))
    df['clean_title_body'] = df['clean_title'] + ". " + df["clean_body"]
    return df


def train_so_tfidf(df):
    """
    Trains a BoW model and uses a TF-IDF transformer to compute TF-IDF vectors.
    :param df: The cleaned stack-overflow dataframe which contains the column 'clean_title_body'
    :return: tuple(word count matrix, TF-IDF matrix, BoW model, TF-IDF transformer)
    """
    X_train_counts, X_train_tfidf, count_vectorizer, tfidf_transformer = fit_bow_tfidf(df['clean_title_body'])
    return X_train_counts, X_train_tfidf, count_vectorizer, tfidf_transformer


def show_random_so_ticket(df, count_vectorizer, tfidf_transformer):
    """
    Shows a random stack-overflow ticket from the dataframe.
    :param df: Dataframe containing stack-overflow tickets.
    :param count_vectorizer: trained BoW model.
    :param tfidf_transformer: trained TF-IDF transformer.
    :return: None
    """
    indx = random.randint(1, 101)
    row = df.iloc[indx]
    show_stack_overflow_ticket_tfidf(row, count_vectorizer, tfidf_transformer)
