import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer



music_df = pd.read_csv('songdata.csv')
music_df = music_df.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
music_df['text'] = music_df['text'].str.lower().replace(r'[^\w\s]','').replace(r'\n',' ', regex=True)


stemmer = PorterStemmer()
def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)
music_df['text'] = music_df['text'].apply(lambda x: tokenization(x))

tfidfvector = TfidfVectorizer()
music_matrix = tfidfvector.fit_transform(music_df['text'])
music_similarity = cosine_similarity(music_matrix)


movie_vectorizer = TfidfVectorizer()

movie_df = pd.read_csv('movies.csv')
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movie_df[feature] = movie_df[feature].fillna('')
movie_combined_features = movie_df['genres'] + ' ' + movie_df['keywords'] + ' ' + movie_df['tagline'] + ' ' + movie_df['cast'] + ' ' + movie_df['director']
# movie_matrix = TfidfVectorizer.fit_transform(movie_combined_features)
movie_matrix = movie_vectorizer.fit_transform(movie_combined_features)
movie_similarity = cosine_similarity(movie_matrix)


def main():
    st.set_page_config(page_title="RISE( Robust Intelligent System for Recommendations and Experiences)", page_icon=":bar_chart:")
    st.title("RISE(Robust Intelligent System for Recommendations and Experiences)")
    option = st.sidebar.selectbox("Select a Recommendation System", ("Music", "Movies"))
    
    if option == "Music":
        music_recommendation()
    elif option == "Movies":
        movie_recommendation()

def music_recommendation():
    st.header("Music Recommendation System")
    song_name = st.text_input("Enter your favorite song:", "")
    if st.button("Recommend"):
        recommendations = music_recommend(song_name)
        st.subheader("Recommended Songs:")
        for i, song in enumerate(recommendations, start=1):
            st.write(f"{i}. {song}")
        
def movie_recommendation():
    st.header("Movie Recommendation System")
    movie_name = st.text_input("Enter your favorite movie:", "")
    if st.button("Recommend"):
        recommendations = movie_recommend(movie_name)
        st.subheader("Recommended Movies:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")

def music_recommend(song):
    find_close_match = difflib.get_close_matches(song, music_df['song'].tolist())
    close_match = find_close_match[0]
    index_of_the_song = music_df[music_df.song == close_match].index[0]
    similarity_scores = list(enumerate(music_similarity[index_of_the_song]))
    sorted_similar_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_songs = [music_df.iloc[song[0]]['song'] for song in sorted_similar_songs[1:21]]
    return recommended_songs

def movie_recommend(movie):
    find_close_match = difflib.get_close_matches(movie, movie_df['title'].tolist())
    close_match = find_close_match[0]
    index_of_the_movie = movie_df[movie_df.title == close_match]['index'].values[0]
    similarity_scores = list(enumerate(movie_similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_movies = [movie_df.iloc[movie[0]]['title'] for movie in sorted_similar_movies[1:21]]
    return recommended_movies

if __name__ == '__main__':
    main()
