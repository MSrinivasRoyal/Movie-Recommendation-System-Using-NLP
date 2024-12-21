import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import nltk
from nltk.corpus import stopwords
import re
from io import BytesIO
import ast
from sklearn.metrics import precision_score, recall_score

#Libraries for Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import streamlit as st

st.markdown(
    """
    <style>
    [data-testid="stApp"] {
    opacity: 0.7;
    background-image:  url("https://wallpaper.forfun.com/fetch/51/51ad99d47b0f0080ceb513856dbbdf1d.jpeg");
    background-size: cover;
    }
    
    [data-testid="stHeader"]{
    background-color:rgba(0,0,0,0);
    }
    
    [data-testid="stToolbar"]{
    right: 2rem;
    }
    
    [data-testid="stSidebar"]{
    background-image:  url("https://mrwallpaper.com/images/high/solid-black-4k-batman-saws37lhxhidmjee.jpg");
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

movies_df = pd.read_csv('tmdb_5000_movies.csv')

# Simulated user feedback dataset
# Movies the user has watched and whether they liked it or not (1 = liked, 0 = disliked)
user_feedback = {
    'The Matrix': 1,
    'Inception': 1,
    'The Lion King': 0,
    'Titanic': 0,
    'Pulp Fiction': 1,
    'Interstellar': 1,
    'Joker': 0,
    'Avengers': 1,
    'Avatar': 1,
    'Forrest Gump': 1,
    'Fight Club': 0,
    'The Godfather': 1,
    'The Dark Knight': 1,
    'The Shawshank Redemption': 1,
    'Gladiator': 1,
    'Schindler\'s List': 0,
    'Frozen': 0,
    'Toy Story': 1,
    'The Wolf of Wall Street': 1,
    'The Social Network': 1,
    'Mad Max: Fury Road': 0,
    'Star Wars': 1,
    'Harry Potter and the Sorcerer\'s Stone': 1,
    'The Silence of the Lambs': 1,
    'The Lord of the Rings: The Fellowship of the Ring': 1,
    'Back to the Future': 1,
    'Spider-Man': 0,
    'Black Panther': 1,
    'The Terminator': 1,
    'Iron Man': 1,
    'Jurassic Park': 0,
    'The Hunger Games': 0,
    'Deadpool': 1,
    'The Revenant': 1,
    'The Great Gatsby': 0,
    'Guardians of the Galaxy': 1,
    'Shrek': 1,
    'Wonder Woman': 1,
    'Up': 1,
    'The Departed': 1,
    'Braveheart': 1,
    'Django Unchained': 1,
    'The Big Short': 1,
    'A Beautiful Mind': 1,
    'The Prestige': 1,
    'Finding Nemo': 1,
    'Logan': 1,
    'The Incredibles': 1,
    'La La Land': 0,
    'Pirates of the Caribbean: The Curse of the Black Pearl': 1,
    'It': 0,
    'The Conjuring': 0,
    'A Quiet Place': 1,
    'Zootopia': 1,
    'The Grand Budapest Hotel': 1,
    '12 Years a Slave': 1,
    'The Hateful Eight': 0,
    'The Bourne Identity': 1,
    'Whiplash': 1,
    'Moonlight': 1,
    'Her': 1,
    'Blade Runner 2049': 1,
    'Gravity': 0,
    'Inside Out': 1,
    'Les Misérables': 1,
    'The Truman Show': 1,
    '300': 1,
    'Cast Away': 1,
    'Slumdog Millionaire': 1,
    'The Pianist': 1,
    'The Notebook': 0,
    'The Imitation Game': 1,
    'Good Will Hunting': 1,
    'The Perks of Being a Wallflower': 1,
    'The Girl with the Dragon Tattoo': 1,
    'No Country for Old Men': 1,
    'The Avengers: Age of Ultron': 1,
    'Dr. Strange': 1,
    'Captain America: Civil War': 1,
    'Thor: Ragnarok': 1,
    'John Wick': 1,
    'Pacific Rim': 0,
    'Edge of Tomorrow': 1,
    'Minions': 0,
    'The Hangover': 1,
    'Suicide Squad': 0
}

def parse_genres(genre_data):
    try:
        genres = ast.literal_eval(genre_data)  # Safely evaluate the string representation of the list
        return [genre['name'] for genre in genres]  # Extract genre names
    except (ValueError, SyntaxError):
        return []

movies_df = movies_df[['title', 'overview', 'genres']].dropna()

movies_df['genres'] = movies_df['genres'].apply(parse_genres)

movies_df['cleaned_title'] = movies_df['title'].apply(lambda x: x.lower().replace(" ", ""))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
    
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movies_df = movies_df.reset_index(drop=True)

from tmdbv3api import TMDb, Movie
import requests
from PIL import Image
from io import BytesIO

# Initialize TMDb API
tmdb = TMDb()
tmdb.api_key = '442bad6c754858897b423098d2732eb3'

def fetch_movie_poster(title):
    movie = Movie()
    search_results = movie.search(title)
    
    if search_results:
        movie_id = search_results[0].id
        movie_details = movie.details(movie_id)
        poster_path = movie_details.poster_path
        full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        return full_poster_url  # Returning the poster URL
    return None

def recommend_movies_with_posters(title, cosine_sim, df):
    # Preprocess the input title
    title = title.lower().replace(" ", "")
    
    # Find the closest matching title using fuzzy matching
    closest_match = process.extractOne(title, df['cleaned_title'])
    
    if closest_match is not None:
        closest_title = closest_match[0]
        idx = df[df['cleaned_title'] == closest_title].index[0]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 similar movies
        
        movie_indices = [i[0] for i in sim_scores]
        movie_titles = df['title'].iloc[movie_indices].tolist()  # Convert to list
        movie_overviews = df['overview'].iloc[movie_indices].tolist()  # Convert to list
        movie_genres = df['genres'].iloc[movie_indices].tolist()  # Convert to list
        similarity_scores = [score[1] for score in sim_scores]
        
        return movie_titles, movie_overviews, movie_genres, similarity_scores
    else:
        return [], [], [], []



def plot_word_cloud(overviews):
    text = " ".join(overviews)
    wordcloud = WordCloud(width=600, height=300, background_color='black').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
def evaluate_recommendations(recommended_movies, user_feedback):
    true_labels = []
    predicted_labels = []
    
    for movie in recommended_movies:
        if movie in user_feedback:
            true_labels.append(user_feedback[movie])  # Actual feedback
            predicted_labels.append(1)  # Recommended movie (predicted as relevant)
        else:
            # If the movie is not in the feedback dataset, assume it's not relevant
            predicted_labels.append(0)
    
    # Handle cases where there are no true labels (i.e., no relevant movies to evaluate)
    if len(true_labels) == 0 or sum(true_labels) == 0:
        return None, None
    
    # Precision and Recall Calculation
    precision = precision_score(true_labels, predicted_labels, zero_division=1)
    recall = recall_score(true_labels, predicted_labels, zero_division=1)
    
    return precision, recall
    
user_feedback_live = {}
    
# Streamlit app layout
st.title('Movie Recommendation System')


# Sidebar for filters
st.sidebar.header('Filters')
selected_genre = st.sidebar.multiselect('Select Genre', options=movies_df['genres'].explode().unique())
min_rating = st.sidebar.slider('Minimum Rating', 0.0, 10.0, 5.0)

user_input = st.text_input("Enter a Recently Watched Movie Title:")

if user_input:
    # Get movie recommendations
    movie_titles, movie_overviews, movie_genres, similarity_scores = recommend_movies_with_posters(user_input, cosine_sim, movies_df)

    # Handle empty recommendation case
    if not movie_titles:
        st.write("No recommendations found. Please check the movie title or try another one.")
    else:
        # Filter by selected genre if applicable
        if selected_genre:
            filtered_titles = []
            filtered_overviews = []
            filtered_genres = []
            
            for title, overview, genre in zip(movie_titles, movie_overviews, movie_genres):
                if any(g in genre for g in selected_genre):
                    filtered_titles.append(title)
                    filtered_overviews.append(overview)
                    filtered_genres.append(genre)

            movie_titles = filtered_titles
            movie_overviews = filtered_overviews
            movie_genres = filtered_genres

        # Plot word cloud for movie overviews
        plot_word_cloud(movie_overviews)
        st.write(f"Recommendations for **{user_input}**:")

        # Display recommendations and collect feedback
        user_feedback_live = {}
        for i in range(len(movie_titles)):
            st.subheader(movie_titles[i])
            st.write(f"Genres: {', '.join(movie_genres[i])}")
            st.write(f"Plot: {movie_overviews[i]}")

            # Fetch poster and display
            poster_url = fetch_movie_poster(movie_titles[i])
            if poster_url:
                response = requests.get(poster_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=f"Poster for {movie_titles[i]}", use_column_width=True)

            # Collect user feedback
            user_feedback_live[movie_titles[i]] = st.radio(f"Do you like {movie_titles[i]}?", ('No', 'Yes'))
            
            
        st.write("Evaluating based on user feedback...")
        # Map feedback to binary format
        mapped_feedback = {title: 1 if feedback == 'Yes' else 0 for title, feedback in user_feedback_live.items()}

        # Evaluate the recommendations using precision and recall
        precision, recall = evaluate_recommendations(movie_titles, mapped_feedback)

        # Display precision and recall results
        if precision is not None and recall is not None:
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
        else:
            st.write("Not enough data for evaluation.")
    
    
    
# Store user preferences in session state
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

if st.button('Save Preferences'):
    st.session_state.user_preferences['selected_genre'] = selected_genre
    st.session_state.user_preferences['min_rating'] = min_rating
    st.success("Preferences saved!")
