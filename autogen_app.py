import streamlit as st
import requests
import json
import urllib
from ollama import Ollama  # Assuming Ollama is set up locally
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Constants
CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"
REDIRECT_URI = "YOUR_REDIRECT_URI"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPE = "playlist-read-private playlist-read-collaborative user-library-read"
BASE_URL = "https://api.spotify.com/v1"

# Initialize Ollama
ollama_client = Ollama(model="gemma3:1B")

# Function to fetch access token from Spotify
def get_spotify_token():
    auth_response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials"
        },
        auth=(CLIENT_ID, CLIENT_SECRET)
    )
    return auth_response.json().get("access_token")

# Function to fetch playlists from the user
def get_user_playlists(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{BASE_URL}/me/playlists", headers=headers)
    return response.json().get("items", [])

# Function to fetch song features (e.g., genre, tempo) from Spotify
def get_song_features(track_id, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{BASE_URL}/audio-features/{track_id}", headers=headers)
    return response.json()

# Function to analyze the song lyrics or features using Ollama LLM (Gemma3:1B)
def analyze_song_with_ollama(song_features):
    # Construct prompt for Ollama to analyze features or lyrics
    prompt = f"Analyze the following song features and provide a recommendation for a similar song: {song_features}"
    
    # Get response from Ollama LLM
    response = ollama_client.chat(messages=[{"role": "user", "content": prompt}])
    return response['text']

# Function to recommend songs based on analysis
def recommend_similar_songs(song_features, all_songs):
    # Use cosine similarity or some other metric to recommend songs
    recommendations = []
    for song in all_songs:
        sim_score = cosine_similarity([song_features], [song['features']])
        recommendations.append((song['name'], sim_score[0][0]))
    
    # Sort by similarity score
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:5]

# Streamlit UI to interact with the assistant
st.set_page_config(page_title="Music Playlist Assistant", layout="wide")

st.title("Playlist Management Assistant")
st.markdown("This assistant will help you manage your playlists and suggest new songs based on your preferences.")

# Fetch Spotify token
access_token = get_spotify_token()

# If token is valid, show playlists and song features
if access_token:
    playlists = get_user_playlists(access_token)
    st.sidebar.selectbox("Select Playlist", [pl["name"] for pl in playlists])

    selected_playlist = st.sidebar.selectbox("Select Playlist to View", playlists)
    if selected_playlist:
        playlist_name = selected_playlist["name"]
        st.write(f"Fetching details for '{playlist_name}'...")

        # Fetch tracks from the selected playlist
        playlist_tracks = []
        for track in selected_playlist["tracks"]["items"]:
            track_name = track["track"]["name"]
            track_id = track["track"]["id"]
            features = get_song_features(track_id, access_token)

            # Append the song name and its features for analysis
            playlist_tracks.append({
                "name": track_name,
                "features": features
            })

        # Analyze the features of songs using Ollama
        for song in playlist_tracks:
            song_analysis = analyze_song_with_ollama(song["features"])
            st.write(f"Song: {song['name']}")
            st.write(f"Analysis: {song_analysis}")

        # Recommend new songs based on song analysis
        recommendations = recommend_similar_songs(song["features"], playlist_tracks)
        st.write("Song Recommendations based on your playlist:")
        for rec in recommendations:
            st.write(f"- {rec[0]} (Similarity Score: {rec[1]:.2f})")
else:
    st.write("Error: Could not fetch Spotify token. Please check your credentials.")
