# 🎵 AutoGen Lyrics

This project uses AutoGen with Gemini Pro to generate song lyrics based on user input. It demonstrates how to set up and run a conversational AI agent that creates customized lyrics.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autogenlyrics-krgsknod3xugbb6mghgokr.streamlit.app/)

## 🚀 Features

- Integration with Gemini Pro via API
- AutoGen multi-agent chat setup
- Customizable prompts for lyric generation
- CLI-based interaction

## 🛠️ Setup

1. **Clone the repository**

```bash
git clone https://github.com/gelicheng/autogen_lyrics.git
cd autogen_lyrics
```

2. **Create and activate a virtual environment**

```bash
python -m venv autogen-env
autogen-env\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Replace your APIs**
   Update the autogen_spotify_app.py file with your Spotify API keys.
   Update the config.json file with your Gemini API key. Make sure that your API key is compatible with the Gemini model being used.

5. **Run the script**

```bash
.\autogen-env\Scripts\python.exe -m streamlit run .\autogen_spotify_app.py
```
