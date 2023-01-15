import streamlit as st
# Basic Streamlit Settings
st.set_page_config(page_title='SongSuggest', layout = 'wide', initial_sidebar_state = 'auto')
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from load_css import local_css
from PIL import Image
import pydeck as pdk
import plotly.figure_factory as ff
import base64
import streamlit.components.v1 as components
import webbrowser

# Loading css file
local_css("style.css")


# Sidebar Section
def spr_sidebar():
    with st.sidebar:
        st.info('**SongSuggest**')
        home_button = st.button("About Us")
        data_button = st.button("Dataset")
        rec_button = st.button('Recommendation Engine')
        algo_button = st.button('Algorithm and Prediction')
        conc_button = st.button('Conclusion')
        st.session_state.log_holder = st.empty()
        if home_button:
            st.session_state.app_mode = 'home'
        if data_button:
            st.session_state.app_mode = 'dataset'
        if algo_button:
            st.session_state.app_mode = 'algo'
        if rec_button:
            st.session_state.app_mode = 'recommend'
        if conc_button:
            st.session_state.app_mode = 'conclusions'


# Dataset Page
def dataset_page():
    st.markdown("<br>", unsafe_allow_html=True)
    """
    # Spotify Gen Track Dataset
    -----------------------------------
    Here We are using Spotity Gen Track Dataset, this dataset contains n number of songs and some metadata is included as well such as name of the playlist, duration, number of songs, number of artists, etc.
    """

    dataset_contains = Image.open('images/dataset_contains.png')
    st.image(dataset_contains, width =900)

    """
   
    - The data has three files namely spotify_albums, spotify_artists and spotify_tracks from which We are extracting songs information.
    - There is spotify features.csv files which contains all required features of the songs that We are using.

    """
    
    st.markdown("<br>", unsafe_allow_html=True)
    '''
    # Final Dataset 
    '''
    '''
    - Enhanced data
    '''
    dataframe1 = pd.read_csv('filtered_track_df.csv')
    st.dataframe(dataframe1)
    st.markdown("<br>", unsafe_allow_html=True)


# Algorithm and Prediction Page
def algo_page():
    st.header("1. Calculate Algorithms Accuracy")
    st.markdown(
        'Trainig the model and using Popularity as a Y-parameter to judge how accurate the algorithm comes out')

    st.header("Algorithms")
    st.subheader("Linear Regression")
    code = '''LR_Model = LogisticRegression()
    LR_Model.fit(X_train, y_train)
    LR_Predict = LR_Model.predict(X_valid)
    LR_Accuracy = accuracy_score(y_valid, LR_Predict)
    print("Accuracy: " + str(LR_Accuracy))
    LR_AUC = roc_auc_score(y_valid, LR_Predict)
    print("AUC: " + str(LR_AUC))
    Accuracy: 0.7497945543198379
    AUC: 0.5'''
    st.code(code, language='python')

   
    st.subheader("K-Nearest Neighbors Classifier")
    code = '''KNN_Model = KNeighborsClassifier()
    KNN_Model.fit(X_train, y_train)
    KNN_Predict = KNN_Model.predict(X_valid)
    KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
    print("Accuracy: " + str(KNN_Accuracy))
    KNN_AUC = roc_auc_score(y_valid, KNN_Predict)
    print("AUC: " + str(KNN_AUC))
    Accuracy: 0.7763381361967896
    AUC: 0.6890904291795135'''
    st.code(code, language='python')


    st.subheader("Decision Tree Classifier")
    code = '''DT_Model = DecisionTreeClassifier()
    DT_Model.fit(X_train, y_train)
    DT_Predict = DT_Model.predict(X_valid)
    DT_Accuracy = accuracy_score(y_valid, DT_Predict)
    print("Accuracy: " + str(DT_Accuracy))
    DT_AUC = roc_auc_score(y_valid, DT_Predict)
    print("AUC: " + str(DT_AUC))
    Accuracy: 0.8742672437407549
    AUC: 0.8573960839474465
    '''
    st.code(code, language='python')

    
    st.subheader("Random Forest")
    code = '''RFC_Model = RandomForestClassifier()
    RFC_Model.fit(X_train, y_train)
    RFC_Predict = RFC_Model.predict(X_valid)
    RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
    print("Accuracy: " + str(RFC_Accuracy))
    RFC_AUC = roc_auc_score(y_valid, RFC_Predict)
    print("AUC: " + str(RFC_AUC))
    Accuracy: 0.9357365912452748
    AUC: 0.879274665020435'''
    st.code(code, language='python')


    
    
# Load Data and n_neighbors_uri_audio are helper functions inside Recommendation Page
# Loads the track from filtered_track_df.csv file
def load_data():
    df = pd.read_csv(
        "filtered_track_df.csv")
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df


genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

# Fetches the Nearest Song according to Genre start_year and end year.
def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"] == genre) & (
        exploded_track_df["release_year"] >= start_year) & (exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios


# Recommendation Page

def rec_page():
    
    st.header("RECOMMENDATION ENGINE")
   
    with st.container():
        col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col3:
        st.markdown("***Choose your genre:***")
        genre = st.radio(
            "",
            genre_names, index=genre_names.index("Pop"))
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider(
            'Select the year range',
            1990, 2019, (2015, 2019)
        )
        acousticness = st.slider(
            'Acousticness',
            0.0, 1.0, 0.5)
        danceability = st.slider(
            'Danceability',
            0.0, 1.0, 0.5)
        energy = st.slider(
            'Energy',
            0.0, 1.0, 0.5)
        instrumentalness = st.slider(
            'Instrumentalness',
            0.0, 1.0, 0.0)
        valence = st.slider(
            'Valence',
            0.0, 1.0, 0.45)
        tempo = st.slider(
            'Tempo',
            0.0, 244.0, 118.0)
        tracks_per_page = 12
        test_feat = [acousticness, danceability,
                     energy, instrumentalness, valence, tempo]
        uris, audios = n_neighbors_uri_audio(
            genre, start_year, end_year, test_feat)
        tracks = []
        for uri in uris:
            track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
                uri)
            tracks.append(track)
    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [
            genre, start_year, end_year] + test_feat
    current_inputs = [genre, start_year, end_year] + test_feat

    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page

    current_tracks = tracks[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    if st.session_state['start_track_i'] < len(tracks):
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            if i % 2 == 0:
                with col1:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

            else:
                with col3:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

    else:
        st.write("No songs left to recommend")

    st.code("Algorithms that we have used in this filtering are the k-nearest neighbours and Random Forest")

    

    st.subheader("Graph Representing Audio Features Importance")
    random_forest_audio_importance = Image.open('images/random_forest_audio_importance_feature.jpg')
    st.image(random_forest_audio_importance, caption ="random_forest_audio_feature_importance", width = 900)


# Home Page

def home_page():
    st.subheader('Team Members A_D_T:')
    st.write('Kalyan Venkatesh Poludasu (Student ID:- 110089120)')
    st.write('Marc Smith (Student ID:- 110087601)')
    st.write('Nencyben Patel (Student ID:- 110086804)')
    st.write('Pranjali Prajapati (Student ID:-110087663)')

# Footer Section
def spr_footer():
    st.markdown('---') 
    st.markdown(
        '© Copyright 2022 - SongSuggest By Team A_D_T')
# Conclusion Page

def conclusions_page():

    st.header('Conclusion ')
    st.subheader("Model Performance Summary")
    st.success("Accuracy Test Results")
    algo_accuracy = Image.open(
        'images/algos_accuracy.png')
    st.image(algo_accuracy, width=400)
    
    algo_auc = Image.open(
        'images/models_auc_area_under_curve.png')
    st.image(algo_auc, width=400)
    st.write("The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classess. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.")
     
    st.write('In this project, we have presented a novel framework for Music recommendation that is driven by data and simple effective recommendation system for generating songs as per users choice.')
    st.write('Moving forward, we will use a larger Spotify database by using the Spotify API to collect our own data, and explore different algorithms to predict popularity score rather than doing binary classification.')
st.session_state.app_mode = 'recommend'

def main():
    
    spr_sidebar()
    st.header("SongSuggest (SF)")
    st.markdown(
        '**SongSuggest** is a online Robust Music Recommendation Engine where in you can finds the best songs that suits your taste.  ')
    st.markdown('Along with the rapid expansion of digital music formats, managing and searching for songs has become signiﬁcant. The purpose of this project is to build a recommendation system to allow users to discover music based on their listening preferences. Therefore in this model I focused on the public opinion to discover and recommend music.')    
    
    if st.session_state.app_mode == 'dataset':
        dataset_page()

    if st.session_state.app_mode == 'algo':
        algo_page()

    if st.session_state.app_mode == 'recommend':
        rec_page()

    if st.session_state.app_mode == 'conclusions':
        conclusions_page()

    if st.session_state.app_mode == 'home':
        home_page()

    spr_footer()


# Run main()
if __name__ == '__main__':
    main()


