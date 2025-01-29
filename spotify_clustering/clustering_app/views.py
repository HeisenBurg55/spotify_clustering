import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from bs4 import BeautifulSoup
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# Initialize Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=settings.SPOTIPY_CLIENT_ID,
    client_secret=settings.SPOTIPY_CLIENT_SECRET,
    redirect_uri=settings.SPOTIPY_REDIRECT_URI,
    scope="user-library-read playlist-read-private"
))

def spotify_clustering(request):
    context = {}
    if request.method == "POST":
        playlist_id = request.POST.get("playlist_id", "5hWptzO88cWoQwvwz5G1kK")  # 3cEYpjA9oz9GiPac4AsH4n

        try:
            results = sp.playlist_tracks(playlist_id, fields="items(track(id,name,artists(id,name),popularity))")
            # results = sp.playlist_tracks(playlist_id)

            tracks = results['items']

            # Extract data
            data = []
            for track in tracks:
                track_info = track['track']
                for artist in track_info['artists']:
                    data.append({
                        "artist_id": artist['id'],
                        "artist_name": artist['name'],
                        "track_popularity": track_info['popularity']
                    })

            df = pd.DataFrame(data)

            # Save the DataFrame as a CSV file locally
            # df.to_csv('spotify_data.csv', index=False)  # 'index=False' prevents saving the index as a column

            context = data_analysis(df)

        except Exception as e:
            context["error"] = str(e)

    return render(request, "spotify_clustering.html", context)

def upload_csv(request):
    context = {}
    if request.method == 'POST' and request.FILES['csv_file']:
        try:
            # Handle the uploaded csv
            uploaded_file = request.FILES['csv_file']
            fs = FileSystemStorage()
            file_path = fs.save(uploaded_file.name, uploaded_file)
            file_url = fs.url(file_path)

            df = pd.read_csv(file_path)

            context = data_analysis(df)

        except Exception as e:
            context["error"] = str(e)

    return render(request, "spotify_clustering.html", context)



def get_base64_plot():
    """Helper function to convert Matplotlib plots to base64 for rendering in templates."""
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode("utf-8")

def data_analysis(df):
    context = {}
    artist_stats = df.groupby("artist_name").agg(
        num_songs=("track_popularity", "count"),
        avg_popularity=("track_popularity", "mean")
    ).reset_index()

    # EDA Visualizations
    # Bar plot: Number of songs per artist
    plt.figure(figsize=(10, 6))
    artist_stats.sort_values("num_songs", ascending=False).head(10).plot.bar(
        x="artist_name", y="num_songs", legend=False, color="skyblue"
    )
    plt.title("Top 10 Artists by Number of Songs")
    plt.xlabel("Artist")
    plt.ylabel("Number of Songs")
    plt.tight_layout()
    bar_plot = get_base64_plot()

    # Scatter plot: Average popularity vs Number of songs
    plt.figure(figsize=(10, 6))
    plt.scatter(artist_stats['num_songs'], artist_stats['avg_popularity'], c="orange", alpha=0.7)
    plt.title("Artists: Number of Songs vs Average Popularity")
    plt.xlabel("Number of Songs")
    plt.ylabel("Average Popularity")
    plt.tight_layout()
    scatter_plot = get_base64_plot()

    # Clustering
    #Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(artist_stats[["num_songs", "avg_popularity"]])

    #Automatically determine optimal no of clusters with silhouette score metric
    silhouette_scores = []
    k_values = range(2, 11)  # Silhouette score requires at least 2 clusters

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    plt.tight_layout()
    elbow_plot = get_base64_plot()

    optimal_k = k_values[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    artist_stats["cluster"] = kmeans.fit_predict(X)

    # Cluster Visualization
    plt.figure(figsize=(10, 6))
    for cluster in range(optimal_k):
        cluster_data = artist_stats[artist_stats["cluster"] == cluster]
        plt.scatter(cluster_data["num_songs"], cluster_data["avg_popularity"], label=f"Cluster {cluster}")
    plt.title("Artist Clusters")
    plt.xlabel("Number of Songs")
    plt.ylabel("Average Popularity")
    plt.legend()
    plt.tight_layout()
    cluster_plot = get_base64_plot()

    # Pass plots and data to the template
    context.update({
        "bar_plot": bar_plot,
        "scatter_plot": scatter_plot,
        "elbow_plot": elbow_plot,
        "cluster_plot": cluster_plot,
        "artist_stats": artist_stats.to_dict(orient="records"),
    })

    return context


def web_scraping(request):
    if request.method == 'POST':
        artist_code = request.POST.get('artist_code')

        # URL for Pitbull's MusicBrainz page
        # url = "https://musicbrainz.org/artist/d262ea27-3ffe-40f7-b922-85c42d625e67"
        # url = "https://musicbrainz.org/artist/73e5e69d-3554-40d8-8516-00cb38737a1c"
        if artist_code:
            try:
                url = f"https://musicbrainz.org/artist/{artist_code}"

                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find all <a> tags with href starting with "/release-group/"
                release_links = soup.find_all('a', href=lambda href: href and href.startswith('/release-group/'))

                music_titles = [link.text.strip() for link in release_links]

                if music_titles:
                    df = pd.DataFrame(music_titles, columns=['Music Title'])

                    return render(request, 'spotify_clustering.html', {
                        'scraped_data': df.to_html(index=False),
                    })
                else:
                    return render(request, 'index.html', {
                        'error': 'No music titles found for this artist.'
                    })

            except Exception as e:
                return render(request, 'spotify_clustering.html', {
                    'error_scraping': str(e)
                })

    return render(request, 'spotify_clustering.html')