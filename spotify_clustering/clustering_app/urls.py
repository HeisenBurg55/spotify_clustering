from django.urls import path
from . import views

urlpatterns = [
    path('', views.spotify_clustering, name='spotify_clustering'),
    path('web_scraping/', views.web_scraping, name='web_scraping'),
    path('upload_csv/', views.upload_csv, name='upload_csv')
]