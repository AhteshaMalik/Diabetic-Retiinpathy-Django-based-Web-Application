
from django.contrib import admin
from django.urls import path, include 
from authenticate import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('authenticate.urls')),
    path('about/', views.about, name='about'),
    path('demo/', views.demo, name='demo'),
]
