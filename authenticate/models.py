from django.db import models
from django.contrib.auth.models import User

class Birds(models.Model):
    # hotel_Main_Img = models.ImageField(upload_to='images/')
    file = models.FileField(upload_to='media')


class UserPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prediction = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} - {self.timestamp}'



# Create your models here.
