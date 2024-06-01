from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash 
from django.contrib.auth.forms import UserCreationForm, UserChangeForm, PasswordChangeForm
from django.contrib import messages 
from .forms import SignUpForm, EditProfileForm , BirdsForm
import os
from django.http import HttpResponse
from .models import UserPrediction
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use('agg')
# from keras.preprocessing.image import load_img,img_to_array

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import load_img
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from keras.preprocessing import image
# Create your views here.
def home(request): 
	return render(request, 'home.html', {})

def login_user (request):
	if request.method == 'POST': #if someone fills out form , Post it 
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(request, username=username, password=password)
		if user is not None:# if user exist
			login(request, user)
			messages.success(request,('Youre logged in'))
			return redirect('home') #routes to 'home' on successful login  
		else:
			messages.success(request,('Error logging in'))
			return redirect('login') #re routes to login page upon unsucessful login
	else:
		return render(request, 'login.html', {})

def logout_user(request):
	logout(request)
	messages.success(request,('Youre now logged out'))
	return redirect('home')

def register_user(request):
	if request.method =='POST':
		form = SignUpForm(request.POST)
		if form.is_valid():
			form.save()
			username = form.cleaned_data['username']
			password = form.cleaned_data['password1']
			user = authenticate(username=username, password=password)
			login(request,user)
			messages.success(request, ('Youre now registered'))
			return redirect('home')
	else: 
		form = SignUpForm() 

	context = {'form': form}
	return render(request, 'register.html', context)

def edit_profile(request):
	if request.method =='POST':
		form = EditProfileForm(request.POST, instance= request.user)
		if form.is_valid():
			form.save()
			messages.success(request, ('You have edited your profile'))
			return redirect('home')
	else: 		#passes in user information 
		form = EditProfileForm(instance= request.user) 

	context = {'form': form}
	return render(request, 'edit_profile.html', context)
	#return render(request, 'authenticate/edit_profile.html',{})



def change_password(request):
	if request.method =='POST':
		form = PasswordChangeForm(data=request.POST, user= request.user)
		if form.is_valid():
			form.save()
			update_session_auth_hash(request, form.user)
			messages.success(request, ('You have edited your password'))
			return redirect('home')
	else: 		#passes in user information 
		form = PasswordChangeForm(user= request.user) 

	context = {'form': form}
	return render(request, 'change_password.html', context)







def display(request):
    if request.method == 'GET':
        return render(request, 'app.html')

def about(request):
    return render(request, 'about.html')


def fonction(request):
    if request.method == 'GET':
        return render(request, 'fonction.html')


def demo(request):
    return render(request, 'demo.html')


from io import BytesIO
from django.http import JsonResponse

model = tf.keras.models.load_model('static/keras_model.h5')
model2 = tf.keras.models.load_model('static/binary.h5')


from PIL import Image



#preprocessing f image is fundus image 
def process_image(image):
    # Open the image using Pillow
    img = Image.open(image)
    # Resize the image (assuming the required dimensions are 224x224)
    img = img.resize((224, 224))
    # Convert the image to a NumPy array and normalize pixel values
    processed_image = np.array(img) / 255.0
    # image_array = np.asarray(img)
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # data = normalized_image_array

    return processed_image






# Preprocessing for the second model
def process_image_binary(image):
    	
    img = load_img(image, target_size=(256, 256))
    input_arr = img_to_array(img) 
    input_arr = np.expand_dims(input_arr, axis=0)
    return input_arr

import tempfile

# # New predict function with two models
# def predict(request):
#     if request.method == 'POST':
#         # Get the uploaded image from the request.FILES dictionary
#         image_file = request.FILES.get('image')
#         if image_file:
#             try:
#                 # Save the uploaded image to a temporary file
#                 with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#                     for chunk in image_file.chunks():
#                         temp_file.write(chunk)
#                 # Get the file path of the temporary file
#                 image_file_path = temp_file.name
                
#                 # Preprocess the image for binary prediction
#                 binary = process_image_binary(image_file_path)
#                 binary_prediction = model2.predict(binary)[0][0]
#                 print("Binary Prediction:", binary_prediction)
                
#                 if binary_prediction < 1.0:
#                     # Process the image
#                     processed_image = process_image(image_file_path)
#                     # Make predictions using the first model
#                     predictions = model.predict(np.expand_dims(processed_image, axis=0))
#                     # Log predictions to console
#                     print("Predictions:", predictions.tolist())
                    
#                     # Return predictions as JSON
#                     return JsonResponse({'predictions': predictions.tolist()})
#                 else:
#                     # Return a message indicating it is not a fundus image
#                     return JsonResponse({'message': "The uploaded image is not a fundus image."})
                
#             except Exception as e:
#                 print("An error occurred:", e)
#                 return JsonResponse({'error': str(e)}, status=500)
#             finally:
#                 # Delete the temporary file after use
#                 if image_file_path:
#                     os.unlink(image_file_path)
#         else:
#             print("No image file found in the request.")
#             return JsonResponse({'error': 'No image file found'}, status=400)
#     else:
#         print("Only POST requests are supported.")
#         return JsonResponse({'error': 'Only POST requests are supported'}, status=405)

from .models import UserPrediction

def predict(request):
    if request.method == 'POST':
        # Get the uploaded image from the request.FILES dictionary
        image_file = request.FILES.get('image')
        if image_file:
            try:
                # Save the uploaded image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    for chunk in image_file.chunks():
                        temp_file.write(chunk)
                # Get the file path of the temporary file
                image_file_path = temp_file.name
                
                # Preprocess the image for binary prediction
                binary = process_image_binary(image_file_path)
                binary_prediction = model2.predict(binary)[0][0]
                print("Binary Prediction:", binary_prediction)
                
                if binary_prediction < 1.0:
                    # # Process the image
                    # processed_image = process_image(image_file_path)
                    # # Make predictions using the first model
                    # # predictions = model.predict(np.expand_dims(processed_image, axis=0))
                    # predictions = model.predict(processed_image)
                    # pred_new = predictions[0]
                    # predictions = max(pred_new)
                    # index = pred_new.tolist().index(predictions)
                    # # Log predictions to console
                    # print("Predictions:", predictions.tolist())
                    # tick_label = ['normal', 'légère', 'modérée', 'sévère', 'proliférante']
                    # result = []
                    # if index == 0:
                    #  result.append("No DR")
                    # elif index == 1:
                    #    result.append("Mild")
                    # elif index == 2:
                    #      result.append("Moderate")
                    # elif index == 3:
                    #      result.append("Sever")
                    # elif index == 4:
                    #      result.append("Proliferative")

                    # accuracy = round(predictions, 2)
                    # result.append("-")
                    # result.append(accuracy * 100)
                    # # Save predictions to database
                    # # prediction_text = " ".join(str(prediction) for prediction in predictions.tolist())
                    # prediction_text = " ".join(str(result) for results in result.tolist())

                    # UserPrediction.objects.create(user=request.user, prediction=prediction_text)
                    
                    # # Return predictions as JSON
                    # # return JsonResponse({'predictions': predictions.tolist()})
                    
                    # return JsonResponse({'predictions': result.tolist()})

                    processed_image = process_image(image_file_path)
                    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

# Make predictions using the model
                    predictions = model.predict(processed_image)
                    pred_new = predictions[0]

# Find the index of the maximum prediction
                    index = np.argmax(pred_new)

# Log predictions to console
                    print("Predictions:", pred_new.tolist())

# Define labels
                    tick_label = ['normal', 'légère', 'modérée', 'sévère', 'proliférante']

# Define result list
                    result = []

# Map index to corresponding label
                    labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
                    result.append(labels[index])

# Calculate accuracy
                    accuracy = round(pred_new[index] * 100, 2)

                    # Append accuracy to result list
                    result.append("-")
                    result.append(accuracy)

                    # Save predictions to database
                    prediction_text = " ".join(str(res) for res in result)
                    UserPrediction.objects.create(user=request.user, prediction=prediction_text)

# Return predictions as JSON
                    return JsonResponse({'predictions': result})
                else:
                    # Return a message indicating it is not a fundus image
                    return JsonResponse({'message': "The uploaded image is not a fundus image."})
                
            except Exception as e:
                print("An error occurred:", e)
                return JsonResponse({'error': str(e)}, status=500)
            finally:
                # Delete the temporary file after use
                if image_file_path:
                    os.unlink(image_file_path)
        else:
            print("No image file found in the request.")
            return JsonResponse({'error': 'No image file found'}, status=400)
    else:
        print("Only POST requests are supported.")
        return JsonResponse({'error': 'Only POST requests are supported'}, status=405)

from .utils import render_to_pdf
from django.contrib.auth.decorators import login_required
from django.template.loader import get_template

@login_required
def generate_pdf(request):
    predictions = UserPrediction.objects.filter(user=request.user).order_by('-timestamp')[:10]
    context = {'predictions': predictions, 'user': request.user}
    pdf = render_to_pdf('pdf_template.html', context)
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="predictions.pdf"'
    return response



# def predict(request):
#     if request.method == 'POST':
#         # Get the uploaded image from the request.FILES dictionary
#         image_file = request.FILES.get('image')
#         if image_file:
#             try:
				
#                 # Process the image
#                 processed_image = process_image(image_file)
#                 # Make predictions
#                 predictions = model.predict(np.expand_dims(processed_image, axis=0))
#                 # Log predictions to console
#                 print("Predictions:", predictions.tolist())
				
#                 # Return predictions as JSON
#                 return JsonResponse({'predictions': predictions.tolist()})
#             except Exception as e:
#                 return JsonResponse({'error': str(e)}, status=500)
#         else:
#             return JsonResponse({'error': 'No image file found'}, status=400)
#     else:
#         return JsonResponse({'error': 'Only POST requests are supported'}, status=405)