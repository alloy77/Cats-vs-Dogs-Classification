#Importing libraries
import os
import time
import shutil
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


#defining directories for dogs and cats folder
dog_dir=r"C:\Users\KIIT\OneDrive\Documents\CatvDog_CNN_Project\Dataset\PetImages\Dog"
cat_dir=r"C:\Users\KIIT\OneDrive\Documents\CatvDog_CNN_Project\Dataset\PetImages\Cat"
base_dir = r"C:\Users\KIIT\OneDrive\Documents\CatvDog_CNN_Project\Dataset\PetImages"

for folder in ["Cat", "Dog"]:
    folder_path = os.path.join(base_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.getsize(file_path) == 0:  # Check for empty files
            print(f"Deleting empty file: {file_path}")
            os.remove(file_path)
            

#defining the parameters to create a dataframe for dogs
dog_filenames = glob.glob(os.path.join(dog_dir, "*.jpg"))
dog_label="Dog"

#defining the parameters to create a dataframe for cats
cat_filenames = glob.glob(os.path.join(cat_dir, "*.jpg"))
cat_label="Cat"



#dataframes that contains dog images

dog_df=pd.DataFrame({"filename":dog_filenames, "label":dog_label})
print("dimension of dog df: ",dog_df.shape[0])

#dataframe that contains cat images
cat_df=pd.DataFrame({"filename":cat_filenames, "label":cat_label})
print("Dimension of cat df: ",cat_df.shape[0])


print("dimension of dog df: ",dog_df.shape[0])
print("Dimension of cat df: ",cat_df.shape[0])

#merging both dataframes into a singular dataframe
df = pd.concat([dog_df, cat_df], ignore_index=True)
print(df.head())

print("df size: ", df.size)

#splitting the merged dataframe into train set and test for making it ready for model learning"
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

print("train_df: ",train_df.head())
print("test_df: ",test_df.head())

train_df = train_df[train_df["filename"].apply(os.path.exists)]
test_df = test_df[test_df["filename"].apply(os.path.exists)]

# Display class distribution
print("Class distribution in train set:\n", train_df["label"].value_counts())
print("\nClass distribution in test set:\n", test_df["label"].value_counts())

# Define train and test folder paths
train_folder = os.path.join(base_dir, "train")
test_folder = os.path.join(base_dir, "test")

# Ensure train/test folders and subfolders exist
for folder in [train_folder, test_folder]:
    os.makedirs(os.path.join(folder, "Cat"), exist_ok=True)
    os.makedirs(os.path.join(folder, "Dog"), exist_ok=True)

# Function to move images to respective folders
def move_images(df, destination_folder):
    for index, row in df.iterrows():
        filename = row["filename"]
        label = row["label"]  # 'Cat' or 'Dog'
        label_folder = os.path.join(destination_folder, label)

        if not os.path.exists(filename):
            print(f"⚠️ Skipping missing file: {filename}")
            continue  # Skip missing files

        dest_path = os.path.join(label_folder, os.path.basename(filename))  # Keep original name

        shutil.copy2(filename, dest_path)

# Move train and test images
move_images(train_df, train_folder)
move_images(test_df, test_folder)

print(f"✅ All images have been organized into '{train_folder}' and '{test_folder}' with 'Cat' and 'Dog' subfolders.")

#Rechecking and removing corrupted files
for folder in ["train/Cat", "train/Dog", "test/Cat", "test/Dog"]:
    folder_path = os.path.join(base_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.getsize(file_path) == 0:  # Check for empty files
            print(f"Deleting empty file: {file_path}")
            os.remove(file_path)
            
# Define ImageDataGenerator for training set with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values (0-1)
    rotation_range=20,  # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Shift image width by 20%
    height_shift_range=0.2,  # Shift image height by 20%
    shear_range=0.2,  # Apply shearing transformations
    zoom_range=0.2,  # Zoom in/out by 20%
    horizontal_flip=True,  # Flip images horizontally
    fill_mode="nearest"  # Fill missing pixels
)

# Define ImageDataGenerator for test set (no augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define batch size
BATCH_SIZE = 32
IMG_SIZE = (150, 150)  # Resize images to 150x150 for uniformity

# Create training data generator
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",  # Use 'binary' since we have two classes (cat & dog)
    shuffle=True  # Shuffle the training data
)

# Create test data generator
test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",  
    shuffle=False  # No need to shuffle test data
)

print("✅ ImageDataGenerator applied successfully!")
print("Class labels found:", train_generator.class_indices)