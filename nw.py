# Only works with square images at the moment. Update in progress...

import os
import re
import uuid
import time
import torch
import shutil
import sqlite3
import datetime
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

title = "NeuralWorks System I"
version = "1.0.0"

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

def dynamic_banner(title: str, subtitle: str):
  total_width = max(len(subtitle), 38)
  print(f"|{'-' * total_width}|")
  print(f"| {title}{' ' * (total_width - len(title) - 1)}|")
  print(f"|{'-' * total_width}|")
  print(f"| {subtitle}{' ' * (total_width - len(subtitle) - 1)}|")
  print(f"|{'-' * total_width}|")
  print(" ")

def warns(flag=False):
    print("WARNING! This is designed for professionals only!")
    print("         Please use the default model(s) if in doubt.")
    if flag:
        print("\n         Bad architecture design! Are you sure about this?")
    print("\nNOTICE! Input and output sizes will be recalculated dynamically.")
    print("        Make sure to check your architecture before saving!")

class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = vgg
        self.criterion = nn.MSELoss()
        
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        return self.criterion(x_vgg, y_vgg)

class Database:
    def __init__(self, db_name='architectures.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS Architectures (ArchitectureID CHAR(15) NOT NULL PRIMARY KEY, CreationDateTime TEXT, ModelType TEXT, ModelName TEXT, Architecture TEXT, AdditionalInfo TEXT')
        self.conn.commit()

    def add_architecture(self, arch_id, creation_time, model_type, model_name, architecture, additional_info):
        self.cursor.execute('INSERT INTO Architectures (ArchitectureID, CreationDateTime, ModelType, ModelName, Architecture, AdditionalInfo) VALUES (?, ?, ?, ?, ?, ?)', (arch_id, creation_time, model_type, model_name, architecture, additional_info))
        self.conn.commit()

    def get_all_architectures(self):
        self.cursor.execute('SELECT * FROM Architectures')
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

def data_management(db):
    while True:
        cls()
        dynamic_banner(title, "Data Management Menu")
        print("|-| Architecture Database |-|")
        print("1. Backup")
        print("2. Restore")
        print("\n3. Go Back")
        
        choice = input("\nPick an option: ")
        
        if choice == '1':
            backup_database(db)
            input("\nPress Enter to continue")
        elif choice == '2':
            restore_database(db)
            input("\nPress Enter to continue")
        elif choice == '3':
            return
        else:
            print("\nInvalid choice. Please try again. 2s...")
            time.sleep(2)

def backup_database(db):
    if not os.path.exists("backup"):
        os.makedirs("backup")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"backup/architectures_{timestamp}.db"
    
    shutil.copy2("architectures.db", backup_file)
    print(f"\nDatabase backed up to '{backup_file}'")

def restore_database(db):
    cls()
    dynamic_banner(title, "Data Management Menu")
    backup_files = [f for f in os.listdir("backup") if f.startswith("architectures_") and f.endswith(".db")]
    
    if not backup_files:
        print("No backup files found.")
        return
    
    for i, file in enumerate(backup_files):
        print(f"{i + 1}. {file}")
    
    choice = int(input("\nEnter the number of the backup file to restore: ")) - 1
    if choice < 0 or choice >= len(backup_files):
        print("\nInvalid choice. Restoration cancelled. 2s...")
        time.sleep(2)
        return
    
    confirm = input("\nAre you 100% sure you want to restore this backup? This will overwrite the current database. (Y/N): ")
    if confirm.lower() != 'y':
        print("\nRestoration cancelled. 2s...")
        time.sleep(2)
        return
    
    db.close()
    shutil.copy2(f"backup/{backup_files[choice]}", "architectures.db")
    time.sleep(1)
    cls()
    print("Database restored. Please restart the application.")
    exit(1)

class DefaultSR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DefaultSR, self).__init__()
        mid_channels = int(in_channels * 2)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class DefaultClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, input_size):
        super(DefaultClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Determine size of the f-map after two pool layers
        feature_map_size = input_size // 4
        fc_input_size = 64 * feature_map_size * feature_map_size
        
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DefaultDS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DefaultDS, self).__init__()
        mid_channels = int(in_channels * 1.2)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class DefaultMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DefaultMapping, self).__init__()
        mid_channels = int(in_channels * 1.2)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, layers):
        super(CustomModel, self).__init__()
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(eval(layer))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def main_menu(db, arch_flag):
    cls()
    dynamic_banner(title, "Main Menu")
    print("1. Build a new neural network model")
    
    if arch_flag:
        print("2. Train a neural network model")
        print("3. Run an inference on an existing model")
        print("\n4. Data Management")
        print("5. Information")
    else:
        print("\n2. Data Management")
        print("3. Information")
    
    print("\nX. Quit")
    
    choice = input("\nPick an option: ")
    return choice

def validate_directory(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"The directory '{directory}' does not exist.") # Print this instead. Don't crash everything!
    return directory

def get_image_files(directory):
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]
    if not image_files:
        raise ValueError(f"No valid image files found in the directory '{directory}'.")
    return image_files

def validate_class_count(count):
    try:
        count = int(count)
        if count < 2:
            raise ValueError("The number of classes must be at least 2.")
        return count
    except ValueError:
        raise ValueError("Please enter a valid integer for the number of classes.")

def tag_classes_manually(image_files, class_count):
    cls()
    dynamic_banner(title, "Model Builder Menu: Classifier")
    class_labels = []
    for i in range(class_count):
        label = input(f"Enter the name for class {i + 1}: ")
        class_labels.append(label)

    image_classes = {}
    for image in image_files:
        while True:
            cls()
            dynamic_banner(title, "Model Builder Menu: Classifier")
            print(f"Image: {image}\n")
            for i, label in enumerate(class_labels):
                print(f"{i + 1}. {label}")
            choice = input("\nEnter the class number for this image: ")
            try:
                choice = int(choice)
                if 1 <= choice <= class_count:
                    image_classes[image] = class_labels[choice - 1]
                    break
                else:
                    print("\nInvalid choice. Please try again.")
            except ValueError:
                print("\nPlease enter a valid number.")
    
    return image_classes

def check_class_tag_file(file_path, image_files):
    if not os.path.isfile(file_path):
        raise ValueError(f"The class tag file '{file_path}' does not exist.")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) != len(image_files):
        raise ValueError("The number of lines in the class tag file does not match the number of images.")
    
    image_classes = {}
    for image, label in zip(image_files, lines):
        image_classes[image] = label.strip()
    
    return image_classes

def build_custom_architecture():
    architecture = []
    in_channels = 3  # Assuming the input is RGB
    input_size = None  # Track the input size for fully connected layers
    feature_map_size = None  # Also make sure to track the feature map size
    warn = False  # For final FC layer NOTICE message
    initialised = False

    # Edge case mitigation
    def recalculate_input_size_and_channels():
        nonlocal in_channels, feature_map_size, input_size
        in_channels = 3
        input_size = None
        feature_map_size = None

        for layer in architecture:
            if 'nn.Conv2d' in layer:
                if input_size is None:
                    input_size = int(input("Enter the input size: "))
                    feature_map_size = (input_size, input_size)
                out_channels = int(re.search(r'(\d+),', layer).group(1))
                in_channels = out_channels  # Conv doesn't change the s-dimensions if padding is used!
            elif 'nn.MaxPool2d' in layer or 'nn.AvgPool2d' in layer:
                kernel_size = int(re.search(r'kernel_size=(\d+)', layer).group(1))
                feature_map_size = (feature_map_size[0] // kernel_size, feature_map_size[1] // kernel_size)
            elif 'nn.Flatten' in layer:
                in_features = in_channels * feature_map_size[0] * feature_map_size[1]
                in_channels = in_features
                feature_map_size = None

    while True:
        cls()
        dynamic_banner(title, "Custom Architecture Builder")
        warns()
        
        print("\nCurrent Architecture Stack:")
        for i, layer in enumerate(architecture):
            print(f"{i + 1}. {layer}")
        
        print("\nOptions:")
        print("1. Add convolutional layer")
        print("2. Add fully connected layer")
        print("3. Add activation function")
        print("4. Add pooling layer")
        print("\n5. Move layer up")
        print("6. Move layer down")
        print("7. Delete layer")
        print("\n8. Save architecture")
        print("9. Go Back")  # Change: Modification so that it doesn't cancel it; dump back to the custom/default selector.
        
        if not initialised and (not architecture and input_size is None):
            input_size = int(input("\nEnter the initial input size: "))
            feature_map_size = (input_size, input_size)
            continue
        
        choice = input("\nPick an option: ")
        
        print(" ")
        
        if choice == '1':
            if architecture == []:
                input_size = None  # Bug fix (Doesn't ask for input after deleting all layers)
                print("NOTICE! Ensure that the first layer connecting with the images are equal in size/resolution!\n")
            if input_size is None:
                input_size = int(input("Enter the input size: "))
                feature_map_size = (input_size, input_size)
            
            out_channels = int(input("Enter output channels: "))
            kernel_size = int(input("Enter kernel size: "))
            architecture.append(f"nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, padding={kernel_size//2})")
            in_channels = out_channels  # Conv doesn't change the s-dimensions if padding is used!
        elif choice == '2':
            if feature_map_size is None:
                print("WARNING! You must define the input size or add a convolutional/pooling layer before adding a fully connected layer. 3s...")
                time.sleep(3)
                continue
            if warn:
                print("NOTICE! If making the final layer, ensure the feature size = number of classes!")
                out_features = int(input("\nEnter output features: ")) # Maintain UI style
            else:
                out_features = int(input("Enter output features: "))
            in_features = in_channels * feature_map_size[0] * feature_map_size[1]
            architecture.append(f"nn.Flatten()")
            architecture.append(f"nn.Linear({in_features}, {out_features})")
            in_channels = out_features  # Update in_channels for the next layer
            feature_map_size = None  # Reset feature_map_size after adding a fully connected layer
        elif choice == '3':
            cls()
            warns(True) if architecture == [] else warns()
            print(" ")
            print("1. ReLU (Default)")
            print("2. Sigmoid")
            print("3. Tanh")
            print("4. LeakyReLU")
            print("5. GeLU")
            act_choice = input("\nChoose activation function: ")
            activation_functions = {
                '1': 'nn.ReLU()',
                '2': 'nn.Sigmoid()',
                '3': 'nn.Tanh()',
                '4': 'nn.LeakyReLU()',
                '5': 'nn.GELU()'
            }
            architecture.append(activation_functions.get(act_choice, 'nn.ReLU()'))
        elif choice == '4':
            if architecture == []:
                print("NOTICE! Ensure that the first layer connecting with the images are equal in size/resolution!\n")
            if input_size is None:
                input_size = int(input("Enter the input size: "))
                feature_map_size = (input_size, input_size)
            pool_type = input("Enter pool type (Max/Avg): ").lower()
            kernel_size = int(input("Enter kernel size: "))
            if pool_type == 'max':
                architecture.append(f"nn.MaxPool2d(kernel_size={kernel_size})")
                feature_map_size = (feature_map_size[0] // kernel_size, feature_map_size[1] // kernel_size)
            elif pool_type == 'avg':
                architecture.append(f"nn.AvgPool2d(kernel_size={kernel_size})")
                feature_map_size = (feature_map_size[0] // kernel_size, feature_map_size[1] // kernel_size)
            else:
                print("\nNOTICE! Unrecognised pool type. 3s...")
                time.sleep(3)
                continue
        elif choice == '5':
            try:
                i = int(input("Enter the index of the layer to move up: ")) - 1
                if not (0 < i < len(architecture)):
                    raise ValueError
                architecture[i-1], architecture[i] = architecture[i], architecture[i-1]
                recalculate_input_size_and_channels()
            except ValueError:
                print("\nERROR! Invalid index. Please try again.")
                continue
        elif choice == '6':
            try:
                i = int(input("Enter the index of the layer to move down: ")) - 1
                if not (0 <= i < len(architecture) - 1):
                    raise ValueError
                architecture[i], architecture[i+1] = architecture[i+1], architecture[i]
                recalculate_input_size_and_channels()
            except ValueError:
                print("\nERROR! Invalid index. Please try again.")
                continue
        elif choice == '7':
            try:
                i = int(input("Enter the index of the layer to delete: ")) - 1
                if not (0 <= i < len(architecture)):
                    raise ValueError
                del architecture[i]
                recalculate_input_size_and_channels()
            except ValueError:
                print("\nERROR! Invalid index. Please try again.")
                continue
        elif choice == '8':
            if not architecture:
                print("\nERROR! The architecture is empty! Cannot proceed. 3s...")
                time.sleep(3)
                continue
            else:
                if any('nn.Linear' in layer for layer in architecture):
                    return architecture
                else:
                    print("WARNING! Your architecture does not have a final fully connected layer. This will cause issues!")
                    print("         Please add one before saving. 3s...")
                    time.sleep(3)
                    warn = True
                    continue
        elif choice == '9':
            return None
        else:
            print("\nInvalid choice. 2s...")
            time.sleep(2)

def build_new_model(db):
    cls()
    dynamic_banner(title, "Model Builder Menu")
    print("1. Classifier model")
    print("2. Image processing model (SR, Deblur, etc.)\n")
    print("3. Go Back")
    
    choice = input("\nPick an option: ")
    
    if choice == '1':
        cls()
        dynamic_banner(title, "Model Builder Menu: Classifier")
        model_type = "Classifier"
        
        # Building a classifier model
        input_directory = input("Enter the directory of input images: ")
        input_directory = validate_directory(input_directory)
        
        try:
            image_files = get_image_files(input_directory)
        except ValueError as e:
            print(f"\nERROR! {e}")
            input("\nPress Enter to return to the main menu")
            return
        
        # Check that all the input images have the same resolution
        input_size = None
        for img_file in image_files:
            img_path = os.path.join(input_directory, img_file)
            img_size = Image.open(img_path).size
            if input_size is None:
                input_size = img_size
            elif img_size != input_size:
                print("\nERROR! All input images must have the same resolution.")
                input("\nPress Enter to return to the main menu")
                return
        
        class_count = input("\nEnter the number of classes: ")
        try:
            class_count = validate_class_count(class_count)
        except ValueError as e:
            print(f"\nERROR! {e}")
            input("\nPress Enter to return to the main menu")
            return
        
        cls()
        dynamic_banner(title, "Model Builder Menu: Classifier")
        print("1. Tag classes manually")
        print("2. Use existing class tag file")
        tag_choice = input("\nPick an option: ")
        
        if tag_choice == '1':
            image_classes = tag_classes_manually(image_files, class_count)
        elif tag_choice == '2':
            cls()
            dynamic_banner(title, "Model Builder Menu: Classifier")
            number_of_images = sum(1 for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))) if os.path.isdir(input_directory) else False
            print("NOTICE! Your images and tag file should meet the criteria:") # Add a check to see if the images are ordered, like 1-max value and tell the user off if not....
            if number_of_images != False:
                print(f"\n - Name ordered 0/1 through to {number_of_images - 1}/{number_of_images}.")
            else:
                print(f"\n - Name ordered 0/1 through to the maximum number of images.")
            print("   OR in alphabetical order.")
            print(" - Aligned with your images for each line.")
            print(" - Same number of unique classes as defined earlier.")
            tag_file = input("\nEnter the path to the class tag file: ")
            try:
                image_classes = check_class_tag_file(tag_file, image_files)  # Change: Check for unique classes per line. If number doesn't match with class_count, give ERROR!.
            except ValueError as e:
                print(f"\nERROR! {e}")
                input("\nPress Enter to return to the main menu")
                return
        else:
            print("Invalid choice. Returning to main menu.")
            return
        
        cls()
        dynamic_banner(title, "Model Builder Menu: Classifier")  # Change: Modification to see if the model name already exists, regardless of the model type.
        model_name = input("Enter a name for this model: ")
        while not model_name:
            print("\nNOTICE! Model name cannot be empty.")
            model_name = input("\nEnter a name for this model: ")
        
        cls()
        dynamic_banner(title, "Model Builder Menu: Classifier")
        print("Choose a neural network architecture to use with this model:")
        print("\n1. Use default classifier architecture")
        print("2. Define custom architecture")
        arch_choice = input("\nPick an option: ")
        
        if arch_choice == '1':
            architecture = "DefaultClassifier"
        elif arch_choice == '2':
            architecture = build_custom_architecture()
            if architecture is None:
                print("Custom architecture building cancelled. 2s...")
                time.sleep(2)
                return
            architecture = str(architecture)  # Convert the list to a string for storage
        else:
            print("Invalid choice. Returning to main menu.")
            return
        
        # Save to the database
        arch_id = str(uuid.uuid4())[:15]
        creation_time = datetime.datetime.now().isoformat()
        additional_info = {
            "classes": list(set(image_classes.values())),
            "input_directory": input_directory,
            "image_classes": image_classes,
            "input_size": input_size
        }
        
        cls()
        dynamic_banner(title, "Model Builder Menu: Classifier")
        db.add_architecture(arch_id, creation_time, model_type, model_name, str(architecture), str(additional_info))
        print(f"Model '{model_name}' saved successfully.")
        input("\nPress Enter to continue")
    
    elif choice == '2':
        # Building an image processing model
        cls()
        dynamic_banner(title, "Model Builder Menu: Image-to-Image")
        input_directory = input("Enter the directory of input images: ")
        input_directory = validate_directory(input_directory)
        output_directory = input("Enter the directory of output images: ")
        output_directory = validate_directory(output_directory)
        
        try:
            input_images = get_image_files(input_directory)
            output_images = get_image_files(output_directory)
        except ValueError as e:
            print(f"\nERROR! {e}")
            input("\nPress Enter to return to the main menu")
            return
        
        if len(input_images) != len(output_images):
            print("\nERROR! The number of input and output images must be the same.")
            input("\nPress Enter to continue")
            return
        
        input_size = None
        output_size = None
        for in_img, out_img in zip(input_images, output_images):
            in_path = os.path.join(input_directory, in_img)
            out_path = os.path.join(output_directory, out_img)
            in_size = Image.open(in_path).size
            out_size = Image.open(out_path).size
            
            if input_size is None:
                input_size = in_size
                output_size = out_size
            elif in_size != input_size or out_size != output_size:
                print("\nERROR! All input images must have the same size, and all output images must have the same size.")
                input("\nPress Enter to return to the main menu")
                return
        
        cls()
        dynamic_banner(title, "Model Builder Menu: Image-to-Image")
        model_name = input("Enter a name for this model: ")
        while not model_name:
            print("Model name cannot be empty.")
            model_name = input("Enter a name for this model: ")
        
        if output_size[0] > input_size[0] or output_size[1] > input_size[1]:
            model_type = "SuperResolution"
        elif output_size[0] < input_size[0] or output_size[1] < input_size[1]:
            model_type = "Downsampling"
        else:
            model_type = "Mapping"
        
        cls()
        dynamic_banner(title, "Model Builder Menu: Image-to-Image")
        print(f"1. Use default {model_type} architecture")
        print("2. Define custom architecture")
        arch_choice = input("\nPick an option: ")
        
        if arch_choice == '1':
            architecture = f"Default{model_type}"
        elif arch_choice == '2':
            architecture = build_custom_architecture()
            if architecture is None:
                print("Custom architecture building cancelled. 2s...")
                time.sleep(2)
                return
        else:
            print("Invalid choice. 2s...")
            time.sleep(2)
            return
        
        # Save to the database
        arch_id = str(uuid.uuid4())[:15]
        creation_time = datetime.datetime.now().isoformat()
        additional_info = {
            "input_size": input_size,
            "output_size": output_size,
            "input_directory": input_directory,
            "output_directory": output_directory
        }
        
        cls()
        dynamic_banner(title, "Model Builder Menu: Image-to-Image")
        db.add_architecture(arch_id, creation_time, model_type, model_name, str(architecture), str(additional_info))
        print(f"Model '{model_name}' saved successfully.")
        input("\nPress Enter to continue")
    
    elif choice == '3':
        return
    
    else:
        print("\nInvalid choice. 2s...")
        time.sleep(2)

class CustomDataset(Dataset):
    def __init__(self, input_dir, output_dir=None, image_classes=None, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_classes = image_classes
        self.transform = transform
        self.image_files = get_image_files(input_dir)
        if image_classes:
            self.class_to_i = {cls: i for i, cls in enumerate(set(image_classes.values()))}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        img_name = self.image_files[i]
        img_path = os.path.join(self.input_dir, img_name)
        input_image = Image.open(img_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)

        if self.output_dir:
            output_path = os.path.join(self.output_dir, img_name)
            output_image = Image.open(output_path).convert('RGB')
            if self.transform:
                output_image = self.transform(output_image)
            return input_image, output_image
        elif self.image_classes:
            class_name = self.image_classes[img_name]
            class_i = self.class_to_i[class_name]
            return input_image, torch.tensor(class_i, dtype=torch.long)
        else:
            return input_image

def train_model(db, arch_flag):
    cls()
    dynamic_banner(title, "Model Training Menu")
    for i, arch in enumerate(arch_flag):
        print(f"{i + 1}. {arch[3]} ({arch[2]})")
    
    choice = input("\nEnter the number of the model you want to train: ") # System to check if the model is already trained. Y/N...
    try:
        choice = int(choice) - 1
        if choice < 0 or choice >= len(arch_flag):
            raise ValueError
    except ValueError:
        print("\nInvalid choice. 2s...")
        time.sleep(2)
        return

    arch = arch_flag[choice]
    model_type = arch[2]
    model_name = arch[3]
    architecture = arch[4]
    additional_info = eval(arch[5])

    # Set defaults
    if model_type == "Classifier":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Determine input size(s)
    if model_type == "Classifier":
        sample_img = Image.open(os.path.join(additional_info["input_directory"], list(additional_info["image_classes"].keys())[0]))
        input_size = sample_img.size
    else:
        input_size = additional_info["input_size"]

    if architecture.startswith("Default"):
        if model_type == "Classifier":
            num_classes = len(additional_info["classes"])
            input_size = additional_info["input_size"][0]
            model = eval(architecture)(3, num_classes, input_size)  # Assuming RGB input
        else:
            model = eval(architecture)(3, 3)  # Assuming RGB input and output
    else:
        layers = eval(architecture)
        model = CustomModel(layers)

    if model_type == "Classifier":
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        dataset = CustomDataset(additional_info["input_directory"], image_classes=additional_info["image_classes"], transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        dataset = CustomDataset(additional_info["input_directory"], output_dir=additional_info["output_directory"], transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = input("\nEnter number of epochs (default: 100): ")
    try:
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError
    except ValueError:
        print("\nInvalid input. Using default value of 100 epochs.")
        epochs = 100

    print("\nAvailable loss functions:")
    print("1. MSE (Mean Squared Error)")
    print("2. BCE (Binary Cross-Entropy)")
    print("3. Cross-Entropy")
    print("4. VGG16 Perceptual Loss")
    
    loss_choice = input("\nEnter the number of the loss function you want to use (default: 1): ")
    
    if loss_choice == "2":
        criterion = nn.BCELoss()
    elif loss_choice == "3":
        criterion = nn.CrossEntropyLoss()
    elif loss_choice == "4" and model_type != "Classifier":
        vgg = torchvision.models.vgg16(pretrained=True).features[:29].eval()
        criterion = VGGPerceptualLoss(vgg)
    else:
        criterion = nn.MSELoss()

    print("\nAvailable optimizers:")
    print("1. Adam")
    print("2. SGD (Stochastic Gradient Descent)")
    
    optimizer_choice = input("\nEnter the number of the optimizer you want to use (default: 1): ")
    
    if optimizer_choice == "2":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = optim.Adam(model.parameters())

    cls()
    dynamic_banner(title, "Model Training Menu")
    start_time = datetime.datetime.now()
    print("NOTICE! Training in progress! Please do not shutdown your computer. 5s...\n")
    time.sleep(5)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if model_type == "Classifier":
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, labels.long())
                else:
                    # For MSE and BCE, labels are to use one-hot encoding
                    num_classes = len(set(additional_info["image_classes"].values()))
                    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
                    loss = criterion(outputs, labels_one_hot)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"NeuralWorks Trainer: [Epoch {epoch + 1}/{epochs}] [Loss: {running_loss / len(dataloader):.8f}]")

    cls()
    dynamic_banner(title, "Model Training Menu")
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), f"models/{model_name}.pth")
    print(f"Model '{model_name}' has been trained and saved.\nIt is now ready for inference.")
    print("\nStatistics:")
    print(f"\n - Elapsed Time: {elapsed_time}")
    print(f" - Final Loss: {running_loss / len(dataloader)}")
    input("\nPress Enter to continue")

def run_inference(db):
    
    # Fix: Models which are not trained yet are shown. Additional checks to be implemented.
    
    # Debug Log:
    
    # raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    # RuntimeError: Error(s) in loading state_dict for CustomModel:
    # Missing key(s) in state_dict: "layers.1.weight", "layers.1.bias", "layers.3.weight", "layers.3.bias".
    # Unexpected key(s) in state_dict: "layers.2.weight", "layers.2.bias".
    # size mismatch for layers.0.weight: copying a param with shape torch.Size([64, 3, 3, 3]) from checkpoint, the shape in current model is torch.Size([128, 3, 3, 3]).
    # size mismatch for layers.0.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([128]).
    
    cls()
    dynamic_banner(title, "Inference Menu")
    architectures = db.get_all_architectures()
    valid_models = []
    for i, arch in enumerate(architectures):
        model_path = f"models/{arch[3]}.pth"
        if os.path.exists(model_path):
            print(f"{len(valid_models) + 1}. {arch[3]} ({arch[2]})")
            valid_models.append(arch)
    
    if not valid_models:
        print("No valid models found. Please train a model first.")
        input("Press Enter to return to the main menu")
        return

    choice = input("\nEnter the number of the model you want to use for inference: ")
    try:
        choice = int(choice) - 1
        if choice < 0 or choice >= len(valid_models):
            raise ValueError
    except ValueError:
        print("\nInvalid choice. 2s...")
        time.sleep(2)
        return

    arch = valid_models[choice]
    model_type = arch[2]
    model_name = arch[3]
    architecture = arch[4]
    additional_info = eval(arch[5])

    if architecture.startswith("Default"):
        if model_type == "Classifier":
            num_classes = len(additional_info["classes"])
            input_size = additional_info["input_size"][0]
            model = eval(architecture)(3, num_classes, input_size)
        else:
            model = eval(architecture)(3, 3)
    else:
        layers = eval(architecture)
        model = CustomModel(layers)
    
    model.load_state_dict(torch.load(f"models/{model_name}.pth"))
    model.eval()

    cls()
    dynamic_banner(title, "Inference Menu")
    
    input_file = input("Enter the filename of the input image: ")
    if not os.path.exists(input_file):
        print("\nERROR! Input file not found.")
        input("\nPress Enter to continue")
        return

    input_image = Image.open(input_file).convert('RGB')
    
    if model_type == "Classifier":
        transform = transforms.Compose([
            transforms.Resize(additional_info["input_size"]),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(additional_info["input_size"]),
            transforms.ToTensor(),
        ])

    input_tensor = transform(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    if model_type == "Classifier":
        _, predicted = torch.max(output, 1)
        print(f"\nPredicted class: {additional_info['classes'][predicted.item()]}")
    else:
        output_image = transforms.ToPILImage()(output.squeeze(0))
        output_file = f"output_{os.path.basename(input_file)}"
        output_image.save(output_file)
        print(f"\nOutput image saved as '{output_file}'")

    input("\nPress Enter to continue")

def display_information():
    cls()
    # To be done later. NeuralWorks by Oguzhan Cagirir CC BY-NC-SA 4.0

def main():
    db = Database()
    arch_flag = db.get_all_architectures()
    
    while True:
        choice = main_menu(db, arch_flag)
        
        if choice == '1':
            build_new_model(db)
            # Update arch_flag after building a new model
            arch_flag = db.get_all_architectures()
        elif choice == '2':
            if arch_flag:
                train_model(db, arch_flag)
            else:
                data_management(db)
        elif choice == '3':
            if arch_flag:
                run_inference(db)
            else:
                display_information()
        elif choice == '4' and arch_flag:
            data_management(db)
        elif choice == '5' and arch_flag:
            display_information()
        elif choice in ["x", "X"]:
            cls()
            print(f"Thanks for using {title}!")
            exit(1)

if __name__ == "__main__":
    main()
