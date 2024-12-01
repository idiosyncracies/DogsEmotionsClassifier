'''
Train and predict dog emotion by using UI
'''

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from tqdm import tqdm
import random
import os
import time
import numpy as np
from torchvision.io import read_image
import torch.optim.lr_scheduler as lr_scheduler
import math
import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tkinter import messagebox
from tkinter import font
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import matplotlib.pyplot as plt
# %matplotlib inline


root = tk.Tk()
root.title("Dog emotion classification")


root.geometry("900x800")
root.resizable(False, False)
fixed_font_size = 14
fixed_font = font.Font(family="DejaVu Sans", size=fixed_font_size)


notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)


predict_frame = ttk.Frame(notebook)
notebook.add(predict_frame, text='Predict')


choose_net_label_pred = tk.Label(predict_frame, text='Classification:          ', font=fixed_font)
choose_net_label_pred.grid(row=0, column=13,columnspan=3 ,padx=10, pady=10, sticky='w')
choose_net_combobox_pred = ttk.Combobox(predict_frame,values=['ViT', 'resnet', 'alexnet', 'CNN'], font=fixed_font)
choose_net_combobox_pred.grid(row=1, column=13,columnspan=5 ,padx=10, pady=10, sticky='w')
choose_net_combobox_pred.current(3)


num_class = 4
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_class)  

from torchvision import models

# from torchvision.models import ResNet50_Weights, AlexNet_Weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_class = 4

def import_image():
    global image_file_path
    global image
    image_file_path = filedialog.askopenfilename(title="Select images", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_file_path:
        print("Image path:", image_file_path)
        image = Image.open(image_file_path)
        show_image = Image.open(image_file_path)
        width, height = show_image.size
        if(width/height!=1):
            if(width/height>1):
                show_image = show_image.resize((350, int(height/(width/350))))
            if(width/height<1):
                show_image = show_image.resize((int(width/(height/350)),350))
        if(width/height==1):
            show_image = show_image.resize((350,350)) 
        photo = ImageTk.PhotoImage(show_image)
        output_label.configure(image=photo)
        output_label.image = photo  

def import_model():
    global model_file_path
    model_file_path = filedialog.askopenfilename(title="Select Weights file", filetypes=[("Model files", "*.pth")])
    if model_file_path:
        print("Path:", model_file_path)

    # model.load_state_dict(torch.load(model_file_path, map_location=device))  


def load_model():
    global model
    model_name = choose_net_combobox_pred.get()  
    # model_name = 'resnet'

    assert model_name in ['alexnet', 'resnet', 'ViT', 'CNN'], f'Model to be selected as alexnet, resnet, or ViT.'

    if model_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)  
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_class) 
    elif model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)  
    elif model_name == 'ViT':
        # optional 1: use torch's buildin model
        model = models.vit_b_16(weights=models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_class)

    #     import sys
    #     sys.path.append('./models')
    #     from my_ViT import ViT
    #     model = ViT(
    #         img_size=224,
    #         patch_size=16,
    #         embed_dim=768,
    #         hidden_dim=3072,
    #         num_heads=12,
    #         num_layers=12,
    #         num_classes=4 )

    else:  ## CNN
        import sys

        sys.path.append('./models')
        from cnn import CNN

        model = CNN()

    model = model.to(device)
    print(f'Model: {model_name}, Device: {device}.')

    # current_file_path = os.path.abspath(__file__)


    checkpoint = torch.load(model_file_path, map_location=device)


    model.load_state_dict(checkpoint['model_state_dict'])
    model_text = ''

    epochs = checkpoint['epoch']
    model_state_dict = checkpoint['model_state_dict']
    learning_rate = checkpoint['learning_rate']
    batch_size = checkpoint['batch_size']
    # optimizer_state_dict = checkpoint['optimizer_state_dict']
    # loss_function = checkpoint['loss_function']

    model_text += 'Model：' + model_name + '\n'
    model_text += 'Epoch：' + str(epochs) + '\n'
    model_text += 'Learning rate：' + str(learning_rate) + '\n'
    model_text += 'Batch size：' + str(batch_size) + '\n'
    # model_text+='Loss function：'+loss_function+'\n' 
    # model_text+='OPtimizer：'+str(optimizer_state_dict)+'\n' 

    model_info_text.config(state=tk.NORMAL)
    model_info_text.delete('1.0', tk.END)
    model_info_text.insert(tk.END, model_text + '\n')
    model_info_text.config(state=tk.DISABLED)


# 预测操作
def predict():
    model.eval()  
    model.to(device) 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(), 
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)
    input_image = input_image.to(device)
    with torch.no_grad():
        output = model(input_image)
    probabilities = F.softmax(output[0], dim=0)
    labels = ['angry', 'happy', 'relaxed', 'sad']
    text = ''
    sorted_indices = torch.argsort(probabilities, descending=True)  
    for index in sorted_indices:
        predicted_label = labels[index.item()]
        probability = probabilities[index].item()
        print(f"{predicted_label}: {probability}")
        text += predicted_label + ':' + "{:.1f}%".format(probability * 100) + '\t'
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_label = labels[predicted_class_index]

    result_text.config(state=tk.NORMAL)
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, 'Emotion type：' + labels[predicted_class_index] + '\n')
    result_text.insert(tk.END, 'Emotion probability：\n' + text + '\n')
    result_text.config(state=tk.DISABLED)

    print("Result:", predicted_label)





root_path = os.path.join(os.environ['HOME'],'./Documents/datasets/dog_emotion/')
images_path = os.path.join(root_path,'./images/')  


def start_training():
    df = pd.DataFrame(columns=['file', 'emotion'])

    emotions = os.listdir(images_path)

    for emotion in tqdm(emotions):
        images_names = pd.Series(os.listdir(images_path + emotion), name='file')
        images_names = emotion + '/' + images_names

        emotions_types = pd.Series(len(images_names) * [emotion], name='emotion')
        tmp_df = pd.concat([images_names, emotions_types], axis=1)

        df = pd.concat([df, tmp_df], ignore_index=True)

    labels_file = os.path.join(root_path, './images_labels.csv')
    if not os.path.isfile(labels_file):
        df.to_csv(labels_file)

    def label_to_class(label):
        labels = ['angry', 'happy', 'relaxed', 'sad']
        return torch.Tensor([float(label == l) for l in labels], )

    def class_to_label(c):
        labels = ['angry', 'happy', 'relaxed', 'sad']
        idx = c.tolist().index(1)
        if idx >= len(labels):
            print('NO LABEL TO THIS CLASS')
            return None
        return labels[idx]

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
    ])

    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    from torch.utils.data import Dataset
    class Dataset(Dataset):
        def __init__(self, images_path, labels_csv_file_path, transform=None, augment=None):
            self.images_path = images_path
            self.transform = transform
            self.labels = pd.read_csv(labels_csv_file_path)
            self.augment = augment

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)  
            label = torch.tensor([0., 0., 0., 0.])  
            try:
                image = read_image(self.images_path + self.labels.iloc[idx, 1])
                label = label_to_class(self.labels.iloc[idx, 2])
                if self.transform:
                    image = self.transform(image)
                if self.augment:
                    if random.random() > 0.33:  
                        image = self.augment(image)
            except Exception as e:
                pass
            return image, label

    # batch_size = 8  
    batch_size = int(choose_batch_size_combobox.get())

    full_dataset = Dataset(images_path, labels_file, transform=tf, augment=data_augmentation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # train_images, train_labels = next(iter(train_data))

    # train_model_name = 'ViT'   
    train_model_name = choose_net_combobox.get()

    num_class = 4
    # assert train_model_name in ['alexnet', 'resnet', 'ViT', 'CNN'], f'Model to be selected as alexnet, resnet, or ViT.'

    if train_model_name == 'alexnet':
        train_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT) 
        num_ftrs = train_model.classifier[-1].in_features
        train_model.classifier[-1] = nn.Linear(num_ftrs, num_class)  
    elif train_model_name == 'resnet':
        train_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  
        num_ftrs = train_model.fc.in_features
        train_model.fc = nn.Linear(num_ftrs, num_class) 
    elif train_model_name == 'ViT':
        # optional 1: use torch's buildin model
        train_model = models.vit_b_16(weights=models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = train_model.heads.head.in_features
        train_model.heads.head = nn.Linear(num_features, num_class)

    #     ### optional 2: use scatched ViT code , 
    #     import sys
    #     sys.path.append('./models')
    #     from my_ViT import ViT
    #     train_model = ViT(
    #         img_size=224,
    #         patch_size=16,
    #         embed_dim=768,
    #         hidden_dim=3072,
    #         num_heads=12,
    #         num_layers=12,
    #         num_classes=4 )

    else:  ## CNN
        import sys
        sys.path.append('./models')
        from cnn import CNN
        train_model = CNN()

    train_model = train_model.to(device)

    def train(train_model, optim, criterion, train_data, val_data, epochs=20):
        losses = {
            'train': [],
            'val': []
        }

        precision_draw={
            'train': [],
            'val': []
        }

        recalls_train=[]
        recalls_val=[]

        f1_list = []

        def update_plot(train_losses, val_losses, train_precision, val_precision, train_recalls,
                        f1_scores):
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            axs[0, 0].plot(train_losses, label='train loss')
            axs[0, 0].plot(val_losses, label='val loss')
            axs[0, 0].set_xlabel('Epochs', fontsize=14)
            axs[0, 0].set_ylabel('Loss', fontsize=14)
            axs[0, 0].legend()

            axs[0, 1].plot(train_precision, label='train precision')
            axs[0, 1].plot(val_precision, label='val precision')
            axs[0, 1].set_xlabel('Epochs', fontsize=14)
            axs[0, 1].set_ylabel('Precision', fontsize=14)
            axs[0, 1].legend()

            axs[1, 0].plot(train_recalls, label='train recall')
            axs[1, 0].set_xlabel('Epochs', fontsize=14)
            axs[1, 0].set_ylabel('Recall', fontsize=14)
            axs[1, 0].legend()

            axs[1, 1].plot(f1_scores, label='f1 score')
            axs[1, 1].set_xlabel('Epochs', fontsize=14)
            axs[1, 1].set_ylabel('F1 Score', fontsize=14)
            axs[1, 1].legend()

            canvas = FigureCanvas(fig)
            fig.canvas.draw()

            img = ImageTk.PhotoImage(fig)

            l_e_label.config(image=img)
            l_e_label.config(image=img)
            precision_label.config(image=img)
            precision_label.config(image=img)
            recall_label.config(image=img)
            recall_label.config(image=img)
            f1_label.config(image=img)

            l_e_label.img = img
            l_e_label.img = img
            precision_label.img = img
            precision_label.img = img
            recall_label.img = img
            recall_label.img = img
            f1_label.img = img






        def calculate_f1(predictions, labels, threshold=0.5):
            predictions_binary = (predictions > threshold).astype(int)  
            true_positives = np.sum(predictions_binary * labels)  
            predicted_positives = np.sum(predictions_binary)  
            actual_positives = np.sum(labels) 

            precision = true_positives / (predicted_positives + 1e-10)  
            recall = true_positives / (actual_positives + 1e-10)  

            f1_score = 2 * precision * recall / (precision + recall + 1e-10) 

            return f1_score


        for epoch in tqdm(range(epochs)):
            # train
            train_model.train()
            train_loss = 0
            for X_batch, Y_batch in train_data:
                optim.zero_grad()
                X_batch_cuda = X_batch.to(device)
                Y_batch_cuda = Y_batch.to(device)
                X_pred_cuda = train_model(X_batch_cuda)
                loss = criterion(X_pred_cuda, Y_batch_cuda)
                loss.backward()  
                optim.step()  
                scheduler.step() 
                train_loss += loss
            train_loss = train_loss.item() / len(train_data)
            losses['train'].append(train_loss)


            print(X_pred_cuda.argmax(dim=1).size())
            print(Y_batch_cuda.size())
            TP_train = np.sum((X_pred_cuda.argmax(dim=1) == Y_batch_cuda.argmax(dim=1)).cpu().numpy())
            FP_train = np.sum((X_pred_cuda.argmax(dim=1) != Y_batch_cuda.argmax(dim=1)).cpu().numpy())
            precision_train = TP_train / (TP_train + FP_train)
            precision_draw['train'].append(precision_train)

            # val
            train_model.eval()
            X_valbatch, Y_valbatch = next(iter(val_data))  
            with torch.no_grad():
                X_valpred = train_model(X_valbatch.to(device))

            TP_val = np.sum((X_valpred.argmax(dim=1) == Y_valbatch.argmax(dim=1).to(device)).cpu().numpy())
            FP_val = np.sum((X_valpred.argmax(dim=1) != Y_valbatch.argmax(dim=1).to(device)).cpu().numpy())
            precision_val = TP_val / (TP_val + FP_val)
            precision_draw['val'].append(precision_val)

            TP = np.sum((X_valpred.cpu().numpy() >= 0.5) & (Y_valbatch.cpu().numpy() == 1))
            FN = np.sum((X_valpred.cpu().numpy() < 0.5) & (Y_valbatch.cpu().numpy() == 1))
            recall = TP / (TP + FN)
            recalls_train.append(recall)

            val_loss = criterion(X_valpred, Y_valbatch.to(device))
            losses['val'].append(val_loss.item())


            predictions_all = []  
            labels_all = [] 
            for X_batch, Y_batch in val_data:
                with torch.no_grad():
                    X_batch_cuda = X_batch.to(device)
                    predictions_batch = train_model(X_batch_cuda)
                predictions_all.extend(predictions_batch.cpu().numpy())
                labels_all.extend(Y_batch.numpy())
            f1 = calculate_f1(np.array(predictions_all), np.array(labels_all))
            f1_list.append(f1)


            print(f'Epoch: {epoch + 1}/{epochs}; train_loss: {train_loss}; val_loss: {val_loss}')

            update_plot(losses['train'], losses['val'], precision_draw['train'], precision_draw['val'], recalls_train
                        , f1_list)

        return losses



    epochs=int(choose_epochs_combobox.get())

    # lr = 0.001      
    lr=float(choose_lr_combobox.get())
    lrf = 0.01


    # criterion = torch.nn.CrossEntropyLoss() 
    if choose_loss_fun_combobox.get()=='Cross Entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif choose_loss_fun_combobox.get()=='Mean Squared Error':
        criterion = torch.nn.MSELoss()

    pg = [p for p in train_model.parameters() if p.requires_grad]

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    optim = torch.optim.SGD(pg, lr=lr, momentum=0.9,
                            weight_decay=5E-5)  ## option 2: # optim = torch.optim.Adam(train_model.parameters())
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=lf)

    ## start to train
    start = time.time()

    losses = train(train_model, optim, criterion, train_data, val_data, epochs=epochs)
    end = time.time()
    print('Train time per epoch:', (end - start) / epochs, 'seconds.')

    checkpoint = {
        'epoch': epochs,
        'model_state_dict': train_model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'learning_rate': lr,
        'batch_size': batch_size,
        'loss function': criterion
    }

    folder_name = os.path.join(root_path, './checkpoints')

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    current_time = datetime.datetime.now()


    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

    ensure_directory_exists(folder_name)
    pth_name = os.path.join(folder_name, train_model_name + '_' + formatted_time + '_'  + str(epochs) + 'epochs.pth')
    # check_point = torch.save(checkpoint, pth_name)
    torch.save(checkpoint, pth_name) 

    plt.figure(figsize=(7, 4))
    plt.plot(losses['train'], label='train loss')
    plt.plot(losses['val'], label='val loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    # plt.savefig(os.path.join(folder_name, train_model_name + '_' + 'loss.pdf'))

    #evaluate:
    val_images, val_labels = next(iter(val_data))
    emotions = os.listdir(images_path)
    val_preds = train_model(val_images.to(device))

    class_names = ['angry', 'happy', 'relaxed', 'sad']

    def evaluate(train_model, val_data):
        train_model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in val_data:
                outputs = train_model(inputs.to(device))
                _, predicted = torch.max(outputs, 1)  
                _, targets = torch.max(targets, 1)  ## convert label of one-hot to scalar in 0,1,2,3
                all_targets.extend(targets.cpu().numpy())  
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='macro')
        recall = recall_score(all_targets, all_predictions, average='macro')
        f1 = f1_score(all_targets, all_predictions, average='macro')

        return accuracy, precision, recall, f1, all_targets, all_predictions

    accuracy, precision, recall, f1, all_targets, all_predictions = evaluate(train_model, val_data)
    result = f'Accuracy:{accuracy} \nPrecision: {precision}\nRecall: {recall}\nF1 score: {f1}'
    print(result)   
    messagebox.showinfo("Train result", result+'\nweights are saved in：\n'+pth_name)



    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # Plot confusion matrix
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    # plt.xlabel('Predicted labels', fontsize=14)
    # plt.ylabel('True labels', fontsize=14)
    # plt.title('Model: ' + train_model_name, fontsize=14)
    # plt.savefig(os.path.join(folder_name, train_model_name + '_' + 'confusionMatrix_' + result + '.pdf'))
    # plt.show()




def stop_training():
    # TODO
    pass


output_label = tk.Label(predict_frame)
output_label.grid(row=0, column=0, rowspan=10, columnspan=13, padx=10, pady=10, sticky='nsew')
image_init = Image.new("RGB", (350, 350), "white")
photo_init = ImageTk.PhotoImage(image_init)
output_label.configure(image=photo_init)

# null_label = tk.Label(predict_frame)
# null_label.grid(row=14, column=16, columnspan=21, padx=10, pady=10, sticky='nsew')

import_button = tk.Button(predict_frame, text='Load images', command=import_image, font=fixed_font)
import_button.grid(row=14, column=3, padx=10, pady=10, sticky='w')

predict_button = tk.Button(predict_frame, text='Select weights file', command=import_model, font=fixed_font)
predict_button.grid(row=14, column=7, padx=10,  pady=10, sticky='w')

predict_button = tk.Button(predict_frame, text='Load model', command=load_model, font=fixed_font)
predict_button.grid(row=14, column=11, padx=10,  pady=10, sticky='w')

predict_button = tk.Button(predict_frame, text='Predict', command=predict, font=fixed_font)
predict_button.grid(row=14, column=15, padx=10,  pady=10, sticky='w')

# weights_label = tk.Label(predict_frame, text='Model:')
# weights_label.grid(row=0, column=11, padx=10, pady=10, sticky='w')
# weights_combobox = ttk.Combobox(predict_frame)
# weights_combobox.grid(row=0, column=12, padx=10, pady=10, sticky='w')


model_info_label = tk.Label(predict_frame, text='Model info:           ', font=fixed_font)
model_info_label.grid(row=2, column=13, padx=10, pady=10, sticky='w')
model_info_text = tk.Text(predict_frame)
model_info_text.grid(row=3, column=13, columnspan=8, rowspan=7, padx=10, pady=10, sticky='nsew')

result_label = tk.Label(predict_frame, text='Prediction:', font=fixed_font)
result_label.grid(row=10, column=0, padx=10, pady=10, sticky='w')
result_text = tk.Text(predict_frame, height=10, width=10)
result_text.grid(row=11, column=0, columnspan=21, rowspan=3, padx=10, pady=10, sticky='nsew')


train_frame = ttk.Frame(notebook)
notebook.add(train_frame, text='Train')

choose_net_label = tk.Label(train_frame, text='Model:', font=fixed_font)
choose_net_label.grid(row=0, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_net_combobox = ttk.Combobox(train_frame,values=['ViT', 'resnet', 'alexnet', 'CNN'], font=fixed_font)
choose_net_combobox.grid(row=1, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_net_combobox.current(0)

choose_lr_label = tk.Label(train_frame, text='Learning rateL):', font=fixed_font)
choose_lr_label.grid(row=3, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_lr_combobox = ttk.Combobox(train_frame, values=['0.0001', '0.001', '0.01', '0.1'], font=fixed_font)
choose_lr_combobox.grid(row=4, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_lr_combobox.current(3)

choose_epochs_label = tk.Label(train_frame, text='Epochs:', font=fixed_font)
choose_epochs_label.grid(row=6, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_epochs_combobox = ttk.Combobox(train_frame, values=['50', '100', '200', '300','2','3'], font=fixed_font)
choose_epochs_combobox.grid(row=7, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_epochs_combobox.current(4)

choose_batch_size_label = tk.Label(train_frame, text='Batch size:', font=fixed_font)
choose_batch_size_label.grid(row=9, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_batch_size_combobox = ttk.Combobox(train_frame, values=['4', '8', '16', '32'], font=fixed_font)
choose_batch_size_combobox.grid(row=10, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_batch_size_combobox.current(1)

choose_loss_fun_label = tk.Label(train_frame, text='Loss function:', font=fixed_font)
choose_loss_fun_label.grid(row=12, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_loss_fun_combobox = ttk.Combobox(train_frame, values=['Cross Entropy', 'Mean Squared Error'], font=fixed_font)
choose_loss_fun_combobox.grid(row=13, column=0,columnspan=2 ,padx=10, pady=10, sticky='w')
choose_loss_fun_combobox.current(0)

start_button = tk.Button(train_frame, text='Start training', command=start_training, font=fixed_font)
start_button.grid(row=14, column=0 ,padx=10, pady=10, sticky='w')

# stop_button = tk.Button(train_frame, text='End training', command=stop_training, font=fixed_font)
# stop_button.grid(row=14, column=1 ,padx=10, pady=10, sticky='w')

# 训练过程可视化
image_init2 = Image.new("RGB", (225, 225), "white")
photo_init2 = ImageTk.PhotoImage(image_init2)
l_e_label = tk.Label(train_frame, text='loss-epochs:', font=fixed_font)
l_e_label.grid(row=0, column=2 ,padx=10, pady=10, sticky='w')
l_e_text = tk.Label(train_frame)
l_e_text.grid(row=1, column=2, rowspan=7, padx=10, pady=10, sticky='nsew')
l_e_text.configure(image=photo_init2)

recall_label = tk.Label(train_frame, text='recall:', font=fixed_font)
recall_label.grid(row=0, column=3 ,padx=10, pady=10, sticky='w')
recall_text = tk.Label(train_frame, height=5, width=30)
recall_text.grid(row=1, column=3, rowspan=7, padx=10, pady=10, sticky='nsew')
recall_text.configure(image=photo_init2)

precision_label = tk.Label(train_frame, text='precision:', font=fixed_font)
precision_label.grid(row=8, column=2 ,padx=10, pady=10, sticky='w')
precision_text = tk.Label(train_frame, height=5, width=30)
precision_text.grid(row=9, column=2, rowspan=6, padx=10, pady=10, sticky='nsew')
precision_text.configure(image=photo_init2)

f1_label = tk.Label(train_frame, text='f1:                        ', font=fixed_font)
f1_label.grid(row=8, column=3 ,padx=10, pady=10, sticky='w')
f1_text = tk.Label(train_frame, height=5, width=30)
f1_text.grid(row=9, column=3, rowspan=6, padx=10, pady=10, sticky='nsew')
f1_text.configure(image=photo_init2)


root.mainloop()
