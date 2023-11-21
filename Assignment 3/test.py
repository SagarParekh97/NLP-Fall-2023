import os
import torch
import matplotlib.pyplot as plt



load = all([os.path.exists(f'save_data/{f}') for f in ['model_checkpoint.pth', 'training_loss.json', 'training_accuracy.json',
                                                       'validation_loss.json', 'validation_accuracy.json']])
resume = False

if resume:
    training_loss, training_accuracy, validation_loss, validation_accuracy = trainer(model, optimizer, train_dataset_loader, val_dataset_loader, resume, load)
elif load:
    fname = os.path.join(f'./save_data/model_checkpoint.pth')
    model_checkpoint = torch.load(fname)
    training_loss = model_checkpoint['training_loss']
    training_accuracy = model_checkpoint['training_accuracy']
    validation_loss = model_checkpoint['validation_loss']
    validation_accuracy = model_checkpoint['validation_accuracy']
    # with open(fname, 'r') as f:
    #     training_loss = json.load(f)
    # fname = os.path.join(f'./save_data/training_accuracy.json')
    # with open(fname, 'r') as f:
    #     training_accuracy = json.load(f)
    # fname = os.path.join(f'./save_data/validation_loss.json')
    # with open(fname, 'r') as f:
    #     validation_loss = json.load(f)
    # fname = os.path.join(f'./save_data/validation_accuracy.json')
    # with open(fname, 'r') as f:
    #     validation_accuracy = json.load(f)

else:
    os.makedirs('save_data', exist_ok=True)
    training_loss, training_accuracy, validation_loss, validation_accuracy = trainer(model, optimizer, train_dataset_loader, val_dataset_loader)

    # fname = os.path.join(f'./save_data/training_loss.json')
    # with open(fname, 'w') as f:
    #     json.dump(training_loss, f)
    # fname = os.path.join(f'./save_data/training_accuracy.json')
    # with open(fname, 'w') as f:
    #     json.dump(training_accuracy, f)
    # fname = os.path.join(f'./save_data/validation_loss.json')
    # with open(fname, 'w') as f:
    #     json.dump(validation_loss, f)
    # fname = os.path.join(f'./save_data/validation_accuracy.json')
    # with open(fname, 'w') as f:
    #     json.dump(validation_accuracy, f)

_, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].plot(training_loss, marker='.')
ax[0, 0].set_title('Training Loss')
ax[0, 1].plot(*zip(*training_accuracy))
ax[0, 1].set_title('Training Accuracy')
ax[1, 0].plot(validation_loss, marker='.')
ax[1, 0].set_title('Validation Loss')
ax[1, 1].plot(*zip(*validation_accuracy))
ax[1, 1].set_title('Validation Accuracy')

ax[0, 1].set_ylim([0, 1])
ax[1, 1].set_ylim([0, 1])
plt.show()