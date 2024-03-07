# TransUnet_multisegmentation
## Train
revise'train_normal_config.txt'
```
dataset/CamVid                                      # Dataset Path
model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz   # Pre-trained model path
16                                                  # batchsize
0.001                                               # learning rate
300                                                 # epoch
trans_cam                                           # Model name (models are automatically stored in the weights folder)
```

## Dataset format：
```
dataset
├── CamVid 
│   ├── train
│   │   ├── images
│   │   ├── labels
│   ├── val
│   │   ├── images
│   │   ├── labels
│   ├── test
│   │   ├── images
        └── labels
```
