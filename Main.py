import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    
    runTest()
    #runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    import pickle
    with open('pos_weight.pkl', 'rb') as f:
        pos_weight_tensor = pickle.load(f)
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/kaggle/working/all_images'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_10k.txt'
    pathFileVal = './dataset/val_2k.txt'
    pathFileTest = './dataset/test_official.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 8
    trMaxEpoch = 8
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    checkpointPath = './checkpoints/focal_epoch2_checkpoint.pth.tar'
    pathModel = './checkpoints/focal_loss_best_epoch8.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    print ('Resuming from checkpoint = ', checkpointPath)
    print ('Best checkpoint will be saved to = ', pathModel)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpointPath, pos_weight_tensor)
    
    print ('Training finished.')
    print ('Checkpoint must be pushed to GitHub before testing.')

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/kaggle/working/all_images'
    pathFileTest = './dataset/test_official.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 2
    imgtransResize = 256
    imgtransCrop = 224
    
    models = [
    './checkpoints/focal_loss_best_epoch8.pth.tar',
    './checkpoints/sqrt_weighted_bce_best.pth.tar'
]
    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





