from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_InceptionResNetV2

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50

from keras.applications.vgg16 import VGG16 
from keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16

from keras.applications.imagenet_utils import decode_predictions

import numpy as np
import pandas as pd
from PIL import Image

def GetInception():
    model = InceptionResNetV2(weights='imagenet', include_top=True)
    for layer in model.layers:
        layer.trainable = False
    return model
    
def GetResnet():
    model = ResNet50(weights='imagenet', include_top=True)
    for layer in model.layers:
        layer.trainable = False
    return model
    
def GetVGG16():
    model = VGG16(weights='imagenet', include_top=True)
    for layer in model.layers:
        layer.trainable = False
    return model
    
def GetImage(size):
    path = '1.jpg'  
    im = Image.open(path)
    im = im.resize((size,size))
    im = np.array(im).astype(np.float32)
    # im = im.transpose((2,0,1))
    return np.expand_dims(im, axis=0)
 

    
def PredictResnet():
    im = GetImage(224)
    im = preprocess_input_ResNet50(im)
    resnet = GetResnet()
    preds = resnet.predict(im)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    '''
    Predicted: [('n02099601', 'golden_retriever', 0.42833978), ('n02099712', 'Labrador_retriever', 0.2755284), ('n02111500', 'Great_Pyrenees', 0.037853464)]
    金毛猎犬
    拉布拉多猎犬
    大比利牛斯山脉
    '''
    
def PredictInception():
    im = GetImage(299)
    im = preprocess_input_InceptionResNetV2(im)
    inception = GetInception()
    preds = inception.predict(im)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    '''
    Predicted: [('n02099712', 'Labrador_retriever', 0.44182286), ('n02099601', 'golden_retriever', 0.4193824), ('n02104029', 'kuvasz', 0.003050961)]
    拉布拉多猎犬
    金毛猎犬
    kuvasz
    '''         
    
def PredictVgg():
    im = GetImage(224)
    im = preprocess_input_VGG16(im)
    vgg = GetVGG16()
    preds = vgg.predict(im)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    '''
    Predicted: [('n02099601', 'golden_retriever', 0.5450744), ('n02091635', 'otterhound', 0.21957648), ('n02101556', 'clumber', 0.0678478)]
    金毛猎犬
    水獭猎犬
    clumber
    '''   
    
PredictVgg()