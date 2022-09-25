#!/usr/bin/env python
# coding: utf-8

# ## ** <font size=3 color=>< /font> ** 更改字型大小&顏色

# In[1]:


import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow.keras as keras


# In[2]:


(xr,yr),(xt,yt) = mnist.load_data()


# ### **<font size=5 color=red>(xr=train_image，yr=train_label),(xt=test_image,yt=test_label)</font>**

# In[3]:


xr.shape, yr.shape, xt.shape, yt.shape


# In[4]:


x= xr[3]
#%%
for i in range(28):
    for j in range(28):
        z= x[i,j]
        print(f'{z:3d}', end='')
    print()


# # range(28)的28代表是由28X28的矩陣所組成。

# In[5]:


import matplotlib.pyplot as plt
plt.imshow(x,cmap="RdPu")


# # colormap色碼表:
# ## https://matplotlib.org/stable/tutorials/colors/colormaps.html
# ## https://blog.csdn.net/qq_43426078/article/details/123635851
# **<font size=5>可以選擇你要產生的圖形顏色</font>**

# In[6]:


y= yr[0]
print(f'{y= }')


# In[7]:


xr1= xr.reshape(-1, 28*28) 
xt1= xt.reshape(-1, 28*28) 


# In[8]:


aModel= keras.Sequential([
    keras.Input(28*28),
    keras.layers.Dense(100),    
    keras.layers.Dense(10)   
    ])

aModel.compile(
    loss=     'sparse_categorical_crossentropy',
    metrics= ['accuracy']
    )


# In[9]:


aModel.summary()

keras.utils.plot_model(aModel, 
    show_shapes= True, 
    show_layer_activations= True)


# # **<font color=red size=15>下面是未更改前的code</font>**

# In[14]:


aModel.fit(xr1, yr, 
           epochs= 10, 
           batch_size= 100)

aModel.evaluate(xt1, yt)


# # **<font color=red>epoch:</font>**所有data丟進Neural Network裡訓練一次，就等於1個epoch，所以epoch=?，?代表所有data會被訓練幾次。
# # **<font color=red>batch_size:</font>**把所有data分成一堆一堆的，當batch_size=?時，?等於一次會丟多少data到Neural Network裡。batch_size=100，代表一次丟100個data到Neural Network。

# # **<font color = blue size=20>進行優化</font>**

# In[15]:


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))


# In[16]:


network.compile(
    optimizer = 'rmsprop', # 優化器→控制梯度下降的確切規則
    loss = 'categorical_crossentropy', # 損失函數loss function
    metrics = ['accuracy'] # 準確度
)


# In[24]:


train_images= xr.reshape((60000,28*28))
train_images = xr.astype('float32')/ 255

test_images = xt.reshape((10000,28*28))
test_images = xt.astype('float32')/ 255


# In[25]:


from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(yr)
test_labels = to_categorical(yt)


# In[44]:


network.fit(train_images,train_labels,epochs=20,batch_size=200)


# # 總共train了19次終於accurancy能到達100%了

# # **<font size=5 color=red>epoch下方有個300/300這個是60000/200=300的結果，因為batch_size=200，所以總共把data分成300次給Neural Network進行訓練。</font>**

# In[45]:


aModel.summary()

keras.utils.plot_model(aModel, 
    show_shapes= True, 
    show_layer_activations= True)


# # **<font color=blue size=20>測試準確度</font>**

# In[48]:


test_loss , test_acc = network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)


# In[ ]:




