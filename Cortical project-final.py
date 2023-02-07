#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nibabel as nib #access to medical file formats
from scipy import ndimage as ndi #for multidimensional image processing
import matplotlib.pyplot as plt #for plotting images

import numpy as np #for mathematical functions
from nilearn import plotting #provides general linear model analysis

import pylab as plt #for plotting
from nilearn import image as nli 

from skimage import morphology #skimage is for image processing
from skimage import measure

from skimage.util import montage 
from skimage.transform import rotate
from nilearn.masking import apply_mask


from skimage.segmentation import mark_boundaries


# In[2]:


#Loading dataset 
data_frame=nib.load("raw_t1_subject_02.nii")
plot_data=data_frame.get_fdata()


# In[3]:


thresh = np.percentile(plot_data,90) #compute 90th percentile of plot_data to get better regions
thresh


# In[4]:


a=plot_data.copy()
a[a>thresh]=thresh
a.max()


# In[5]:


b=plot_data.copy()
b[b<thresh]=0


# In[6]:


plt.imshow(b[:,:,80],cmap='seismic')
plt.title('Original image')


# In[7]:


b.shape


# In[8]:


#creating instance of array with evenly shaped values
for i in np.arange(b.shape[0]):
    for j in np.arange(b.shape[1]):
        for k in np.arange(b.shape[2]):
            if(b[i,j,k]>50 and b[i,j,k]<100):
                if(b[i+3,j,k]<40 and b[i-3,j,k]<40):    
                    b[i,j,k]=0
                elif(b[i,j+3,k]<40 and b[i,j-3,k]<40):
                    b[i,j,k]=0
            elif(b[i,j,k]<50):
                b[i,j,k]=0


# In[9]:


plt.imshow(b[:,:,80],cmap='seismic')


# In[11]:


erosion_i = morphology.erosion(b) #erosion removes pixels on image boundaries


# In[12]:


plt.imshow(erosion_i[:,:,80],cmap='seismic')
plt.title('Eroded image')


# In[13]:


dilation_i = morphology.dilation(erosion_i) #adds pixels to object boundaries


# In[14]:


plt.imshow(dilation_i[:,:,80],cmap='seismic')
plt.title('Dilated image')


# In[15]:


thresh_d = np.percentile(erosion_i,99)
thresh_d


# In[16]:


v=dilation_i.copy()
v[v<100]=0


# In[17]:


plt.imshow(v[:,:,80],cmap='seismic')


# In[18]:


grey = dilation_i-v


# In[19]:


plt.imshow(grey[:,:,150],cmap='seismic')


# In[20]:


grey_b = grey[np.where(grey>0)]


# In[21]:


np.min(grey_b)


# In[22]:


grey.shape


# In[23]:


b_g = grey_b[np.where(grey_b==np.min(grey_b))]


# In[24]:


b_g.shape


# In[25]:


#find gray matter boundary
grey_b=np.array([0]*256**3).reshape(256,256,256)
for i in np.arange(dilation_i.shape[0]):
    for j in np.arange(dilation_i.shape[1]):
        for k in np.arange(dilation_i.shape[2]):
            if (dilation_i[i,j,k]>0):
                if (dilation_i[i-1,j,k] == 0 or
                   dilation_i[i,j-1,k] == 0 or
                   dilation_i[i,j,k-1] == 0 or
                   dilation_i[i+1,j,k] == 0 or
                   dilation_i[i,j+1,k] == 0 or
                   dilation_i[i,j,k+1] == 0 ):
                    
                    grey_b[i,j,k] = 1;
                    
plt.imshow(grey_b[:,:,80], cmap='seismic')
plt.title('Grey matter boundaries')


# In[26]:


#find white matter boundary
white_boundary = np.array([0]*256**3).reshape(256,256,256)

for x in np.arange(grey.shape[0]):
    for y in np.arange(grey.shape[1]):
        for z in np.arange(grey.shape[2]):
            if(grey[x,y,z]>0):
                if(v[x+1,y,z]>=100 or v[x,y+1,z]>=100 or v[x,y,z+1]>=100 or v[x-1,y,z]>=100 or v[x,y-1,z]>=100 or v[x,y,z-1]>=100 or v[x+1,y+1,z+1]>=100 or v[x-1,y-1,z-1]>=100):
                    white_boundary[x,y,z]=1;
                    

from skimage.measure import label

gry = label(grey)

gry=gry-white_boundary

plt.imshow(white_boundary[:,:,130],cmap='gist_heat')


# In[27]:


from skimage import segmentation

label_img=np.array([0]*256**2).reshape(256,256)
label_img=label_img.astype(int)


# In[28]:


plt.hist(erosion_i[:,:,100],bins=5,linewidth=2.5)


# In[29]:


#Finding thickness of cerebral vortx
from scipy import ndimage
distanceThick = ndimage.distance_transform_edt(grey)
plt.imshow(distanceThick[:,:,80],cmap='seismic')
plt.title('Cortical thickness')

affine = np.eye(4)
final_file = nib.Nifti1Image(distanceThick,affine)
nib.save(final_file,"D:\Anjali\Semester 3\Masters Project\grey_thick.nii")


# In[ ]:




