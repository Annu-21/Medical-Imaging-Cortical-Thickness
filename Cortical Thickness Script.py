#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn import plotting
import pylab as plt
from nilearn import image as nli
from skimage.util import montage 
from skimage.transform import rotate
from nilearn.masking import apply_mask
from skimage import morphology
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.segmentation import mark_boundaries
from math import sqrt



# In[2]:


#Loading dataset 
data_frame=nib.load("raw_t1_subject_02.nii")
plot_data=data_frame.get_fdata()


# In[43]:


thresh = np.percentile(plot_data,90)
thresh


# In[44]:


a=plot_data.copy()
a[a>thresh]=thresh
a.max()


# In[45]:


b=plot_data.copy()
b[b<thresh]=0


# In[46]:


plt.imshow(b[:,:,80],cmap='seismic')


# In[7]:


b.shape


# In[8]:


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


# In[37]:


plt.imshow(b[:,:,80],cmap='seismic')


# In[10]:


erosion_i = morphology.erosion(b)


# In[36]:


plt.imshow(erosion_i[:,:,80],cmap='seismic')


# In[12]:


dilation_i = morphology.dilation(erosion_i)


# In[35]:


plt.imshow(dilation_i[:,:,80],cmap='seismic')


# In[14]:


thresh_d = np.percentile(erosion_i,99)
thresh_d


# In[15]:


v=dilation_i.copy()
v[v<100]=0


# In[34]:


plt.imshow(v[:,:,80],cmap='seismic')


# In[17]:


grey = dilation_i-v


# In[31]:


plt.imshow(grey[:,:,150],cmap='seismic')


# In[19]:


grey_b = grey[np.where(grey>0)]


# In[20]:


np.min(grey_b)


# In[21]:


grey.shape


# In[22]:


b_g = grey_b[np.where(grey_b==np.min(grey_b))]


# In[23]:


b_g.shape


# In[24]:


from skimage import segmentation

label_img=np.array([0]*256**2).reshape(256,256)
label_img=label_img.astype(int)


# In[51]:


plt.hist(erosion_i[:,:,100],bins=5,linewidth=2.5)


# In[29]:


affine = np.eye(4)
final_file = nib.Nifti1Image(distanceThick,affine)
nib.save(final_file,"D:\Anjali\Semester 3\Masters Project\grey_thick.nii")

from scipy import ndimage
distanceThick = ndimage.distance_transform_edt(grey)
plt.imshow(distanceThick[:,:,80],cmap='seismic')


# In[30]:





# In[ ]:




