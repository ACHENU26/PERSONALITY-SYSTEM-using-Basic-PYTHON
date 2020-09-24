#!/usr/bin/env python
# coding: utf-8

# # PERSONALITY SYSTEM with KMEANS

# In[95]:


import pandas as pd

df = pd.read_csv('data-final.csv', delimiter="\t")


# In[96]:


df


# In[97]:


columns = df.columns


# In[98]:


for column in columns:
    print(column)

#*** *** ***
# In[99]:


import numpy as np


# In[100]:


X = df[df.columns[0:50]]


# In[101]:


pd.set_option('display.max_columns',None)


# In[102]:


X


# In[103]:


X = X.fillna(0)


# In[104]:


from sklearn.cluster import MiniBatchKMeans


# In[105]:


kmeans = MiniBatchKMeans(n_clusters=12, random_state=0, batch_size=100, max_iter=100).fit(X)

#*** *** ***
# In[106]:


len(kmeans.cluster_centers_)


# In[107]:


one = kmeans.cluster_centers_[0]


# In[108]:


two = kmeans.cluster_centers_[1]


# In[109]:


three = kmeans.cluster_centers_[2]


# In[110]:


four = kmeans.cluster_centers_[3]


# In[111]:


five =kmeans.cluster_centers_[4]


# In[112]:


six = kmeans.cluster_centers_[5]


# In[113]:


seven = kmeans.cluster_centers_[6]


# In[114]:


eight = kmeans.cluster_centers_[7]


# In[115]:


nine= kmeans.cluster_centers_[8]


# In[116]:


ten = kmeans.cluster_centers_[9]


# In[117]:


eleven = kmeans.cluster_centers_[10]


# In[118]:


twelve = kmeans.cluster_centers_[11]


# In[119]:


one

#*** *** ***
# In[120]:


one_scores = {}

one_scores['extroversion_score'] =  one[0] - one[1] + one[2] - one[3] + one[4] - one[5] + one[6] - one[7] + one[8] - one[9]
one_scores['neuroticism_score'] =  one[0] - one[1] + one[2] - one[3] + one[4] + one[5] + one[6] + one[7] + one[8] + one[9]
one_scores['agreeableness_score'] =  - one[0] + one[1] - one[2] + one[3] - one[4] - one[5] + one[6] - one[7] + one[8] + one[9]
one_scores['conscientiousness_score'] =  one[0] - one[1] + one[2] - one[3] + one[4] - one[5] + one[6] - one[7] + one[8] + one[9]
one_scores['openness_score'] =  one[0] - one[1] + one[2] - one[3] + one[4] - one[5] + one[6] + one[7] + one[8] + one[9]


# In[121]:


one_scores

#*** *** ***
# In[166]:


all_types = {'one': one, 'two': two, 'three' : three, 'four': four, 'five': five, 'six': six, 'seven': seven, 'eight': eight,
             'nine': nine, 'ten': ten, 'eleven': eleven, 'twelve' : twelve}

all_types_scores ={}

for name, personality_type in all_types.items():
    personality_trait = {}

    personality_trait['extroversion_score'] =  personality_type[0] - personality_type[1] + personality_type[2] - personality_type[3] + personality_type[4] - personality_type[5] + personality_type[6] - personality_type[7] + personality_type[8] - personality_type[9]+ personality_type[10] + personality_type[11]
    personality_trait['neuroticism_score'] =  personality_type[0] - personality_type[1] + personality_type[2] - personality_type[3] + personality_type[4] + personality_type[5] + personality_type[6] + personality_type[7] + personality_type[8] + personality_type[9] + personality_type[10] + personality_type[11]
    personality_trait['agreeableness_score'] =  -personality_type[0] +personality_type[1] - personality_type[2] + personality_type[3] - personality_type[4] - personality_type[5] + personality_type[6] - personality_type[7] + personality_type[8] + personality_type[9] + personality_type[10] + personality_type[11]
    personality_trait['conscientiousness_score'] = personality_type[0] - personality_type[1] + personality_type[2] -personality_type[3] +personality_type[4] - personality_type[5] +personality_type[6] -personality_type[7] + personality_type[8] + personality_type[9] + personality_type[10] + personality_type[11]
    personality_trait['openness_score'] =  personality_type[0] -personality_type[1] + personality_type[2] - personality_type[3] + personality_type[4] - personality_type[5] +personality_type[6] + personality_type[7] + personality_type[8] + personality_type[9] + personality_type[10] + personality_type[11]
    
    all_types_scores[name] = personality_trait


# In[167]:


all_types_scores

#*** *** ***
# In[168]:


all_extroversion = []
all_neuroticism =[]
all_agreeableness =[]
all_conscientiousness =[]
all_openness =[]


# In[169]:


for personality_type, personality_trait in all_types_scores.items():
    all_extroversion.append(personality_trait['extroversion_score'])
    all_neuroticism.append(personality_trait['neuroticism_score'])
    all_agreeableness.append(personality_trait['agreeableness_score'])
    all_conscientiousness.append(personality_trait['conscientiousness_score'])
    all_openness.append(personality_trait['openness_score'])


# In[170]:


all_extroversion_normalized = (all_extroversion-min(all_extroversion))/(max(all_extroversion)-min(all_extroversion))
all_neuroticism_normalized = (all_neuroticism-min(all_neuroticism))/(max(all_neuroticism)-min(all_neuroticism))
all_agreeableness_normalized = (all_agreeableness-min(all_agreeableness))/(max(all_agreeableness)-min(all_agreeableness))
all_conscientiousness_normalized = (all_conscientiousness-min(all_conscientiousness))/(max(all_conscientiousness)-min(all_conscientiousness))
all_openness_normalized = (all_openness-min(all_openness))/(max(all_openness)-min(all_openness))


# In[171]:


all_extroversion_normalized


# In[172]:


counter = 0

normalized_all_types_scores ={}

for personality_type, personality_trait in all_types_scores.items():
    normalized_personality_trait ={}
    normalized_personality_trait['extroversion_score'] = all_extroversion_normalized[counter]
    normalized_personality_trait['neuroticism_score'] = all_neuroticism_normalized[counter]
    normalized_personality_trait['agreeableness_score'] = all_agreeableness_normalized[counter]
    normalized_personality_trait['conscientiousness_score'] = all_conscientiousness_normalized[counter]
    normalized_personality_trait['openness_score'] = all_openness_normalized[counter]
    
    normalized_all_types_scores[personality_type] = normalized_personality_trait
    
    counter+=1


# In[173]:


normalized_all_types_scores

#*** *** ***
# In[174]:


import numpy as np
import matplotlib.pyplot as plt


# In[175]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['one'].keys()), normalized_all_types_scores['one'].values(), color='b')
plt.show()


# In[176]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['two'].keys()), normalized_all_types_scores['two'].values(), color='b')
plt.show()


# In[177]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['three'].keys()), normalized_all_types_scores['three'].values(), color='b')
plt.show()


# In[178]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['four'].keys()), normalized_all_types_scores['four'].values(), color='b')
plt.show()


# In[179]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['five'].keys()), normalized_all_types_scores['five'].values(), color='b')
plt.show()


# In[180]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['six'].keys()), normalized_all_types_scores['six'].values(), color='b')
plt.show()


# In[181]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['seven'].keys()), normalized_all_types_scores['seven'].values(), color='b')
plt.show()


# In[182]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['eight'].keys()), normalized_all_types_scores['eight'].values(), color='b')
plt.show()


# In[183]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['nine'].keys()), normalized_all_types_scores['nine'].values(), color='b')
plt.show()


# In[184]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['ten'].keys()), normalized_all_types_scores['ten'].values(), color='b')
plt.show()


# In[185]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['eleven'].keys()), normalized_all_types_scores['eleven'].values(), color='b')
plt.show()


# In[186]:


plt.figure(figsize=(15,10))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['twelve'].keys()), normalized_all_types_scores['twelve'].values(), color='b')
plt.show()


# In[ ]:




