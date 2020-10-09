#!/usr/bin/env python
# coding: utf-8

# In[80]:


## Importing the libraries for the analysis.
from io import StringIO
import nltk
from nltk import tokenize
from iteration_utilities import deepflatten
import re
import pandas as pd
import spacy
nltk.download('punkt')
spacy.__version__
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[81]:


df = pd.read_csv("W://NLP qualitative/validation_corrected_05.28.2019.csv")

# df = pd.read_csv("W://NLP qualitative/location.csv")
# df = pd.read_csv("W://NLP qualitative/HFHS_corrected.csv", encoding = "ISO-8859-1", engine='python')

# location.csv
left_eye = 'corrected_left'   ## input smart text column for correction (abreviation extended)
right_eye = 'corrected_right'  ## input smart text column for correction (abreviation extended)
                 
# HFHS_corrected.csv
# left_eye = 'corrected_free_text'
# right_eye = 'corrected_free_text'
    
patid = 'PAT_ID' # For location.csv
# patid = 'Number' # For HFHS_corrected

encounter_identifier = 'DX_SOURCE_ID' # For location.csv
# encounter_identifier = 'Number' # For HFHS_corrected


## Making all the columns of the type string.
for i in df.columns:
    df[[i]] = df[[i]].astype('str')

## Description is Cancatenation of the df_lefteye and df_righteye

df['description'] = df[left_eye] + '. '+ df[right_eye] # location.csv
# df['description'] = df['corrected_free_text'] # HFHS_corrected

df.head()


# In[82]:


# Selecting only the encounter_id's,pat_mrn,left_eye,right_eye and description

df=df[[encounter_identifier,patid,left_eye,right_eye,'description']] # location.csv
# df=df[[patid,'description']] # HFHS

df.head()


# In[83]:


## Before applying any kind of transformation to the notes.
df['description'][10]


# In[84]:


## Replacing the following text with the corresponding replacements for example pericentral is replaced with paracentral
df['description'] = pd.Series(list(map(lambda x: 
                                       re.sub(r'\bpinpoint[ \w]+ defects?','PEE',x.replace('pericentral','paracentral')
                                       .replace('endothelium pigment','endopigment')
                                       .replace('limbus to limbus','centrally')
                                       .replace('entire graft','central')
                                       .replace('mid stromal','deep')
                                       .replace('midstromal', 'deep')
                                       .replace('central visual axis','visual axis')
                                       .replace('stromal loss','thin')
                                       .replace('endothelium dusting', '')
                                       .replace('seis','subepithelial infiltates')
                                       .replace('superficial punctate keratitis','spk')),
                                       df.description)),index=df.index)


# In[85]:


## The superficial punctate keratitis was replaced as spk after the above conversion.
df['description'][10]


# In[86]:


## Split_para function
# Input paramter is description file which is combination of left eye and right eye.
# The function takes in tokenizes the paragraph into sentences.
# Further the sentences were tokenized for every presence of ,|;|and|but

def split_para(paragraph):
    '''splitting paragraph into sentenses, extra splitter like ",;" "with", "and" are added'''
    try:
        sents = tokenize.sent_tokenize(paragraph)
    except TypeError: 
        print('expecting a string input')
    sents_new = list(map(lambda x:re.split("[" + ',;' + "]+", x), sents))  ## extra splitters 
    sents_new = list(deepflatten(sents_new,types=list))  ##flatten list of lists
#     sents_new = list(map(lambda x:x.split('with'), sents_new))
#     sents_new = list(deepflatten(sents_new,types=list))  ##flatten list of lists
    sents_new = list(map(lambda x:x.split(' and '), sents_new))
    sents_new = list(deepflatten(sents_new,types=list))  ##flatten list of lists
#     sents_new = list(map(lambda x:x.split('surround'), sents_new))
#     sents_new = list(deepflatten(sents_new,types=list))  ##flatten list of lists
# #     sents_new = list(map(lambda x:x.split('previous'), sents_new))
# #     sents_new = list(deepflatten(sents_new,types=list))  ##flatten list of lists
    sents_new = list(map(lambda x:x.split(' but '), sents_new))
    sents_new = list(deepflatten(sents_new,types=list))  ##flatten list of lists   
    
    return sents_new


# In[87]:


df['description'].apply(split_para)[10]


# In[88]:


## Function pos_words the function takes spacy tokenized sentences and looks for token and their childrens and their syntatic relationships
# and ptag
# Input parameters
# sentence : The spacy text with graph of children and their tags.
# token : The token presence in the sentence
# dep : The syntatic dependencies between words (adjective modifiers etc)
# ptag : Tag is part of speech tags 
# return: The tokens which matches the criteria.

def pos_words (sentence, token, dep ,ptag):
    sentences = [sent for sent in sentence.sents if token in sent.string]     
    pwrds = []
    for sent in sentences:
        for word in sent:
            if token in list(map(lambda x:str(x),list(word.children))) and (word.dep_ in dep or word.pos_ in ptag):
#                 if len(set(pos_words(sentence, word.string, dep, ptag)) & set(deepflatten(pwrds,types=list)))==0 and word.string not in pwrds:
#                     pwrds.extend(pos_words(sentence, word.string, dep, ptag))
                pwrds.append(word.text)
                if 'neovascularization' not in word.text and word.text!='edema':
                    pwrds.extend([child.string.strip() for child in word.children 
                                  if (child.dep_ in dep or child.pos_ in ptag) and 'neovascularization' not in child.text and child.text!='edema'] )
                    pwrds.extend([[child.string.strip() for child in childs.children if (child.dep_ in dep or child.pos_ in ptag) and 'neovascularization' not in child.text and child.text!='edema'] 
                                   for childs in word.children if (childs.dep_ in dep or childs.pos_ in ptag) and 'neovascularization' not in childs.text and childs.text!='edema'])
                    pwrds.extend([[[child.string.strip() for child in childs.children if (child.dep_ in dep or child.pos_ in ptag) and 'neovascularization' not in child.text and child.text!='edema'] 
                                  for childs in children.children if (children.dep_ in dep or children.pos_ in ptag) and 'neovascularization' not in childs.text and childs.text!='edema']
                                  for children in word.children if (children.dep_ in dep or children.pos_ in ptag) and 'neovascularization' not in children.text and children.text!='edema'])

            
            elif token in word.string: 
#                 pwrds.extend([pos_words(sentence,x.string,dep,ptag) for x in list(word.children) if x.string not in pwrds])
                pwrds.extend([child.string.strip() for child in word.children 
                               if (child.dep_ in dep or child.pos_ in ptag) and 'neovascularization' not in child.text and child.text!='edema'] )
                pwrds.extend([[child.string.strip() for child in childs.children if (child.dep_ in dep or child.pos_ in ptag) and 'neovascularization' not in child.text and child.text!='edema'] 
                              for childs in word.children if (childs.dep_ in dep or childs.pos_ in ptag) and 'neovascularization' not in childs.text and childs.text!='edema'])       
                pwrds.extend([[[child.string.strip() for child in childs.children if (child.dep_ in dep or child.pos_ in ptag) and 'neovascularization' not in child.text and child.text!='edema'] 
                              for childs in children.children if (children.dep_ in dep or children.pos_ in ptag) and 'neovascularization' not in childs.text and childs.text!='edema']
                              for children in word.children if (children.dep_ in dep or children.pos_ in ptag) and 'neovascularization' not in children.text and children.text!='edema'])

            
    ## Assigning to the set return only the unique words in the list without proper sorting.            
    pwrds = list(set(filter(None,deepflatten(pwrds,types=list))))
    return pwrds


# In[89]:


pos_words(nlp('epithelial defect central ; pinpoint seidel positive ? ca deposit'), 'defect', 'amod',['NOUN','ADJ','ADV'])


# In[90]:


## Cleaning the text beforehand to make sure every word within a single sentence gets tokenized as one rather than multiple. For
# example spacy considers double space as a seperate token which is why any double space were made one space.
parse = list(map(lambda x: nlp(x),list(map(lambda x: re.sub(r'\d*\.?\d+','',x.replace(' x ','')
                                  .replace("+","").replace(" mm","").replace(":","").replace("%","")).replace("  "," "), df.description.values.tolist()))))


# In[91]:


si = list(map(lambda x:pos_words(x,'infiltrate',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
si1 = list(map(lambda x:pos_words(x,'infiltrats',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

ed = list(map(lambda x:pos_words(x,'defect',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
ed1 = list(map(lambda x:pos_words(x,'defects',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

ulcer = list(map(lambda x:pos_words(x,'ulcer',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
ulcer1 = list(map(lambda x:pos_words(x,'ulcers',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

perforation = list(map(lambda x:pos_words(x,'perforation',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))


haze = list(map(lambda x:pos_words(x,'haze',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
hazy = list(map(lambda x:pos_words(x,'hazy',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

opacity = list(map(lambda x:pos_words(x,'opacity',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
opacity1 = list(map(lambda x:pos_words(x,'opacities',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
opacity2 = list(map(lambda x:pos_words(x,'opacification',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

scar = list(map(lambda x:pos_words(x,'scar',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
scar1 = list(map(lambda x:pos_words(x,'scars',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

lesion = list(map(lambda x:pos_words(x,'lesion',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
lesion1 = list(map(lambda x:pos_words(x,'lesions',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

glue = list(map(lambda x:pos_words(x,'glue',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
glue1 = list(map(lambda x:pos_words(x,'gluing',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))

limbus = list(map(lambda x:pos_words(x,'limbus',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
limbus1 = list(map(lambda x:pos_words(x,'perilimbal',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
limbus2 = list(map(lambda x:pos_words(x,'limbal',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
limbus3 = list(map(lambda x:pos_words(x,'perilimbally',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
 

periphey = list(map(lambda x:pos_words(x,'periphery',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))
periphey1 = list(map(lambda x:pos_words(x,'midperiphery',['amod','advmod','nmod','prep','pobj'],['VERB','ADJ','ADV','NOUN']),parse))


# In[92]:


ulcer_loc = [a + b + c + d + e + f for a, b, c, d, e, f in zip(si, si1, ed, ed1, ulcer, ulcer1)]
haze_loc = [a + b for a, b in zip(haze,hazy)]
scar_loc = [a + b + c + d + e + f + g + h + i + j for a, b, c, d, e, f, g, h,i,j in zip(scar, scar1, opacity, opacity1, lesion, lesion1, opacity2, glue, glue1, perforation)]
# scar_loc = [a + b + c + d + e + f + g for a, b, c, d, e, f, g in zip(scar, scar1, opacity, opacity1, lesion, lesion1, opacity2)]
limbus_loc = [a + b + c + d for a, b, c, d in zip(limbus, limbus1,limbus2,limbus3)]
periphery_loc = [a + b for a, b in zip(periphey,periphey1)]
adj = [a + b + c+d  for a, b, c,d in zip(ulcer_loc,haze_loc,scar_loc,periphery_loc)]
deep_loc = [a + b+c+d for a, b, c, d in zip(ulcer_loc, glue, glue1, perforation)]


# In[93]:


ulcer_loc


# In[94]:


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


# In[95]:


print('central','\n', list(filter(re.compile(r'^(?!para)(\w*)?centra(?!t)').match, list(set(deepflatten(adj,types=list))))))
print('quadrum', '\n',list(filter(re.compile(r'(\w*)?(inf|nasa|temp|sup)(?!ection)(?!lam)(?!iltrat)(?!ectious)(?!erficial)(?!parative)').match, list(set(deepflatten(adj,types=list))))))
print('periphery','\n',list(filter(re.compile(r'(\w*)?periphe').match, list(set(deepflatten(adj,types=list))))))
print('limbus','\n',list(filter(re.compile(r'(\w*)?limb').match,list(set(deepflatten(adj,types=list))))))


# In[96]:


r1 = re.compile(r'^(inf|nasa|temp|sup)(?!erficial)(?!ection)(?!lam)(?!iltrat)(?!ectious)(?!erficial)(?!parative)')
r2 = re.compile(r'^(inf|nasa|temp|sup).*(central|centrally)$')
## quadrant
# quadrant_wd = list(filter(re.compile(r'(\w*)?(inf|nasa|temp|sup)(?!ection)(?!lam)(?!iltrat)(?!ectious)(?!erficial)(?!parative)(?!central)').match, list(set(deepflatten(adj,types=list)))))
adjectives = list(set(deepflatten(adj,types=list)))
quadrant_wd = list(set(filter(r1.match, adjectives)) - set(filter(r2.match, adjectives)))
# #### central, paracentral and periphery


# In[97]:


paracentral = list(map(lambda x: 1 if re.compile(r'paracentral').search(" ".join(x))!=None else None, ulcer_loc))
df['ulcer_paracentral']=pd.Series(paracentral,index=df.index,dtype=str)


central = list(map(lambda x: 1 if list(filter(re.compile(r'^(?!para)(\w*)?centra(?!t)').match, x))!=[] else None,
                   ulcer_loc))
df['ulcer_central']=pd.Series(central,index=df.index,dtype=str)

periphery = list(map(lambda x: 1 if re.compile(r'(\w*)?periphe').search(" ".join(x))!=None 
                     else(1 if re.compile(r'(\w*)?limb').search(" ".join(x))!=None else None) ,
                   ulcer_loc))
df['ulcer_periphery']=pd.Series(periphery,index=df.index,dtype=str)


# In[98]:


for i in df.index:
# df[df['description'].str.contains('visual axis', case=False)].index:
    for j in split_para(df.loc[i,'description']):
        if 'into visual axis' in j and ("infiltrat" in j or "defect" in j or "ulcer" in j):
                df.loc[i,'ulcer_central']=1
        elif ('central to visual axis' in j or 'centrally to visual axis' in j) and ("infiltrat" in j or "defect" in j or "ulcer" in j):
                #df.loc[i,'ulcer_paracentral']=1
                df.loc[i,'ulcer_central']=1
        elif 'to visual axis' in j and ("infiltrat" in j or "defect" in j or "ulcer" in j):
                df.loc[i,'ulcer_paracentral']=1
                df.loc[i,'ulcer_central']=0                
        elif 'towards visual axis' in j and ("infiltrat" in j or "defect" in j or "ulcer" in j):
                df.loc[i,'ulcer_paracentral']=1
                df.loc[i,'ulcer_central']=0
        elif re.compile(
            r'(\w*)?(inf|nasa|temp|sup)(?!ection)(?!lam)(?!iltrat)(?!ectious)(?!erficial)(?!parative)(?!central).* to central').search(
            j)!=None and ("infiltrat" in j or "defect" in j or "ulcer" in j or "perforat" in j):
                df.loc[i,'ulcer_paracentral']=1
                df.loc[i,'ulcer_central']=0
        elif re.compile(r'out(\w*?)(?<!throughout) (\w* ){0,}visual axis').search(j)!=None and ("infiltrat" in j or "defect" in j or "ulcer" in j):
                df.loc[i,'ulcer_central']=0
        elif 'visual axis' in j and ("infiltrat" in j or "defect" in j or "ulcer" in j or "perforat" in j):
                df.loc[i,'ulcer_central']=1
        elif 'limb' in j and ("infiltrat" in j or "defect" in j or "ulcer" in j or "perforat" in j):
                df.loc[i,'ulcer_periphery']=1
                adj[i]=adj[i]+limbus_loc[i]
    if df.loc[i,'ulcer_central']==None and df.loc[i,'ulcer_paracentral']==None and 'axis' in ulcer_loc[i]:
        df.loc[i,'ulcer_central']=1


# In[99]:


haze_paracentral = list(map(lambda x: 1 if re.compile(r'paracentral').search(" ".join(x))!=None else None, haze_loc))
df['haze_paracentral']=pd.Series(haze_paracentral,index=df.index,dtype=str)


haze_central = list(map(lambda x: 1 if list(filter(re.compile(r'^(?!para)(\w*)?centra(?!t)').match, x))!=[] else None,
                   haze_loc))
df['haze_central']=pd.Series(haze_central,index=df.index,dtype=str)

haze_periphery = list(map(lambda x: 1 if re.compile(r'(\w*)?periphe').search(" ".join(x))!=None 
                     else(1 if re.compile(r'(\w*)?limb').search(" ".join(x))!=None else None) ,
                   haze_loc))
df['haze_periphery']=pd.Series(haze_periphery,index=df.index,dtype=str)


for i in df.index:
# df[df['description'].str.contains('visual axis', case=False)].index:
    for j in split_para(df.loc[i,'description']):
        if 'into visual axis' in j and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_central']=1
        elif ('central to visual axis' in j or 'centrally to visual axis' in j) and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_central']=1
        elif 'to visual axis' in j and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_paracentral']=1
                df.loc[i,'haze_central']=0
        elif 'towards visual axis' in j and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_paracentral']=1
                df.loc[i,'haze_central']=0
        elif re.compile(
            r'(\w*)?(inf|nasa|temp|sup)(?!ection)(?!lam)(?!iltrat)(?!ectious)(?!erficial)(?!parative).* to central').search(
            j)!=None and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_paracentral']=1
                df.loc[i,'haze_central']=0
        elif re.compile(r'out(\w*?)(?<!throughout) (\w* ){0,}visual axis').search(j)!=None and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_central']=0
        elif 'visual axis' in j and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_central']=1
        elif 'limb' in j and ("haze" in j or "hazy" in j):
                df.loc[i,'haze_periphery']=1
                adj[i]=adj[i]+limbus_loc[i]
    if df.loc[i,'haze_central']==None and df.loc[i,'haze_paracentral']==None and 'axis' in haze_loc[i]:
        df.loc[i,'haze_central']=1


# In[100]:


scar_paracentral = list(map(lambda x: 1 if re.compile(r'paracentral').search(" ".join(x))!=None else None, scar_loc))
df['scar_paracentral']=pd.Series(scar_paracentral,index=df.index,dtype=str)


scar_central = list(map(lambda x: 1 if list(filter(re.compile(r'^(?!para)(\w*)?centra(?!t)').match, x))!=[] else None,
                   scar_loc))
df['scar_central']=pd.Series(scar_central,index=df.index,dtype=str)

scar_periphery = list(map(lambda x: 1 if re.compile(r'(\w*)?periphe').search(" ".join(x))!=None 
                     else(1 if re.compile(r'(\w*)?limb').search(" ".join(x))!=None else None) ,
                   scar_loc))
df['scar_periphery']=pd.Series(scar_periphery,index=df.index,dtype=str)


for i in df.index:
# df[df['description'].str.contains('visual axis', case=False)].index:
    for j in split_para(df.loc[i,'description']):
        if 'into visual axis' in j and ("scar" in j or "opaci" in j or "lesion" in j or "glue" in j):
                df.loc[i,'scar_central']=1
                
        elif ('central to visual axis' in j or 'centrally to visual axis' in j) and ("scar" in j or "opaci" in j or "lesion" in j or "glue" in j):
                df.loc[i,'scar_central']=1
        elif 'to visual axis' in j and ("scar" in j or "opaci" in j or "lesion" in j or "glue" in j):
                df.loc[i,'scar_paracentral']=1
                df.loc[i,'scar_central']=0
        elif 'towards visual axis' in j and ("scar" in j or "opaci" in j or "lesion" in j or "glue" in j):
                df.loc[i,'scar_paracentral']=1
                df.loc[i,'scar_central']=0
        elif re.compile(
            r'(\w*)?(inf|nasa|temp|sup)(?!ection)(?!lam)(?!iltrat)(?!ectious)(?!erficial)(?!parative).* to central').search(
            j)!=None and ("scar" in j or "opaci" in j or "lesion" in j or "glue" in j):
                df.loc[i,'scar_paracentral']=1
                df.loc[i,'scar_central']=0
        elif re.compile(r'out(\w*?)(?<!throughout) (\w* ){0,}visual axis').search(j)!=None and ("scar" in j or "lesion" in j or "opaci" in j or "glue" in j):
                df.loc[i,'scar_central']=0
        elif 'visual axis' in j and ("scar" in j or "lesion" in j or "opaci" in j or "glue" in j):
                df.loc[i,'scar_central']=1
        elif 'limb' in j and ("scar" in j or "opaci" in j or "lesion" in j or "glue" in j):
                df.loc[i,'scar_periphery']=1
                adj[i]=adj[i]+limbus_loc[i]
    if df.loc[i,'scar_central']==None and df.loc[i,'scar_paracentral']==None and 'axis' in scar_loc[i]:
        df.loc[i,'scar_central']=1


# In[101]:


df['ulcer_central']=df.ulcer_central.astype(float)
df['ulcer_paracentral']=df.ulcer_paracentral.astype(float)
df['ulcer_periphery']=df.ulcer_periphery.astype(float)

df['haze_central']=df.haze_central.astype(float)
df['haze_paracentral']=df.haze_paracentral.astype(float)
df['haze_periphery']=df.haze_periphery.astype(float)

df['scar_central']=df.scar_central.astype(float)
df['scar_paracentral']=df.scar_paracentral.astype(float)
df['scar_periphery']=df.scar_periphery.astype(float)

df['central']=df.ulcer_central
df['paracentral']=df.ulcer_paracentral
df['periphery']=df.ulcer_periphery


# In[102]:


df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)| 
      (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
       'central'] = df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
    'haze_central']

df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
       'paracentral'] = df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)),
    'haze_paracentral']

df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
       'periphery'] = df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
    'haze_periphery']


# In[103]:




df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
       'central'] = df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
    'scar_central']

df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
       'paracentral'] = df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
    'scar_paracentral']

df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
       'periphery'] = df.loc[
    ~((df.central==1) | (df.paracentral==1) | (df.periphery==1)|
     (df.central==0) | (df.paracentral==0) | (df.periphery==0)),
    'scar_periphery']



# In[104]:


# In[49]:


df['flag']=pd.Series([None]*len(df))

df.loc[(df.paracentral==1) & (df.periphery==1),'flag'] = pd.Series(
    [1]*len(df[(df.paracentral==1) & (df.periphery==1)]),
    index = df[(df.paracentral==1) & (df.periphery==1)].index )

df.loc[(df.paracentral==1) & (df.central==1),'flag'] = pd.Series(
    [1]*len(df[(df.paracentral==1) & (df.central==1)]),
    index = df[(df.paracentral==1) & (df.central==1)].index )
df.loc[(df.periphery==1) & (df.central==1),'flag'] = pd.Series(
    [1]*len(df[(df.periphery==1) & (df.central==1)]),
    index = df[(df.periphery==1) & (df.central==1)].index )

print(len(df[df.flag==1]))

df.loc[(df.paracentral==1) & (df.periphery==1),'periphery'] = pd.Series(
    [None]*len(df[(df.paracentral==1) & (df.periphery==1)]),
    index = df[(df.paracentral==1) & (df.periphery==1)].index)

df.loc[(df.paracentral==1) & (df.central==1),'paracentral'] = pd.Series(
    [None]*len(df[(df.paracentral==1) & (df.central==1)]),
    index = df[(df.paracentral==1) & (df.central==1)].index) 

df.loc[(df.central==1) & (df.periphery==1),'periphery'] = pd.Series(
    [None]*len(df[(df.central==1) & (df.periphery==1)]),
    index = df[(df.central==1) & (df.periphery==1)].index) 


# In[105]:



df[(df.paracentral==1)&(df.periphery)==1]


## updates on 10.26.2018
## decided that central, paracentral and periphery should be mutually exclusive
## meaning if paracentral = 1, periphery = 0 and central = 0
#df.loc[df.central==1,'paracentral'] = pd.Series([0]*len(df[df.central == 1]), index = df[df.central==1].index)
#df.loc[df.central==1,'periphery'] = pd.Series([0]*len(df[df.central == 1]), index = df[df.central==1].index)
df.loc[df.periphery==1,'central']= pd.Series([0]*len(df[df.periphery == 1]), index = df[df.periphery==1].index)
#df.loc[df.periphery==1,'paracentral']= pd.Series([0]*len(df[df.periphery == 1]), index = df[df.periphery==1].index)
df.loc[df.paracentral==1,'central']= pd.Series([0]*len(df[df.paracentral == 1]), index = df[df.paracentral==1].index)
#df.loc[df.paracentral==1,'periphery']= pd.Series([0]*len(df[df.paracentral == 1]), index = df[df.paracentral==1].index)


# In[106]:





quadratic = list(map(lambda x: 1 if intersection(quadrant_wd, x)!=[] else None,
                   adj))

df['quadrant']=pd.Series(quadratic,index=df.index)




# In[107]:


# In[44]:

sent = list(map(lambda x: re.sub(r'(\d+)(\-)(\d+)(\%)',r'\1\4 \3\4',x),df.description))

# periphery = list(map(lambda x: 1 if re.compile(r'paracentral').search(" ".join(x))!=None else 'na', adj))
# df['paracentral']=pd.Series(paracentral,index=df.index)


# #### Multifocal


# In[108]:


df['multifocal'] = pd.Series([None]*len(df),index=df.index)


#and len(re.compile(r'defect').findall(sent[i])) > 2
#and len(re.compile(r'infiltrat').findall(sent[i])) > 2 
for i in df.index:
    x = re.sub(r'no (\w* ){0,}defects|without (\w* ){0,}defects|no (\w* ){0,}infiltrates|without (\w* ){0,}infiltrates|no (\w* ){0,}perforations|without (\w* ){0,}perforations|no (\w* ){0,}lesions|without (\w* ){0,}lesions','',df.loc[i,'description'])
    x = re.sub(r'((\d{4})|(\d+\:\d+)|(\d*\-?\d+\%)|(\d+ oclock)|(\sx\s\d*\.?\d+))', '',x)
    if "multiple" in adj[i] or "multifocal" in adj[i] or "infiltrates" in x or "defects" in x or "ulcers" in x or "perforations" in x    or "lesions" in x    or "several" in adj[i]:
        df.loc[i,'multifocal'] = 1
    elif re.compile(r'\d+ areas of( \w+){0,4} defect|\d+ areas of( \w+){0,4} infiltrat|\d+ areas of( \w+){0,4} perforation').search(df.loc[i,'description'])!=None:
        df.loc[i,'multifocal'] = 1
    elif re.compile(r'defect').search(sent[i]) != None      and len(set(re.compile(r'\d*\.?\d+\s').findall(sent[i]))) >= 2:
        df.loc[i,'multifocal'] = 1
    elif re.compile(r'infiltrat').search(sent[i])!= None     and len(set(re.compile(r'\d*\.?\d+\s').findall(sent[i]))) >= 2:
        df.loc[i,'multifocal'] = 1
    else:
        df.loc[i,'multifocal'] = 0

    if df.loc[i,'multifocal']== 1 and 'resolv' in sent[i]:
        print(i)
        df.loc[i,'multifocal'] = -1


# In[109]:


thin = []

for i in df.index:
    if 'significant thin' in sent[i]:
        thin.append(1)
    elif re.compile(r'(no|not|without)( \w+){0,2} thin').search(sent[i])!=None:
        thin.append(0)
    elif 'thin' in sent[i]:
        thin.append(1)
    elif 'perforat' in sent[i] or 'glue' in sent[i] or 'gluing' in sent[i] or 'seidel positive' in sent[i]:
        thin.append(1)
    elif re.compile(r'resolv\w+ (\w+ )*thin|thin\w+ (\w+ )*(resolv|gone)').search(sent[i])!=None:
        thin.append(-1)
    else:
        thin.append(None)
                

df['thin'] = pd.Series(thin,index=df.index)


# In[110]:



thin_percent = []
re_percent = re.compile(r'\d+\%')

for i in df.index:
    if re.compile(r'neovascularization( \w+){0,} \d+\%( \w+){0,} thin').search(sent[i])==None and     re.compile(r'\d+\%( \w+){0,} thinning( \w+){0,} neovascularization').search(sent[i])==None:
        if 'glue' in sent[i] or 'gluing' in sent[i] or 'perforat' in sent[i]:
            thin_percent.append(100)
        elif 'seidel positive' in sent[i]:
            thin_percent.append(100)
        elif re.compile(r'\d+\%( \w+){0,} thick').search(sent[i])!=None or         re.compile(r'thickness( \w+){0,} \d+\%').search(sent[i])!=None:
            thin_percent.append(100.0-min([float(x[:-1]) for x in re.findall(re_percent,sent[i])]))
        elif re.compile(r'\d+\%.*thin').search(sent[i])!=None or         re.compile(r'thin.*\d+\%').search(sent[i])!=None:
            if re.compile(r' to( approximately)?( about)? \d+\%').search(sent[i])!=None:
                thin_percent.append(100.0-min([float(x[:-1]) for x in
                        re.findall(re_percent,
                                   re.compile(r' to( approximately)?( about)? (\d+\%\s?)+').search(sent[i]).group(0))]))
            elif re.compile(r'\d+\%( \w+){0,} remain').search(sent[i])!=None:
                thin_percent.append(100.0-min([float(x[:-1]) for x in re.findall(re_percent,re.compile(r'\d+\%( [\d%\w_]+){0,} remain').search(sent[i]).group(0))]))
            elif re_percent.search(sent[i])!=None:
                thin_percent.append(max([float(x[:-1]) for x in re.findall(re_percent,sent[i])]))
            else:
                thin_percent.append(None)

        else:
            thin_percent.append(None)
    else:
        thin_percent.append(None)
                


# In[111]:


depth_percent = []

for i in df.index:
    if re.compile(r'neovascularization( \w+){0,} \d+\%( \w+){0,} depth').search(sent[i])==None and     re.compile(r'\d+\%( \w+){0,} depth( \w+){0,} neovascularization').search(sent[i])==None:
        if re.compile(r'\d+\%.*depth').search(sent[i])!=None or         re.compile(r'depth.*\d+\%').search(sent[i])!=None:
            if re.compile(r' to \d+\%').search(sent[i])!=None:
                depth_percent.append(100.0-float(re.findall(re_percent,sent[i])[0][:-1]))
            elif re.compile(r'\d+\%( \w+){0,} remain').search(sent[i])!=None:
                depth_percent.append(100.0-float(re.findall(re_percent,sent[i])[0][:-1]))
            elif re_percent.search(sent[i])!=None:
                depth_percent.append(float(re.findall(re_percent,sent[i])[0][:-1]))
            else:
                depth_percent.append(None)
        else:
            depth_percent.append(None)
    else:
        depth_percent.append(None)


# In[112]:



df['thin%(Gone)'] = pd.Series(thin_percent,index=df.index)
df['depth%'] = pd.Series(depth_percent,index=df.index)

df['thin%(Gone)'] = df['thin%(Gone)'].astype(float)
df['depth%'] = df['depth%'].astype(float)

# In[52]:


len(list(filter(None,thin_percent)))


# In[113]:


deep = []
for i in df.index:
    if 'less deep' not in df.loc[i,'description'] and 'deep' in deep_loc[i] or 'deeper' in deep_loc[i]:
        deep.append(1)
    elif 'less deep' in df.loc[i,'description']:
        deep.append(-1)
    elif df.loc[i,'thin%(Gone)']>=50.0:
        deep.append(1)
    elif df.loc[i,'thin%(Gone)']<50.0:
        deep.append(0)
    elif df.loc[i,'depth%']>=50.0:
        deep.append(1)
    elif df.loc[i,'depth%']<50.0:
        deep.append(0)
    
    elif 'glu' in df.loc[i,'description'] or 'perforat' in df.loc[i,'description'] or 'seidel positive' in sent[i]:
        deep.append(1)
    elif 'superficial' in deep_loc[i] or 'superficially' in deep_loc[i]:
        deep.append(0)
    elif re.compile(r'no (\w* ){0,}perforat|without (\w* ){0,}perforat').search(df.loc[i,'description'])!=None:
        deep.append(None)
    elif 'endothelium' in deep_loc[i]:
        deep.append(1)
    
    #elif 'superficial' in adj[i] or 'superficially' in adj[i]:
     #   deep.append(0)
    else:
        deep.append(None)


# In[114]:


df['deep']=pd.Series(deep,index=df.index)


# In[115]:



# print('# pat:',len(df.PAT_ID.unique()))
print('# visits:',len(df))
print('# central = 1:',len(df[df.central==1]))
print('# central = 0:',len(df[df.central==0]))
print('# paracentral = 1:',len(df[df.paracentral==1]))
print('# periphery = 1:',len(df[df.periphery==1]))
print('# flag:',len(df[df.flag==1]))
print('# quadrant:',len(df[df.quadrant==1]))
print('# multifocal:',len(df[df.multifocal==1]))
print('# thin %:',len(list(filter(None,thin_percent))))
print('# deep %:',len(list(filter(None,depth_percent))))
print('# deep = 1:',len(df[df.deep==1]))
print('# deep = 0:',len(df[df.deep==0]))


# In[116]:


# df.to_csv("W://NLP qualitative/output_file_20200304.csv")
df.to_csv("W://NLP qualitative/output_file_valid_um_20200814.csv")

