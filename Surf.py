# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:07:39 2024

@author: Hugues-Edouard
"""

#Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from yellowbrick.features import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer, InterclusterDistance

##############################################
### Data Importation
#############################################
Data = pd.read_csv("surfing_csv.csv", sep=';')
print(Data.columns.tolist())


########################################## Mise en forme des données ###################################
qualitative_columns = [
    "board_adequate", "board_nose_shape", "board_tail_shape", "board_type",
    "manoeuvres_01_paddling", "manoeuvres_02_drop", "manoeuvres_03_straight_ahead", "manoeuvres_04_wall_riding",
    "manoeuvres_05_floater","manoeuvres_06_cut_back","manoeuvres_07_carve","manoeuvres_08_off_the_lip",
    "manoeuvres_09_tube","manoeuvres_10_air","performance_control","performance_ease_paddling",
    "performance_flotation","performance_hold","performance_passing_through","performance_stability",
    "performance_surf_speed","surfer_exercise_frequency","surfer_experience","surfer_style","surfer_gender","wave_shape"
]

#Création d'un Data Frame des valeurs à encoder
df_encoding = Data[qualitative_columns]

#Création d'un Data Frame des valeurs Digitales
df_digit = Data.drop(columns=qualitative_columns) 

#Mise à part de la colonnes "wave_height" pour le moment car compliqué à 
columns_to_drop = ["wave_height","Unnamed: 0"]
df_digit = df_digit.drop(columns=columns_to_drop) 

#Gestion des valeurs des Nan






######################################## Standardisation et Encodage ###################################
#Encodage Ordinaire
encoder = OrdinalEncoder()
Data_encoder = encoder.fit_transform(df_encoding)
# Convertir le tableau NumPy en DataFrame pandas
Data_encoder_converti = pd.DataFrame(Data_encoder, columns=df_encoding.columns)


# Data standardization
# Creating the scaler instance
scaler = StandardScaler()
scaler.fit(df_digit)
df_scale = scaler.transform(df_digit) # standardise et créer le dfs standarsé les valeur
#sont centrées sur 0, l'algo fait [(variable-moyenne)/ecart-type]

# Créez un nouveau DataFrame pandas avec les données transformées
df_scaled = pd.DataFrame(df_scale, columns=df_digit.columns)



######################################################################################################
#On a maintenant nos deux tableau df_scaled et Data_encoder_converti
df_Processed = pd.concat([df_scaled, Data_encoder_converti], axis=1)



##################################################################
# Hierarchical Clustering with Agglomerative Hierarchical Clustering (AHC)
##################################################################

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



# Generate the linkage matrix
Z = linkage(df_Processed, method='ward', metric='euclidean')

##############################  Dendrogram   ########################################
# Fine-Tuning (determine the optimal number of clusters)
# Display the dendrogram
plt.title("Hierarchical Clustering Dendrogram (AHC)")
dendrogram(Z, orientation='top', color_threshold=27)
plt.show()




 



