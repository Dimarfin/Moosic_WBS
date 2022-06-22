import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as skl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

path = '..\Data\\'

#songs=pd.read_csv(path+'df_audio_features_10', index_col=["song_name", "artist"])
songs=pd.read_csv(path+'df_audio_features_1000', index_col=["name", "artist"])

songs_slice = songs.loc[:,['danceability', 'energy', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence']]

#scaler = skl.RobustScaler()
#scaler = skl.QuantileTransformer()
scaler = skl.MinMaxScaler(feature_range=(0,1))
scaler.fit(songs_slice)
songs_scaled = scaler.transform(songs_slice)
 
songs_scaled_df = pd.DataFrame(songs_scaled,
             index=songs_slice.index,
             columns=songs_slice.columns)

my_kmeans = KMeans(n_clusters= 6)
my_kmeans.fit(songs_scaled)
clusters = my_kmeans.predict(songs_scaled)
songs["cluster"] = clusters
songs_scaled_df["cluster"] = clusters

songs_scaled_df.sort_values(by=['cluster'], inplace=True)

# plt.plot(songs_scaled_df.columns,np.array(songs_scaled_df).transpose(),'bo',alpha=0.01)

songs_scaled_df.plot(subplots=True,marker='.',linestyle='none')
plt.tight_layout()

# songs_scaled_df.hist()









