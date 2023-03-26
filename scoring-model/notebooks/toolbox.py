import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

from scipy.stats import chi2_contingency
from scipy.stats import shapiro
import scipy.stats as st

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
from matplotlib.collections import LineCollection
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import (preprocessing,
                     manifold,
                     decomposition)
from sklearn.metrics import roc_auc_score, fbeta_score, roc_curve, confusion_matrix
import time
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import time
import plotly.graph_objs as go



import string as st

from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt

import matplotlib.cm as cm

from contextlib import contextmanager
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, StratifiedKFold


# Cette fonction a pour objectif d'ouvrir un fichier en onction de son extension et de le return sous forme de dataframe
def read_data(file_extension, path):
    if file_extension == 'xlsx':
        data = pd.read_excel(path, engine='openpyxl')
    elif file_extension == 'xls':
        data = pd.read_excel(path)
    elif file_extension == 'csv':
        data = pd.read_csv(path)           
    return data


# Cette fonction a pour objectif d'afficher un aperçu et une description d'un dataframe ainsi que le nbre de missing values qu'il contient
def describe_data(df, figsize=(6,4)):
    #print('*'*35,'Data infos','*'*35)
 
    #Check nombre de colonnes
    print("Nombre de colonnes : ",df.shape[1],"\n")

    #Check nombre de lignes
    print("Nombre de lignes : ",df.shape[0],"\n")
    
    # Analyse des valeurs manquantes
    plt.figure(figsize=(9,6))
    #print('*'*34,"Valeurs manquantes",'*'*34)
    all_df = df.isnull().sum().sum(), df.notnull().sum().sum()
    plt.pie(all_df, autopct='%1.1f%%', shadow=False, startangle=90,labels=['Missing values', 'Not missing values'], explode = (0, 0.02), colors=["lightblue","steelblue"], pctdistance=0.4, labeldistance=1.1)
    circle = plt.Circle( (0,0), 0.65, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.show()
    
    print("Nombre total de valeurs manquantes : ",df.isna().sum().sum(),'\n')

def cles_potentielles(df, max_allowed=10):
    from itertools import chain, combinations
    combi_list = chain.from_iterable( combinations(list(df), x) for x in range(1, len(list(df))+1) )
    found = 0
    for candidate in combi_list:
        tmp = df.drop_duplicates(candidate)
        if len(tmp) == len(df):
            print( list(candidate) )
            found +=1
        if found > max_allowed:
            print( 'Nombre maximum autorisé atteint.', end=' ')
            print( 'Veuillez augmenter cette valeur si vous voulez rechercher davantage de clés primaires candidates.' )
            return
    if found == 0:
        print('''Aucune clé primaire, simple ou composée, n'a pu être trouvée ! Il y a forcément des doublons.''')
    
        
def infos_columns(df):
    print('*'*26,"Nombre de valeurs uniques par colonne", '*'*26,'\n')
    for column in list(df):
        print(column, " : ",len(df[column].unique()),'\n')
        
              
def check_outliers(data, threshold = 2):
    outliers=[]
    mean = np.mean(data)
    std = np.std(data)
    
    for i in data:
        z_score = (i- mean)/std
        if z_score > threshold:
            outliers.append(i)
    print("Le nombre d'outliers détectés  " + str(len(outliers)))
    return outliers


def stats(data, region_col, indicators):
    # On parcourt les regions
    for region in data[region_col].unique():
        # On initialise un dict avec la colonne qui indique les indicateurs statistiques à calculer
        stats = {'Indicateur statistique':['mean','median','std','mode', 'kurtosis']}
        # On parcourt les indicateurs pertinents
        for indicator in indicators:
            # On calcule les stats 
            mean = data[data[region_col]==region][indicator].mean()
            median = data[data[region_col]==region][indicator].median()
            mode = data[data[region_col]==region][indicator].mode()[0]
            std = data[data[region_col]==region][indicator].std()
            kurtosis = data[data[region_col]==region][indicator].kurt()
            # On met à jour le dictionnaire avec les 
            stats.update({indicator:[mean,median,std,mode,kurtosis]})
        stats2=pd.DataFrame(stats)
        print("\n","*"*75,region,"*"*75)
        display(stats2.round(2))
        

palette = sns.color_palette("bright", 10)

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            #fig, ax = plt.subplots(figsize=(8,8))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # Initialise the matplotlib figure      
            #fig = plt.figure(figsize=(7,6))
        
            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
   
def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    plt.show(block=False)

def append_class(df, class_name, feature, thresholds, names):
    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''
    
    n = pd.cut(df[feature], bins = thresholds, labels=names)
    df[class_name] = n

def plot_dendrogram(Z, names):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    #plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    #plt.show()

def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    
    return (colour[0],colour[1],colour[2],alpha)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        


def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)
        
def split_words(df, column = None):
  list_words = set()
  for word in df[column].str.split(','):
    if isinstance(word, float):
      continue
    list_words = set().union(word, list_words)
  return list(list_words)




def clustering_eval(preprocessor, model, data, metric, elbow=True, mds=False, sil=False, KBest=None):
    
    if((elbow==True) & (mds==True)):
        ncols=3
    elif((elbow==False) | (mds==False)):
        ncols=2
    else:
        ncols=1
        
    fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=False, sharey=False, figsize=(24,8))
    
    ax=0
    if(elbow==True):
        # Elbow visualizer
        kmeans_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("kelbowvisualizer", KElbowVisualizer(model,K=(4,12), metric=metric, ax=axes[ax]))])
        kmeans_visualizer.fit(data)
        KBest = kmeans_visualizer.named_steps['kelbowvisualizer'].elbow_value_
        kmeans_visualizer.named_steps['kelbowvisualizer'].finalize()
        ax+=1
    
    # Set best K
    K = KBest
    model.set_params(n_clusters=K)

    # Silhouette Visualizer
    if(sil==True):
        silhouette_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("silhouettevisualizer", SilhouetteVisualizer(model, ax=axes[ax]))])
        silhouette_visualizer.fit(data)
        silhouette_visualizer.named_steps['silhouettevisualizer'].finalize()
        ax+=1
    
    # Intercluster distance Map with best k
    if(mds==True):
        distance_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("distancevisualizer", InterclusterDistance(model, ax=axes[ax]))])
        distance_visualizer.fit(data)
        distance_visualizer.named_steps['distancevisualizer'].finalize()
    
    return K



def plot_radars(data, group):

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), 
                        index=data.index,
                        columns=data.columns).reset_index()
    
    fig = go.Figure()

    for k in data[group]:
        fig.add_trace(go.Scatterpolar(
            r=data[data[group]==k].iloc[:,1:].values.reshape(-1),
            theta=data.columns[1:],
            fill='toself',
            name='Cluster '+str(k)
        ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
        showlegend=True,
        title={
            'text': "<b>Projection des moyennes par variable des clusters</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="black",
        title_font_size=26)

    fig.show()
    
    
    
def plotTSNE(X_scaled, kmeans_labels):
    '''
        For each given algorithm :
        - fit them to the data
        - Calculate the mean silhouette
        - Gets the calculation time
        The function then plots the identified clusters for each algorithm.
        Parameters
        ----------------
        - algorithm : dictionary with
                        - name and type of input as keys
                        - instantiated algorithm as values
        - data     : pandas dataframe
                     Contains the data to fit the algo on
                     
        - dftnse   : pandas dataframe
                     Contains 2D tsne data
        - long     : int
                     length of the plot figure
        - larg     : int
                     width of the plot figure
        - title    : string
                     title of the plot figure
        Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient, the number of clusters,
                      the calculation time of the algorithm
    '''


    TITLE_SIZE = 40
    TITLE_PAD = 1.05
    LABEL_SIZE = 30
    LABEL_PAD = 20
    
    tsne = manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X_scaled)

    data_to_plot = pd.DataFrame()
    
    data_to_plot["tsne-2d-one"] = tsne_results[:,0]
    data_to_plot["tsne-2d-two"] = tsne_results[:,1]
    data_to_plot["kmeans_label"] = kmeans_labels
    
    fig = plt.figure(figsize=(10, 10))
    
    plt.title("Mise en évidence des clusters t-SNE", fontsize=TITLE_SIZE)

    handle_plot_2 = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                                    data=data_to_plot,
                                    hue="kmeans_label",
                                    palette=sns\
                                      .color_palette("hls",
                                                     data_to_plot["kmeans_label"].nunique()),
                                    legend="full")

    handle_plot_2.set_xlabel("t-SNE 1",
                             fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD,
                             fontweight="bold")

    handle_plot_2.set_ylabel("t-SNE 2",
                             fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD,
                             fontweight="bold")    
#------------------------------------------

def plotSilhouetteVisualizer(data, range_n_clusters):
    
    X = data

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

#------------------------------------------


def getScores(algorithm, data):
    '''
        For each given algorithm :
        - fit them to the data
        - Calculate the mean silhouette
        - Gets the calculation time
        The function then plots the identified clusters for each algorithm.
        Parameters
        ----------------
        - algorithm : dictionary with
                        - name and type of input as keys
                        - instantiated algorithm as values
        - data     : pandas dataframe
                     Contains the data to fit the algo on
                     
         Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient, the number of clusters,
                      the calculation time of the algorithm
    '''

    scores_time = pd.DataFrame(columns=["Algorithme", "silhouette", "Nb Clusters", "Time"])

    for algoname, algo in algorithm.items():
        start_time = time.time()
        cluster_labels = algo.fit_predict(data)
        elapsed_time = time.time() - start_time

        scores_time.loc[len(scores_time)] = [algoname, silhouette_score(data,
                                                                          cluster_labels,
                                                                          metric="euclidean"),
                                                         len(set(cluster_labels)),
                                                         elapsed_time]
    
    return scores_time    
    




def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.
        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : the dataframe for the data
    
        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    nan_percent = []
    duplicate_percent = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data))
        files_nb_columns.append(len(file_data.columns))
        nan_percent.append(round(file_data.isna().sum().sum()/file_data.size*100, 2))
        duplicate_percent.append(round(file_data.duplicated().sum().sum()/file_data.size*100, 2))

                           
    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    '%NaN' :nan_percent,
                                    '%Duplicate' :duplicate_percent})

    presentation_df.index += 1

    return presentation_df

#------------------------------------------

def missingdata(data, name, larg, long):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(larg,long))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"])
    plt.xlabel('Colonnes', fontsize=15)
    plt.ylabel('% valeurs manquantes', fontsize=15)
    plt.title('Pourcentage de valeurs manquantes ('+name+')', fontsize=22, fontweight='bold')
    #ms= ms[ms["Percent"] > 0]
    #return ms

#------------------------------------------

# function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# This function will create two subplots: 
# 1. Count plot of categorical column w.r.t TARGET; 
# 2. Percentage of defaulters within column

def univariate_categorical(applicationDF,feature,titre,ylog=False,label_rotation=False,
                           horizontal_layout=True):
    temp = applicationDF[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))
        
    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1, 
                    x = feature, 
                    data=applicationDF,
                    hue ="TARGET",
                    order=cat_perc[feature],
                    palette=['g','r'])
        
    # Define common styling
    ax1.set_title(titre, fontdict={'fontsize' : 15, 'fontweight' : 'bold'}) 
    ax1.legend(['Remboursé','Défaillant'])
    
    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15, 'fontweight' : 'bold'})   
    
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, 
                    x = feature, 
                    y='TARGET', 
                    order=cat_perc[feature], 
                    data=cat_perc,
                    palette='Set2')
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(titre+" (% Défaillants)", fontdict={'fontsize' : 15, 'fontweight' : 'bold'}) 

    plt.show();

#------------------------------------------

def plot_distribution(applicationDF,feature, title):
    plt.figure(figsize = (10, 4))

    t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
    t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

    
    sns.kdeplot(t0[feature].dropna(), label = 'Remboursé', color='g')
    sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
    plt.title(title, fontsize='20', fontweight='bold')
    #plt.ylabel('Density',fontsize='14')
    #plt.xlabel(fontsize='14')
    plt.legend()
    plt.show()   
    
#------------------------------------------

#------------------------------------------
# PREPROCESSING AND FEATURES INGENEERING
#------------------------------------------

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
#------------------------------------------    
    
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
#------------------------------------------

# Preprocess application_train.csv and application_test.csv
def application_train_test(PATH, num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv(PATH+'/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv(PATH+'/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
   # gc.collect()
    return df

#------------------------------------------

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(PATH, num_rows = None, nan_as_category = True):
    bureau = pd.read_csv(PATH+'/bureau.csv', nrows = num_rows)
    bb = pd.read_csv(PATH+'/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
   # gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
   # gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    #gc.collect()
    return bureau_agg

#------------------------------------------

# Preprocess previous_applications.csv
def previous_applications(PATH, num_rows = None, nan_as_category = True):
    prev = pd.read_csv(PATH+'/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    #gc.collect()
    return prev_agg

#------------------------------------------

# Preprocess POS_CASH_balance.csv
def pos_cash(PATH, num_rows = None, nan_as_category = True):
    pos = pd.read_csv(PATH+'/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
   # gc.collect()
    return pos_agg

#------------------------------------------

# Preprocess installments_payments.csv
def installments_payments(PATH, num_rows = None, nan_as_category = True):
    ins = pd.read_csv(PATH+'/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    #gc.collect()
    return ins_agg

#------------------------------------------

# Preprocess credit_card_balance.csv
def credit_card_balance(PATH, num_rows = None, nan_as_category = True):
    cc = pd.read_csv(PATH+'/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    #gc.collect()
    return cc_agg

#------------------------------------------

#------------------------------------------
# MODELISATION
#------------------------------------------

def evaluate_model(gs, x, y, x_test, y_test, model_name, balancing_method):
    
    #Entrainement
    start = time.time()
    model = gs.fit(x,y)
    end = time.time()-start

    if (model_name != 'Baseline'):
        df_results = pd.DataFrame.from_dict(model.cv_results_)

    #Training Performance
    if (model_name == 'Baseline'):
        #y_pred = model.predict(x)
        y_proba = model.predict_proba(x)

        auc_train = round(roc_auc_score(y, y_proba[:,1]),3) 
        #f2_train = round(fbeta_score(y, y_pred, beta=2), 3)
    else:
        auc_train = round(model.best_score_,3) 
        #f2_train = round(np.mean(df_results[df_results.rank_test_F2 == 1]['mean_test_F2']),3)

    #Testing Performance
    #y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    auc_test = round(roc_auc_score(y_test, y_proba[:,1]),3) 
    #f2_test = round(fbeta_score(y_test, y_pred, beta=2), 3)

    row = [model_name, 
            balancing_method,
            auc_train, 
            auc_test,
            #f2_train,
            #f2_test,
            end]

    return row

#------------------------------------------

def evaluate_model2(model, x, y, x_test, y_test, folds=5, loss_func=None):
    '''
        Uses cross-validation to determine the score of a model 
        on train data, then calculates the score on test data.
    
        Parameters
        --------
        - model     : a machine learning model
        - x         : pandas dataframe
                      The training features
        - y         : pandas dataframe
                      The training labels
        - x_test    : pandas dataframe
                      The test features
        - y_test    : pandas dataframe
                      The test labels
        - scoring   : Cost function
                      The cost function to use for scoring
        - folds     : int
                      The number of folds to use for the cross-validation
        - loss_func : Loss function
                      The loss function to use for the algorithms that allow
                      custom loss functions
            
        Returns
        --------
        -, -, -, - : tuple
                     - The training custom scores for each fold (array)
                     - The custom score for the test data (float)
                     - The training ROC AUC scores for each fold (array)
                     - The ROC AUC score for the test data (float)
    '''

    #cv_custom_scores = []
    cv_auc_scores = []
    cv_f2_scores = []

    y_pred_proba = []

    # create folds
    kf = StratifiedKFold(n_splits=folds)
    
    for train_indices, valid_indices in kf.split(x, y):
        # Training data for the fold
        xtrn, ytrn = x.iloc[train_indices], y.iloc[train_indices]
        # Validation data for the fold
        xval, yval = x.iloc[valid_indices], y.iloc[valid_indices]

        # train
        if loss_func!=None:
            model.fit(xtrn, ytrn, eval_metric = loss_func)
        else:
            model.fit(xtrn, ytrn)

        # predict values on validation set
        ypred = model.predict(xval)
        
        # save probabilities for class 1
        yprob = model.predict_proba(xval)
        y_pred_proba+=(list(yprob[:,1]))

        # calculate and save scores
        auc_score = round(roc_auc_score(yval, ypred), 3)
        cv_auc_scores.append(auc_score)

        f2_score = round(fbeta_score(yval, ypred, beta=2), 3)
        cv_f2_scores.append(f2_score)
        
        #custom_score = round(scoring(yval, ypred), 3)
        #cv_custom_scores.append(custom_score)

    if loss_func!=None:
        model.fit(x, y, eval_metric=loss_func)
        y_pred = model.predict(x_test)
    else:
        model.fit(x, y)
        y_pred = model.predict(x_test)

    auc_score_test = round(roc_auc_score(y_test, y_pred), 3)
    
    f2_score_test = round(fbeta_score(y_test, y_pred, beta=2), 3)

    #custom_score_test = round(scoring(y_test, y_pred), 3)

    return np.array(cv_f2_scores), \
           f2_score_test, \
           np.array(cv_auc_scores), \
           auc_score_test

#------------------------------------------

def plotComparaisonResults(metrics_compare, metric):
    
    fig, ax = plt.subplots()
    
    # create data
    x = np.arange(4)
    y1 = metrics_compare [metrics_compare['Balancing_method'] == "Undersampling"] [metric]
    y2 = metrics_compare [metrics_compare['Balancing_method'] == "Oversampling"] [metric]
    y3 = metrics_compare [metrics_compare['Balancing_method'] == "Balanced"] [metric]
    width = 0.2

    # plot data in grouped manner of bar type
    b1 = plt.bar(x-0.2, y1, width)
    b2 = plt.bar(x, y2, width)
    b3 = plt.bar(x+0.2, y3, width)
    plt.xticks(x, ['Baseline','LinearRegression', 'RandomForest', 'LGBM'])
    
    #if (metric =="F2"):
    #    plt.title('F2-score des modèles (train)')
    
    #if(metric =="F2_test"):
    #    plt.title('F2-score des modèles (test)')

    if (metric =="AUC"):
        plt.title('AUC des modèles (train)')
    
    if(metric =="AUC_test"):
        plt.title('AUC des modèles (test)')
        
    #if (metric =="F2" or metric =="F2_test"):    
     #   plt.ylabel("F2-score")
    if (metric =="Time"):
        plt.ylabel("Time (sec)")
        plt.title("Temps d'exécution du fit")    
    else:
        plt.ylabel("AUC score")
    plt.legend(["Undersampling", "Oversampling", "Balanced"], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
#------------------------------------------


def cf_matrix_roc_auc(model, y_true, y_pred, y_pred_proba, roc_auc, title):
    '''This function will make a pretty plot of 
  an sklearn Confusion Matrix using a Seaborn heatmap visualization + ROC Curve.'''
    fig = plt.figure(figsize=(20,15))
  
    plt.subplot(221)
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

    plt.subplot(222)
    fpr,tpr,_ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='orange', linewidth=5, label='AUC = %0.4f' %roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    fig.suptitle(title, fontsize="30", fontweight="bold")
    plt.show()
    
    
    