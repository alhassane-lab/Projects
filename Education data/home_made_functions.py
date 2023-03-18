import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import warnings
import folium
import missingno as msno



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
    print('*'*35,'Data infos','*'*35)
    #df.info()
    #print()
    
    #Check nombre de colonnes
    print("Nombre de colonnes : ",df.shape[1],"\n")

    #Check nombre de lignes
    print("Nombre de lignes : ",df.shape[0],"\n")
    
    # Analyse des valeurs manquantes
    plt.figure(figsize=(12,8))
    print('\n','*'*34,"Valeurs manquantes",'*'*34)
    all_df = df.isnull().sum().sum(), df.notnull().sum().sum()
    plt.pie(all_df, autopct='%1.1f%%', shadow=False, startangle=90,labels=['Missing values', 'Not missing values'], explode = (0, 0.02), colors=["lightblue","steelblue"], pctdistance=0.4, labeldistance=1.1)
    circle = plt.Circle( (0,0), 0.65, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.show()
    
    print("Nombre total de valeurs manquantes : ",df.isna().sum().sum())

    
# Cette fonction a pour ojectif de remplacer les valeurs manquantes d'un indicateur par la moyenne de ce sa région.
def fill_na_region_mean(data, region_col, indicator):
    for region in data[region_col]:
            data[indicator].fillna(data[data[region_col] == region][indicator].median(), inplace=True)
    print("Les valeurs manquates de l'indicateur {} sont remplacées par la moyenne du bloc géographique correspondant.".format(indicator))
    
    
# Cette fonction a pour ojectif de remplacer les valeurs manquantes d'un indicateur par la moyenne de ce sa région.
def fill_na_region_mean2(data, region_col, year_col, indicator):
    for year in data[year_col]:
        data = data[data[year_col]==year]
        for region in data[region_col].unique():
            mean = data[data[region_col]==region][indicator].mean()    
            data[indicator][(data[year_col]==year)&(data[region_col]==region)].fillna(mean, inplace=True)
    
# Cette fonction a pour objectif de visualiser et d'afficher les statistques d'un indicateur donnée par rapport à un bloc géographique
def univariate(df, var_list, region_col, year_col, region, year):
    palette =["steelblue","lightblue",]
    df = df[(df[region_col]==region) & (df['year_col']==year)]
    print("*"*25,'\033[1m',region, year,'\033[0m',"*"*25,"\n")
    for var in var_list:
        print("Indicateur",'\033[1m' , var,'\033[0m',"\n")
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},)
        mean=df[var].mean()
        median=df[var].median()
        mode=df[var].mode().values[0]

        sns.boxplot(data=df, x=var, ax=ax_box)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        ax_box.axvline(mode, color='b', linestyle='-')

        sns.histplot(data=df, x=var, ax=ax_hist, kde=True, bins=50)
        ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
        ax_hist.axvline(median, color='g', linestyle='-', label="Median")
        ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")

        ax_hist.legend()

        ax_box.set(xlabel='')
        plt.show()

        print ('\033[1m',"Moyenne :",'\033[0m', round(df[var].mean(), 2))
        print ('\033[1m',"Médiane :",'\033[0m', round(df[var].median(), 2))
        print ('\033[1m',"Écart-type :",'\033[0m', round(df[var].kurtosis(), 2))
        print("\n","-"*10,"\n")
        
        
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
        

        
def stats_viz(df, indicators, region_col, region):
    palette =["steelblue","lightblue",]
    print("*"*25,'\033[1m',region,'\033[0m',"*"*25)
    for var in indicators:
        print("\n","Indicateur",'\033[1m' , var,'\033[0m',"\n")
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},)
        mean=df[var].mean()
        median=df[var].median()
      
        sns.boxplot(data=df, x=var, ax=ax_box, showfliers = True)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        
        sns.histplot(data=df, x=var, bins=50, ax=ax_hist,)
        ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
        ax_hist.axvline(median, color='g', linestyle='-', label="Median")
       
        #ax_hist.legend()

       # ax_box.set(xlabel='')
        plt.show()

        print ('\033[1m',"Moyenne :",'\033[0m', round(df[var].mean(), 2))
        print ('\033[1m',"Médiane :",'\033[0m', round(df[var].median(), 2))
        print ('\033[1m',"Écart-type :",'\033[0m', round(df[var].std(), 2))
        

        
def stats_region_year(df, indicators, region_col, year_col, region, year):
    palette =["steelblue","lightblue",]
    df = df[(df[region_col]==region) & (df[year_col]==year)]
    print("*"*25,'\033[1m',region,'\033[0m',"*"*25)
    
    for var in indicators:
        print("\n","Indicateur",'\033[1m' , var,'\033[0m',"\n")
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},)
        mean=df[var].mean()
        median=df[var].median()
      
        sns.boxplot(data=df, x=var, ax=ax_box, showfliers = True)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        
        sns.histplot(data=df, x=var, bins=50, ax=ax_hist,)
        ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
        ax_hist.axvline(median, color='g', linestyle='-', label="Median")
       
        #ax_hist.legend()

       # ax_box.set(xlabel='')
        plt.show()

        print ('\033[1m',"Moyenne :",'\033[0m', round(df[var].mean(), 2))
        print ('\033[1m',"Médiane :",'\033[0m', round(df[var].median(), 2))
        print ('\033[1m',"Écart-type :",'\033[0m', round(df[var].std(), 2))