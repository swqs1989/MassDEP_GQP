import pandas as pd
import numpy as np
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join

from collections import Counter
from sklearn import svm, datasets
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from torch.autograd import Variable
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
import time
import copy
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, scale
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.cm as cm
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
import seaborn as sns

# Concatenate the RTN
def completeRTN(row):
    tr = str(int(row["TRACKING_REGION_NUMBER"]))
    tn = str(int(row["TRACKING_NUMBER"]))
    if len(tn) < 7:
        tn = ("0" * (7 - len(tn))) + tn
    return tr + "-" + tn

# process the BWSC101 form
# Standard process is in the extra excel file
# Special process is hard-code.
def preprocess(df, coldef, type):
    columndef = pd.read_excel(coldef)
    cols = df.columns
    if type == '101191':
        # process extraction
        df["A1"] = (df["A1"] == "AM").astype(int)
        df["B6"] = (df["B6"] == "PRP").astype(int)

        df["D11"] = 0

        df["D4N"] = (df["D4M"] == "Y").astype(int)
        df["D4M"] = (df["D4L"] == "Y").astype(int)
        df["D4L"] = (df["D4K"] == "Y").astype(int)
        df["D4K"] = (df["D4J"] == "Y").astype(int)
        df["D4J"] = (df["D4I"] == "Y").astype(int)
        df["D4I"] = (df["D4H"] == "Y").astype(int)
        df["D4H"] = (df["D4G"] == "Y").astype(int)
        df["D4G"] = (df["D4F"] == "Y").astype(int)
        df["D4F"] = (df["D4E"] == "Y").astype(int)
        df["D4E"] = (df["D4D"] == "Y").astype(int)
        df['D4D'] = 0

        df["D9M"] = (df["D9L"] == "Y").astype(int)
        df["D9L"] = (df["D9J"] == "Y").astype(int)
        df["D9J"] = (df["D9K"] == "Y").astype(int)
        df["D9K"] = 0

        df["D10U"] = (df["D10R"] == "Y").astype(int)
        df["D10T"] = (df["D10Q"] == "Y").astype(int)
        df["D10R"] = (df["D10P"] == "Y").astype(int)
        df["D10P"] = 0
        df["D10Q"] = 0
        df["D10S"] = 0

        df["F12"] = (df["F12"] == "PRP").astype(int)

        df["G"] = df["G1"]
        df["G1"] = (df["G"] == "PRENOT").astype(int)
        df["G2"] = 0
        df["G3"] = (df["G"] == "ASSESS").astype(int)
        df["G4"] = (df["G"] == "APORAL").astype(int)
        df["G5"] = (df["G"] == "APORMD").astype(int)
        df["G6"] = (df["G"] == "REQPLN").astype(int)
        df["G7"] = (df["G"] == "INTENT").astype(int)
        df["G8"] = 0
        df["G9"] = 0
        df.drop("G", axis=1, inplace=True)

        df["G11A"] = (df["G8A"] == 'Y').astype(int)
        df["G11B"] = (df["G8B"] == "Y").astype(int)
        df.drop("G8A", axis=1, inplace=True)
        df.drop("G8B", axis=1, inplace=True)

        df["LUST_ELIGIBLE_NO"] = (df["LUST_ELIGIBLE"] == "N").astype(int)
        df["LUST_ELIGIBLE_UNKNOWN"] = (df["LUST_ELIGIBLE"] == "U").astype(int)
        df["LUST_ELIGIBLE_YES"] = (df["LUST_ELIGIBLE"] == "Y").astype(int)
        df.drop("LUST_ELIGIBLE", axis=1, inplace=True)
    elif type == "101592":
        # process extraction
        df["A1"] = (df["A1"] == "AM").astype(int)

        df["B6"] = (df["B6"] == "PRP").astype(int)

        df["D10U"] = (df["D10R"] == "Y").astype(int)
        df["D10T"] = (df["D10Q"] == "Y").astype(int)
        df["D10R"] = (df["D10P"] == "Y").astype(int)
        df["D10P"] = 0
        df["D10Q"] = 0
        df["D10S"] = 0

        df["F12"] = (df["F12"] == "PRP").astype(int)

        df["G"] = df["G1"]
        df["G1"] = (df["G"] == "IRA-PRENOT").astype(int)
        df["G2"] = (df["G"] == "IRA-NOAPP").astype(int)
        df["G3"] = (df["G"] == "IRA-ASSESS").astype(int)
        df["G4"] = (df["G"] == "IRA-APORAL").astype(int)
        df["G5"] = (df["G"] == "IRA-APORMD").astype(int)
        df["G6"] = (df["G"] == "IRA-REQPLN").astype(int)
        df["G7"] = (df["G"] == "URAM-INTENT").astype(int)
        df["G8"] = (df["G"] == "IRA-D-APORAL").astype(int)
        df["G9"] = (df["G"] == "IRA-D-WORKST").astype(int)
        df.drop("G", axis=1, inplace=True)

        df["LUST_ELIGIBLE_NO"] = (df["LUST_ELIGIBLE"] == "N").astype(int)
        df["LUST_ELIGIBLE_UNKNOWN"] = (df["LUST_ELIGIBLE"] == "U").astype(int)
        df["LUST_ELIGIBLE_YES"] = (df["LUST_ELIGIBLE"] == "Y").astype(int)
        df.drop("LUST_ELIGIBLE", axis=1, inplace=True)
    elif type == "101607":
        df["A1"] = (df["A1AM"] == "Y").astype(int)
        df["B6"] = (df["B6OTHER"] == "Y").astype(int)
        df["F12"] = (df["F12OTHER"] == "Y").astype(int)

        df["LUST_ELIGIBLE_NO"] = (df["LUST_ELIGIBLE"] == "N").astype(int)
        df["LUST_ELIGIBLE_UNKNOWN"] = (df["LUST_ELIGIBLE"] == "U").astype(int)
        df["LUST_ELIGIBLE_YES"] = (df["LUST_ELIGIBLE"] == "Y").astype(int)
        df.drop("LUST_ELIGIBLE", axis=1, inplace=True)

    # standard processing
    for col in cols:
        if col == "RTN":
            continue
        # print(col)
        proc = columndef[columndef["feature"] == col]["proc"].values[0]

        # deal with Y/N
        if proc == "transyo10":
            df[col].replace(to_replace={"Y": 1, "Off": 0, "off": 0}, inplace=True)
            df[col] = df[col].astype(int)
        elif proc == "translate10":
            df[col].replace(to_replace={"Y": 1, "N": 0}, inplace=True)
            df[col] = df[col].astype(int)
        # drop column
        elif proc == "drop":
            df.drop(col, axis=1, inplace=True)
        # to be discussed
        elif proc == "?":
            df.drop(col, axis=1, inplace=True)
        # mostly float, and some str
        elif proc == "floatandstr":
            # df.drop(col, axis=1, inplace=True)
            df[col] = df[col].apply(extractlargenumber)
        # change the type to float
        elif proc == "float":
            df[col] = df[col].astype(float)
        elif proc == "str":
            df[col] = df[col].astype(str)
        else:
            pass

# Return the merged BWSC101 form data
def generatedf101():
    df_101191 = pd.read_excel("data/191_BWSC101 Release Log Form.xlsx")
    df_101592 = pd.read_excel("data/592_BWSC101 Release Log Form.xlsx")
    df_101607 = pd.read_excel("data/607_BWSC101 Release Log Form.xlsx")

    df_101191["RTN"] = df_101191.apply(completeRTN, axis=1)
    df_101592["RTN"] = df_101592.apply(completeRTN, axis=1)
    df_101607["RTN"] = df_101607.apply(completeRTN, axis=1)

    preprocess(df_101607, "101607proc.xlsx", "101607")
    print(df_101607.shape)
    preprocess(df_101592, "101592proc.xlsx", "101592")
    print(df_101592.shape)
    preprocess(df_101191, "101191proc.xlsx", "101191")
    print(df_101191.shape)

    df_101 = df_101191.append(df_101592)
    df_101 = df_101.append(df_101607)

    df_101 = df_101[(df_101["A3A"] == 1) | (df_101["A2A"] == 1)]

    df_101["organType"] = df_101.apply(categorizeOrgan, axis=1)
    df_101 = df_101.drop("F_PRP_ACTOR_NAME", axis=1)
    df_101 = pd.get_dummies(df_101, columns=["organType"])

    print(df_101.shape)
    df_101 = df_101.set_index("RTN")
    print("form101 processing finish.")
    return df_101

# return target labels, Tier1D, with RTN
def generatedfclass(day=400):
    df_tclass = pd.read_excel('data/TClass Phase Action Dates All RTNs mgf A 04-10-2018.xlsm', sheetname="All")

    df_tclass = df_tclass.set_index("RTN")

    exclude_status = ['ADQREG', 'DEPMOU', 'DEPNDS', 'DEPNFA', 'DPS', 'DPSTRM', 'INVSUB', 'STMRET', 'URAM', 'UNCLSS']

    if day == 400:
        df_tclass = df_tclass[(df_tclass["Notification"] >= "2006-06-01") & (df_tclass["Notification"] <= "2016-12-28")]
    else:
        df_tclass = df_tclass[(df_tclass["Notification"] >= "2006-06-01") & (df_tclass["Notification"] <= "2016-03-03")]

    df_tclass = df_tclass[~df_tclass["Status"].isin(exclude_status)]

    drop_index = df_tclass[(df_tclass["Status"].isin(["REMOPS", "ROSTRM", "TCLASS", "TIERI", "TIERII"])) \
    & df_tclass["Phase1Cs"].isnull() & df_tclass["RaoNr"].isnull()].index
    #  Synthetic Minority Over-sampling Technique
    df_tclass = df_tclass.drop(drop_index, axis=0)

    df_tclass["length"] = df_tclass.apply(daylength, axis=1)

    df_tclass = df_tclass[df_tclass["length"] >= 0]

    if day == 400:
        df_tclass["Tier1D"] = df_tclass.apply(isTier1D, axis=1)
    else:
        df_tclass["Tier1D"] = df_tclass.apply(isPTier1D, axis=1)

    print("tclass processing finish.")
    return df_tclass

# return corresponding GIS data with RTN
def generategistract():
    CENSUS_2010_B = pd.read_excel("data/Intersect_Release_CENSUS_2010_Blocks.xls")
    T_group_quarters_pop                = pd.read_excel("data/GIS/TRACT/group_quarters_pop.xlsx")
    T_households_by_age_family_children = pd.read_excel("data/GIS/TRACT/households_by_age_family_children.xlsx")
    T_households_size_by_family = pd.read_excel("data/GIS/TRACT/households_size_by_family.xlsx")
    T_housing_owner_rental_demographics = pd.read_excel("data/GIS/TRACT/housing_owner_rental_demographics.xlsx")
    T_housing_residency_characteristics = pd.read_excel("data/GIS/TRACT/housing_residency_characteristics.xlsx")
    T_pop_by_age_gender = pd.read_excel("data/GIS/TRACT/pop_by_age_gender.xlsx")
    T_pop_by_race = pd.read_excel("data/GIS/TRACT/pop_by_race.xlsx")

    CENSUS_2010_B = CENSUS_2010_B.loc[:, ["rtn", "GEOID10"]]
    gis_block = T_group_quarters_pop.join(T_households_by_age_family_children.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(T_households_size_by_family.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(T_housing_owner_rental_demographics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(T_housing_residency_characteristics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(T_pop_by_age_gender.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(T_pop_by_race.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')

    CENSUS_2010_B['GEOID10'] = CENSUS_2010_B['GEOID10'].apply(lambda x : str(x)[:11])
    gis_block["GEOID10"] = gis_block["GEOID10"].astype(str)
    gis_block = CENSUS_2010_B.join(gis_block.set_index("GEOID10"), how='inner', on="GEOID10", rsuffix='_')
    gis_block = gis_block.set_index("rtn")
    features = gis_block.columns.tolist()
    features = list(filter(lambda a: a != 'GEOID10_', features))
    features = list(filter(lambda a: a != 'GEOID10', features))
    features = list(filter(lambda a: a != 'LOGRECNO', features))
    gis_block = gis_block.loc[:, features]
    return gis_block

def generategisblock():
    CENSUS_2010_B = pd.read_excel("data/Intersect_Release_CENSUS_2010_Blocks.xls")
    B_group_quarters_pop                = pd.read_excel("data/GIS/BLK/group_quarters_pop.xlsx")
    B_households_by_age_family_children = pd.read_excel("data/GIS/BLK/households_by_age_family_children.xlsx")
    B_households_size_by_family = pd.read_excel("data/GIS/BLK/households_size_by_family.xlsx")
    B_housing_owner_rental_demographics = pd.read_excel("data/GIS/BLK/housing_owner_rental_demographics.xlsx")
    B_housing_residency_characteristics = pd.read_excel("data/GIS/BLK/housing_residency_characteristics.xlsx")
    B_pop_by_age_gender = pd.read_excel("data/GIS/BLK/pop_by_age_gender.xlsx")
    B_pop_by_race = pd.read_excel("data/GIS/BLK/pop_by_race.xlsx")

    gis_block = B_group_quarters_pop.join(B_households_by_age_family_children.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_households_size_by_family.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_housing_owner_rental_demographics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_housing_residency_characteristics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_pop_by_age_gender.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_pop_by_race.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')

    features = gis_block.columns.tolist()
    features = list(filter(lambda a: a != 'GEOID10_', features))
    features = list(filter(lambda a: a != 'GEOID10', features))
    gis_block = gis_block.loc[:, features]
    gis_block = gis_block.set_index("LOGRECNO")
    selected = ["LOGSF1", "rtn"]
    df_gisblock = CENSUS_2010_B.loc[:, selected].set_index('LOGSF1').join(gis_block, how='inner')

    df_gisblock = df_gisblock.set_index("rtn")

    return df_gisblock

def generategisblockgroup():
    CENSUS_2010_BG = pd.read_excel("data/Intersect_Release_CENSUS_2010_Block_Groups.xls")
    B_group_quarters_pop                = pd.read_excel("data/GIS/BLKGRP/group_quarters_pop.xlsx")
    B_households_by_age_family_children = pd.read_excel("data/GIS/BLKGRP/households_by_age_family_children.xlsx")
    B_households_size_by_family = pd.read_excel("data/GIS/BLKGRP/households_size_by_family.xlsx")
    B_housing_owner_rental_demographics = pd.read_excel("data/GIS/BLKGRP/housing_owner_rental_demographics.xlsx")
    B_housing_residency_characteristics = pd.read_excel("data/GIS/BLKGRP/housing_residency_characteristics.xlsx")
    B_pop_by_age_gender = pd.read_excel("data/GIS/BLKGRP/pop_by_age_gender.xlsx")
    B_pop_by_race = pd.read_excel("data/GIS/BLKGRP/pop_by_race.xlsx")

    gis_block = B_group_quarters_pop.join(B_households_by_age_family_children.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_households_size_by_family.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_housing_owner_rental_demographics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_housing_residency_characteristics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_pop_by_age_gender.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
    gis_block = gis_block.join(B_pop_by_race.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')

    features = gis_block.columns.tolist()
    features = list(filter(lambda a: a != 'GEOID10_', features))
    features = list(filter(lambda a: a != 'GEOID10', features))
    gis_block = gis_block.loc[:, features]
    gis_block = gis_block.set_index("LOGRECNO")
    selected = ["LOGSF1", "rtn"]
    df_gisblock = CENSUS_2010_BG.loc[:, selected].set_index('LOGSF1').join(gis_block, how='inner')

    df_gisblock = df_gisblock.set_index("rtn")

    return df_gisblock

# return the chemicals information with RTN
def generatechemicals():
    df_chemicals = pd.read_excel("data/Chemical_Class_Features1.xlsx")
    df_chemicals = df_chemicals.set_index("RTN")
    df_chemicals = df_chemicals.drop("ERROR", axis=1)

    list_chemicals = df_chemicals.columns.tolist()

    dict_chemicals = {}
    for che in list_chemicals:
        dict_chemicals[che] = 0.0

def joingeofile(df):
    path = "data/GIS/geo/"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for f in onlyfiles:
        if "xls" in f:
            geo = f[18:-4]
            df_geo = pd.read_excel(path+f)
            df_geo = df_geo.drop_duplicates("rtn")
            df_geo[geo] = 1.0
            df = df.join(df_geo.set_index("rtn")[geo], how='left').fillna(0)
            print("join geo: %r, %r" %(f, str(df.shape)))
    return df


def categorizeOrgan(row):
    organ = row["F_PRP_ACTOR_NAME"]

    organ = str.lower(organ)
    organ = organ.replace(".", " ")
    organ = organ.replace(",", " ")

    organs = organ.split(" ")
    if 'inc' in organs:
        return 'inc'
    elif 'llc' in organs:
        return 'llc'
    elif 'trust' in organs:
        return 'trust'
    elif 'dba' in organs:
        return 'dba'
    else:
        return 'individual'

def prepmissing(df):
    attributes = df.columns
    nominalvalues = {}

    # df = df.replace('N/A', np.NaN)
    # df = df.replace('?', np.NaN)
    for col in df.columns:
        # deal with missing values
        if sum(pd.isnull(df[col])) != 0 or sum(df[col].isin(["?"])) > 0:
            print("%r column (type: %r): %r null" %(col, df[col].dtype, sum(pd.isnull(df[col]))))

def processChemicals(df):
    return None

def extractlargenumber(cell):
    cell = str(cell)
    if cell == "":
        return 0

    strs = cell.split(" ")
    number = 0.
    for s in strs:
        try:
            num = float(s.replace(",", ""))
            if num > number:
                number = num
        except ValueError:
            pass
    return number

def processtiers(df):
    dftier = pd.DataFrame(columns=["RTN", "Tier"])
    for rtn in df["RTN"].unique().tolist():
        if len(df[df['RTN'] == rtn]) > 1:
            flag = df[df['RTN'] == rtn]['newtc'].sum() + df[df['RTN'] == rtn]['revisedtc'].sum()
            if flag > 0:
                dftier = dftier.append({"RTN": rtn, "Tier": 1}, ignore_index=True)
            else:
                dftier = dftier.append({"RTN": rtn, "Tier": 2}, ignore_index=True)
        else:
            flag = (df[df['RTN'] == rtn]['newtc'].values[0]) or (df[df['RTN'] == rtn]['revisedtc'].values[0])
            if flag:
                dftier = dftier.append({"RTN": rtn, "Tier": 1}, ignore_index=True)
            else:
                dftier = dftier.append({"RTN": rtn, "Tier": 2}, ignore_index=True)
    return dftier

def isTier1D(row):
    t = row["length"]
    if t < 400:
        return "Non-Tier1D"
    else:
        return "Tier1D"

def isPTier1D(row):
    t = row["length"]
    if t < 700:
        return "Non-Tier1D"
    else:
        return "Persist-Tier1D"

def daylength(row):
    t1 = row["Notification"]
    t2 = row["Phase1Cs"]
    # t3 = row["End Date"]
    t4 = row["Regulatory End Date"]
    raonr = row["RaoNr"]
    status = row["Status"]
    if status == "TIER1D":
        return 999
    else:
        if status in ["RAO", "PSC", "PSNC", "SPECPR", "TMPS", "RAONR"]:
            T = t4 - t1
        elif status in ["REMOPS", "ROSTRM", "TCLASS", "TIERI", "TIERII"]:
            if not pd.isnull(t2):
                T = t2 - t1
            else:
                t5 = raonr.split(" ")[1]

                m, d, y = t5.split("/")
                t5 = datetime(int(y), int(m), int(d))

                T = t5 - t1
    return T.days

def runclassifier(clf, X_train, y_train, X_test, y_test, labels):
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    plot_confusion_matrix(confusion_matrix(y_test, y_predict, labels=labels), classes=labels)
    print(classification_report(y_test, y_predict, labels=labels, target_names=labels))

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def regressioncm(y_true, y_pred, labels=[1, 0]):
    plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=labels), classes=labels, normalize=False)
    print(classification_report(y_true, y_pred, labels=labels, target_names=["Tier1D", "Non-Tier1D"]))

class Clf():
    def __init__(self, X, y, labels):
        self.X = X
        self.y = y
        self.labels = labels
        self.clf_eval = pd.DataFrame(data=None, columns=['clf', 'params', 'smote', 'Trecall', 'Tprecision', 'Tf1', 'Frecall', 'Fprecision', 'Ff1'])
        # self.clf_eval = {}
        # self.clf_eval['clf'] = str(self.clf).split('(')[0]
        # self.clf_eval['smote'] = self.smote

    def runKfold(self, classifier, param, smote=False, dr=False, drp=10, k=4):
        skf = StratifiedKFold(n_splits=k, random_state=122, shuffle=True)
        param_grid = ParameterGrid(param)
        for params in param_grid:
            Trecalls = []
            Tprecisions = []
            Frecalls = []
            Fprecisions = []
            Tf1s = []
            Ff1s = []
            for train_index, test_index in skf.split(self.X, self.y):

                X_train, X_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
                y_train, y_test = self.y[train_index], self.y[test_index]

                if dr:
                    pca = PCA(n_components=drp)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)

                if smote:
                    X_train, y_train = SMOTE().fit_sample(X_train, y_train)
                # train
                clf = classifier(**params)
                clf.fit(X_train, y_train)

                y_predict = clf.predict(X_test)

                metrics = precision_recall_fscore_support(y_test, y_predict, labels=self.labels)

                Tprecision, Fprecision = metrics[0][0], metrics[0][1]
                Trecall, Frecall = metrics[1][0], metrics[1][1]
                Tf1, Ff1 = metrics[2][0], metrics[2][1]

                Trecalls.append(Trecall)
                Tprecisions.append(Tprecision)
                Frecalls.append(Frecall)
                Fprecisions.append(Fprecision)
                Tf1s.append(Tf1)
                Ff1s.append(Ff1)
                # print(classification_report(y_test, y_predict, labels=[1, 0], target_names=['Tier 1D','other']))
                # report = classification_report(y_test, y_predict, labels=[True, False], target_names=['Tier 1D','other'])

            self.clf_eval = self.clf_eval.append({'clf': str(clf).split('(')[0],
                                  'params': params,
                                  'smote': smote,
                                  'Trecall': np.mean(Trecalls),
                                  'Tprecision': np.mean(Tprecisions),
                                  'Tf1': np.mean(Tf1s),
                                  'Frecall': np.mean(Frecalls),
                                  'Fprecision': np.mean(Fprecisions),
                                  'Ff1': np.mean(Ff1s)}, ignore_index=True)
            print("%r %r" %(str(clf).split('(')[0], params))
            print("Done")
            # self.clf_eval['params'] = params
            # self.clf_eval['recall'] = np.mean(recalls)
