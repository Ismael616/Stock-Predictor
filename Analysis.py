def analyse_num (df) :
    """
    build the summary statistics of a dataframe numerical features  adding :
    Q1: first quartile
    Q3: third quartile
    IQR: Inter Quartile Range 
    outliers%:outlier percentage of every features
    null_values%: null values perentage  of every features
    skewness: the skewness of the feature
    kurtosis: the kustosis of the feature
    mean_dispertion%: dispertion percentage of data points around the mean
    **********PARAMETERS**********
    df : The  pandas dataframe to analyse 
    """
    num_columns=df.select_dtypes(include=['number']).columns
    df_num=df[num_columns]
    summary_df=df_num.describe().T
    summary_df.rename(columns={'50%':'Median'},inplace=True)
    summary_df.rename(columns={'25%':'Q1'},inplace=True)
    summary_df.rename(columns={'75%':'Q3'},inplace=True)
    summary_df['IQR']=summary_df['Q3']-summary_df['Q1']
    up=summary_df['Q3']+ 1.5 *(summary_df['IQR'])
    low=summary_df['Q1']- 1.5 *(summary_df['IQR'])
    summary_df['Variance']=summary_df['std']**2
    summary_df["null_values%"]=df_num.isna().sum()/summary_df['count']*100
    summary_df['outliers%']=df_num[(df_num >= up) | (df_num<=low)].count()/df_num.count()*100
    #summary_df["erreur_type"] = summary_df["std"]/np.sqrt(summary_df["count"])
    summary_df["mean_dispersion%"] = summary_df["std"]/summary_df["mean"]*100
    summary_df['skewness']=df_num.skew() # skew normal 0
    summary_df['kurtosis']=df_num.kurtosis() #kusrtosis normal 3
    return summary_df

def analyse_obj (df) :
    """
    build the summary statistics of a dataframe non-numerical features  adding :
    null_values%: null values perentage  of every features
    **********PARAMETERS**********
    df : the pandas dataframe dataframe to analyse 
    """
    cols=df.select_dtypes(include=['object']).columns
    df=df[cols]
    summary_df=df.describe().T
    summary_df['null values%']=df.isna().sum()/summary_df['count']*100
    return summary_df