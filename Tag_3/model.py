import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree, metrics, ensemble
import graphviz

class model_poverty_prob:

    def __init__(self, data_str, target_str):
        self.data_str = data_str
        self.target_str = target_str

    def plot_tree(self):
        tree.export_graphviz(self.model_tree, out_file="model_tree.dot", 
                    feature_names=self.X_test[self.model_tree.feature_names_in_].columns,
                    # class_names=['0','1','2'], 
                    rounded=True, filled=True)
    
        # Import model graph
        with open("model_tree.dot") as f:
            dot_graph = f.read()
            display(graphviz.Source(dot_graph))

    def tree_mod(self,col_list,min_samples_leaf=2):
        self.model_tree = tree.DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
        self.model_tree = self.model_tree.fit(self.X_train[col_list], self.y_train)


    def forest_mod(self,col_list,n_estimators=100,min_samples_leaf=2):
        self.model_forest = ensemble.RandomForestRegressor(n_estimators=n_estimators,
                            min_samples_leaf=min_samples_leaf)
        self.model_forest = self.model_forest.fit(self.X_train[col_list], self.y_train)
    
    def mean_squared_error(self):
        
        mse_dic = {}
        
        if hasattr(self, 'model_OLS'):
            mse_dic['OLS'] = metrics.mean_squared_error(self.y_test,self.model_OLS.predict(pd.merge(pd.DataFrame(
    data={'const' : [1 for i in range(self.X_test.shape[0])]},
    index=self.X_test[self.model_OLS.params.index[1:]].index),
         self.X_test[self.model_OLS.params.index[1:]],
        left_index = True, right_index = True)))
            
        if hasattr(self, 'model_tree'):
            mse_dic['Tree'] = metrics.mean_squared_error(self.y_test,
                                                         self.model_tree.predict(self.X_test[self.model_tree.feature_names_in_]))
        if hasattr(self, 'model_forest'):
            mse_dic['Forest'] = metrics.mean_squared_error(self.y_test,
                                                         self.model_forest.predict(self.X_test[self.model_forest.feature_names_in_]))

        


        self.mse = pd.DataFrame(data = mse_dic, index = ['mean squared error'])
    
    def tt_split(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_df.drop(columns='poverty_probability'), 
                                                                                self.data_df['poverty_probability'], 
                                                                                test_size=0.2, 
                                                                                random_state=42)

    def MinMax_alter(self):

        scaler = MinMaxScaler()

        self.X_train['age'] = scaler.fit_transform(self.X_train[['age']])
        self.X_test['age'] = scaler.transform(self.X_test[['age']])
        
    # DataFrame mit einer feinere Regel füllen
    # APPLY FUNCTION
    def app_func(self,row):
        """
        Lineare Interpolation von NaN zwischen 2 vorhanden Werte 
        """
    
        # kontrollieren ob NaN in der Spalte sind
        if row.isnull().sum() > 0:
            _row = row # temporär Kopie von der Spalte
            null_indices = _row[_row.isna()].index # Indexe von NaN
    
            first_index = null_indices[0]-1 # Index vom letzten Wert vor NaNs
            last_index = null_indices[-1]+1 # Index vom ertsten Wert nach NaNs
    
            # Series mit den interpolierten Daten sowie die Indexe von NaNs
            null_substitution = pd.Series(data  = [_row[first_index] + i*(_row[last_index]-_row[first_index])/(len(null_indices)+1) for i in range(1,len(null_indices)+1)],
                                       index = null_indices) 
    
            # Überschreiben von NaN in der temporär Kopie der Spalte mit den interpolierten Werte
            _row[null_indices] = null_substitution
    
            # Rückgabe vom gefüllte temporär Kopie der Spalte
            return _row
        # falls keine NaN, geben wir die gleiche Spalte zurück 
        else:
            return row
            
    def prep_dataset(self):
        self.data_df = pd.read_excel(f'../{self.data_str}')
        self.target_df = pd.read_excel(f'../{self.target_str}',
                        index_col= 0)

        self.data_df['gender'] = self.data_df['female'].map({True: 'female', False: 'male'})
        # self.train_df = self.train_df.rename(columns={'female':'gender'})

        self.data_df['row_id'] = self.app_func(self.data_df['row_id'])

        self.data_df = self.data_df.set_index('row_id')

        self.data_df = self.data_df.merge(pd.get_dummies(self.data_df['country'], prefix = 'country_'),left_index=True, right_index=True)

        self.data_df = self.data_df.merge(self.target_df, left_index=True, right_index=True)

        for col in self.data_df.columns:
            if self.data_df[col].dtype == 'bool':
                self.data_df[col] = self.data_df[col].astype(int)

    def describe_dataset(self):

        self.data_df.info()
        
        _corr = pd.Series()
        _skew = pd.Series()
        _kurt = pd.Series()
    
        for col in self.data_df.columns:
            try:
                _corr[col] = np.corrcoef(x = df[col], y = self.data_df['poverty_probability'])[0][1]
            except:
                _corr[col] = np.nan
            try:
                _skew[col] = stats.skew(self.data_df[col])
            except:
                _skew[col] = np.nan
            try:
                _kurt[col] = stats.kurtosis(self.data_df[col])
            except:
                _kurt[col] = np.nan
    
        stats_df = self.data_df.describe(include = 'all').T
        stats_df['corrcoef'] = _corr
        stats_df['skew'] = _skew
        stats_df['kurtosis'] = _kurt

        self.data_df_describe = stats_df

    def drop_cols(self,col_list):

        self.data_df = self.data_df.drop(columns=col_list)
    
    def drop_na(self,col_list):
        self.data_df = self.data_df.dropna(subset = 'education_level')

    def OLS_mod(self,col_list):
        
        self.X_OLS= self.data_df[col_list]#,'can_text','num_financial_activities_last_year','age'

        self.X_OLS = sm.add_constant(self.X_OLS) # Adding a constant to include an intercept in the model
        self.model_OLS = sm.OLS(self.data_df['poverty_probability'], self.X_OLS)
        self.model_OLS = self.model_OLS.fit()
        print(self.model_OLS.summary())

    def OLS_charts(self):

        # # Residual plot
        # axs[0].scatter(self.model_OLS.fittedvalues, self.model_OLS.resid)
        # axs[0].set_xlabel('Fitted values')
        # axs[0].set_ylabel('Residuals')
        # axs[0].set_title('Residual vs. Fitted Values')

        fig = plt.figure(figsize=(10,20))
        fig = sm.graphics.plot_partregress_grid(mod_.model_OLS, fig=fig)
        plt.show()


        fig, axs = plt.subplots(1,2,figsize=(20,10))

        
        # QQ plot for normality
        sm.qqplot(self.model_OLS.resid, fit=True, line='45',ax=axs[0])
        axs[0].set_title('QQ plot for normality')

        sns.histplot(self.model_OLS.resid, kde=True, edgecolor='w', ax=axs[1]);
        axs[1].set_xlabel('Residuals')
        axs[1].set_title('Residuals distribution')

    def OLS_pairplot(self):
        sns.pairplot(data = self.X_OLS.drop(columns='const').merge(self.data_df['poverty_probability'], left_index=True, right_index=True), kind='kde')
        

        
    if __name__ == '__main__':
        pass