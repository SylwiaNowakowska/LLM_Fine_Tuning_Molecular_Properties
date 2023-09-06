import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def dataset_scale_PCA(dataset_df, feature_columns):
    features_df = dataset_df[feature_columns]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(data=features_scaled, columns=feature_columns)
    
    pca = PCA()
    PCA_fit = pca.fit_transform(features_df)
    PCA_df = pd.DataFrame(data=PCA_fit)
    PCA_df.columns = ["PC" + str(i+1) for i in range(PCA_df.shape[1])]
    PCA_df["Category"] = dataset_df["Category"]
    exp_var_pca = pca.explained_variance_ratio_
    
    return PCA_df, exp_var_pca


def plot_explained_variance_ratio(exp_var_ratio, number_components=3, component_label="PC", cum_sum=True, title=True, fig_output_path=None):
    cum_sum_eigenvalues = np.cumsum(exp_var_ratio)
    plt.figure(figsize=(3,2))
    plt.rcParams.update({'font.size': 16})
    plt.bar(range(0, number_components), exp_var_ratio[ 0: number_components], color="violet", alpha=0.8, align='center', label='Individual', tick_label=[component_label + str(i+1) for i in range(number_components)])
    
    if cum_sum:
        cum_sum_eigenvalues = np.cumsum(exp_var_ratio)
        plt.step(range(0,number_components), cum_sum_eigenvalues[ 0: number_components], color="violet", alpha=0.8, where='mid',label='Cumulative')
        plt.legend(loc='center right')
        
    if title:
        components_variance = round(exp_var_ratio[0:number_components].sum(), 2)
        plt.title(f"Explained variance ratio in {number_components} components: {components_variance}")

    tick_spacing = 0.05
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=tick_spacing))
    plt.ylim([0,0.18])
    plt.tight_layout()

    if fig_output_path:
        plt.savefig(fig_output_path, dpi=100, pad_inches=1, facecolor="white")
    
    plt.show()


def plot_projection(dataframes, class_column, colors, projection_dim=3, figure_title=None, plot_titles=None, title_color='black', fig_output_path=None): 
    columns = len(dataframes)
    
    plt.rcParams.update({'font.size': 20, "axes.labelpad" : 20})
    fig = plt.figure(figsize=(20,8))
     
    if figure_title:
        fig.suptitle(figure_title, fontsize=22, y = 0.95, x = 0.52)

    x_min_plots = []
    x_max_plots = []
    y_min_plots = []
    y_max_plots = []
    z_min_plots = []
    z_max_plots = []
    
    for i, df in enumerate(dataframes):
        targets = df[class_column].value_counts().sort_index()
        targets = targets.index
        
        
        if projection_dim == 2:
            df = df.iloc[:, [0,1,-1]]
            ax = fig.add_subplot(1, columns, i+1)

            ax.set_xlabel(df.columns[0])
            
            if i == 0:
                ax.set_ylabel(df.columns[1])

            for target, color in zip(targets, colors):
                indicesToKeep = df[class_column] == target
                ax.scatter(df.loc[indicesToKeep, df.columns[0]],
                           df.loc[indicesToKeep, df.columns[1]],
                           c = color,
                           s = 20)
            
            # Setting equal x,y axis limits for all plots and equal spacing on the scale
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
                
            x_min_plots.append(x_min)
            x_max_plots.append(x_max)
            y_min_plots.append(y_min)
            y_max_plots.append(y_max)
            
            ax.axis("square")
            ax.set_xlim([min(min(x_min_plots), min(y_min_plots)), max(max(x_max_plots), max(y_max_plots))])
            ax.set_ylim([min(min(x_min_plots), min(y_min_plots)), max(max(x_max_plots), max(y_max_plots))])
            
            # display only y axis info for the first plot
            if i > 0:
                ax.yaxis.set_ticklabels([])
                ax.set_yticks([])

            ax.grid(False)
            
            
            
            if plot_titles:
                ax.set_title(plot_titles[i], color=title_color)

            fig.tight_layout(pad=2.0)
             
        
        elif projection_dim == 3:
            df = df.iloc[:, [0,1,2,-1]]
            ax = fig.add_subplot(1, columns, i+1, projection="3d")


            for target, color in zip(targets, colors):
                indicesToKeep = df[class_column] == target
                ax.scatter(df.loc[indicesToKeep, df.columns[0]],
                           df.loc[indicesToKeep, df.columns[1]],
                           df.loc[indicesToKeep, df.columns[2]],
                           c = color,
                           s = 5)

            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])

            # display only z axis label for the last plot
            if i == (len(dataframes) - 1):
                ax.set_zlabel(df.columns[2])

            ax.xaxis.set_tick_params(labelsize=18)
            ax.yaxis.set_tick_params(labelsize=18)
            ax.zaxis.set_tick_params(labelsize=18)
                
            # Setting equal x,y,z axis limits for all plots with x,y axis having the same spacing
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()
                
            x_min_plots.append(x_min)
            x_max_plots.append(x_max)
            y_min_plots.append(y_min)
            y_max_plots.append(y_max)
            z_min_plots.append(z_min)
            z_max_plots.append(z_max)
            

            ax.set_xlim([min(min(x_min_plots), min(y_min_plots)), max(max(x_max_plots), max(y_max_plots))])
            ax.set_ylim([min(min(x_min_plots), min(y_min_plots)), max(max(x_max_plots), max(y_max_plots))])
            ax.set_zlim([min(z_min_plots), max(z_max_plots)])
            
            ax.grid(False)

            if plot_titles:
                ax.set_title(plot_titles[i])

            fig.tight_layout(pad=1.0)

    legend_labels = ['TN', 'FN', 'FP', 'TP']  
    ax.legend(legend_labels, markerscale=2, loc='upper left')             
    fig.tight_layout(pad=0.1)
            
    if fig_output_path:
        plt.savefig(fig_output_path, dpi=100, pad_inches=1, facecolor="white")
        
    plt.show()