# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:40:20 2016

@author: lshu0
"""

from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import datetime

# load test control matching and raw data
test_control_mapping = pd.read_csv('Sears Blackout Week 17 test&control stores.csv')
store_data = pd.read_csv('store data.csv')

# arange raw data, create two DataFrames for test stores and averaged control stores
control_site_df = pd.merge(test_control_mapping , store_data ,  left_on = 'Similar site ID', right_on =  'LOCN_NBR' , how = 'left')
control_site_df = control_site_df.iloc[:,[0,2,6,12,13,14,15,16]]
test_site_df = pd.merge(test_control_mapping , store_data ,  left_on = 'Test site ID', right_on =  'LOCN_NBR' , how = 'left')
test_site_df = test_site_df.iloc[:,[0,6,12,13,14,15,16]]
test_site_df.drop_duplicates(inplace = True)
avg_control = control_site_df.groupby(['Test site ID', 'WK_NBR']).mean().reset_index()
avg_control.rename(columns = {'SEARSTOTALSALES':'Control Avg Sales',
                              'SEARSTOTALSALESUNITS':'Control Avg Units',
                              'margin':'Control Avg Margin', 
                              'OTD':'Control Avg OTD'},inplace = True)

# set post and pre period
pre_period_to = 201609
post_period_from = 201617
test_vs_control = pd.merge(test_site_df, avg_control, left_on = ['Test site ID', 'WK_NBR'] ,right_on = ['Test site ID','WK_NBR'], how = 'left' )
test_vs_control['period'] = np.where(test_vs_control['WK_NBR'] <= pre_period_to, 'Pre',np.where(test_vs_control['WK_NBR'] >= post_period_from, 'Post','Disruption'))

#OTD_df = test_vs_control.icol([0,1,6,12,13])

# seperate different metrics
sales_df = test_vs_control.iloc[:,[0,1,2,8,13]]
unit_df = test_vs_control.iloc[:,[0,1,3,9,13]]
margin_df = test_vs_control.iloc[:,[0,1,5,11,13]]

# create transparency reports for sales, units and margin
metric_list = [sales_df, unit_df, margin_df]
metric_index = ['Item - Net Sales - All','Item - Net Quantity - All', 
                'Item - SPRS Margin - All', 'Item - Average Selling Price']
pivot_list = []
def outlier_rsn(row):
    if row['lift']> 0.95:
        return 'Statistical outlier'
    if row.iget(5) == 0:
        return 'Test site has zero data in the baseline period'
    if row.iget(5) < 0:
        return 'Test site has negatvie data in the baseline period'
    if row.iget(4) == 0:
        return 'Test site has zero data in the analysis period'
    if row.iget(4) < 0:
        return 'Test site has negative data in the analysis period'
    if row.iget(7) == 0:
        return 'Control sites have missing, negative, or zero data in the analysis period'

for metric in metric_list:
    df = metric.groupby(['Test site ID','period']).mean().unstack()
    df.columns = [' '.join(col).strip() for col in df.columns.values]    
    df['test performance'] = df.iloc[:,4]/df.iloc[:,5] -1 
    df['control performance'] = df.iloc[:,7]/df.iloc[:,8] - 1
    df['test expected'] = df.iloc[:,5]*df.iloc[:,7]/df.iloc[:,8]
    df['estimated impact'] = df.iloc[:,4] - df['test expected']
    df['lift'] = df['estimated impact']/df['test expected']
    df['Outlier Reason'] = df.apply(outlier_rsn,axis = 1)
    df['Is outlier?']= np.where(pd.isnull(df['Outlier Reason']) == False, 'Yes', 'No')
    pivot_list.append(df)

# create transparency report for ASP from the ones of sales and units
    # test pre,post,disruption, performance
pivot_ASP = pd.DataFrame(data=None, columns=pivot_list[0].columns,
                         index=pivot_list[0].index) 
pivot_ASP.iloc[:,3] = pivot_list[0].iloc[:,3]/pivot_list[1].iloc[:,3]
pivot_ASP.iloc[:,4] = pivot_list[0].iloc[:,4]/pivot_list[1].iloc[:,4]
pivot_ASP.iloc[:,5] = pivot_list[0].iloc[:,5]/pivot_list[1].iloc[:,5]
pivot_ASP.iloc[:,9] = pivot_ASP.iloc[:,4]/pivot_ASP.iloc[:,5] -1
    # control pre,post,disruption, performance
control_site_df['period'] = np.where(control_site_df['WK_NBR'] <= pre_period_to, 'Pre',np.where(control_site_df['WK_NBR'] >= post_period_from, 'Post','Disruption'))
tot_control = control_site_df[['SEARSTOTALSALES','SEARSTOTALSALESUNITS']].groupby([control_site_df['Test site ID'], control_site_df['period']]).sum().unstack()
tot_control.columns = [' '.join(col).strip() for col in tot_control.columns.values]  
pivot_ASP.iloc[:,6] = tot_control.iloc[:,0]/tot_control.iloc[:,3]
pivot_ASP.iloc[:,7] = tot_control.iloc[:,1]/tot_control.iloc[:,4]
pivot_ASP.iloc[:,8] = tot_control.iloc[:,2]/tot_control.iloc[:,5]
pivot_ASP.iloc[:,10] = pivot_ASP.iloc[:,7]/pivot_ASP.iloc[:,8]-1
pivot_ASP.iloc[:,11] = pivot_ASP.iloc[:,5] * pivot_ASP.iloc[:,7]/pivot_ASP.iloc[:,8]
pivot_ASP.iloc[:,12] = pivot_ASP.iloc[:,4] - pivot_ASP.iloc[:,11]
pivot_ASP.iloc[:,13] = pivot_ASP.iloc[:,12]/pivot_ASP.iloc[:,11]

def ASP_outlier_rsn(row):
    if row['lift']> 0.30 or row['lift'] < -0.30:
        return 'Statistical outlier'
pivot_ASP['Outlier Reason'] = pivot_ASP.apply(ASP_outlier_rsn,axis = 1)
pivot_ASP['Is outlier?']= np.where(pd.isnull(pivot_ASP['Outlier Reason']) == False, 'Yes', 'No')
pivot_ASP.rename(columns = {'SEARSTOTALSALES Disruption':'ASP Disruption',
                            'SEARSTOTALSALES Post':'ASP Post',
                            'SEARSTOTALSALES Pre':'ASP Pre',
                            'Control Avg Sales Disruption':'Control ASP Disruption',
                            'Control Avg Sales Post':'Control ASP Post',
                            'Control Avg Sales Pre':'Control ASP Pre'},inplace = True)

# create grid charts of sales, units and margin
grid_list = []
store_list_metric = []
for pivot in pivot_list:
    grid={}
    df = pivot[pivot['Is outlier?']=='No'] 
    store_list_metric.append(df.index.values)
    grid['Significance'] = 1 - ttest_ind(df.iloc[:,4],df.iloc[:,11])[1]
    grid['Test pre-period (per site)'] = df.iloc[:,5].mean()
    grid['Test post-period (per site)']= df.iloc[:,4].mean()
    grid['Test expected (per site)'] =df.iloc[:,11].mean()
    grid['Control pre-period (per site)'] = df.iloc[:,8].mean() 
    grid['Control post-period (per site)'] = df.iloc[:,7].mean()
    grid['Est. Impact (per site)']= grid['Test post-period (per site)'] - grid['Test expected (per site)']
    grid['Test Count'] = df.shape[0]
    grid['Test pre-period (total)'] = df.iloc[:,5].sum()
    grid['Test post-period (total)']= df.iloc[:,4].sum()
    grid['Test expected (total)'] =df.iloc[:,11].sum()
    grid['Control pre-period (total)'] = df.iloc[:,8].sum() 
    grid['Control post-period (total)'] = df.iloc[:,7].sum() 
    grid['Est. Impact (total)']= grid['Test post-period (total)'] - grid['Test expected (total)']
    grid['Lift'] = grid['Est. Impact (per site)']/ grid['Test expected (per site)']
    grid_list.append(grid)
    
# creat grid chart for ASP
all_info_ASP = pd.merge(pivot_ASP, pivot_list[0].iloc[:,3:12], left_index = True, right_index = True)
all_info_ASP = pd.merge(all_info_ASP, pivot_list[1].iloc[:,3:12], left_index = True, right_index = True)
all_info_ASP = all_info_ASP[all_info_ASP['Is outlier?']=='No']
store_list_metric.append(all_info_ASP.index.values)
grid={}
grid['Significance'] = 1 - ttest_ind(all_info_ASP['ASP Post'] ,all_info_ASP['test expected_x'])[1]
grid['Test pre-period (per site)'] = all_info_ASP['SEARSTOTALSALES Pre'].sum() / all_info_ASP['SEARSTOTALSALESUNITS Pre'].sum()
grid['Test post-period (per site)']= all_info_ASP['SEARSTOTALSALES Post'].sum() / all_info_ASP['SEARSTOTALSALESUNITS Post'].sum()
grid['Test expected (per site)'] =  all_info_ASP['test expected_y'].sum() / all_info_ASP['test expected'].sum()
grid['Control pre-period (per site)'] = all_info_ASP['Control Avg Sales Pre'].sum() / all_info_ASP['Control Avg Units Pre'].sum()
grid['Control post-period (per site)'] = all_info_ASP['Control Avg Sales Post'].sum() / all_info_ASP['Control Avg Units Post'].sum()
grid['Est. Impact (per site)']= grid['Test post-period (per site)'] - grid['Test expected (per site)']
grid['Test Count'] = all_info_ASP.shape[0]
grid['Lift'] = grid['Est. Impact (per site)']/ grid['Test expected (per site)']
grid_list.append(grid)
   
result_grid = pd.DataFrame(grid_list,index = metric_index )
result_grid.index.name = 'metric'
result_grid.reset_index(inplace  = True)
result_grid['order'] = np.arange(4)+1
result_grid['significance level'] = np.where(result_grid['Significance']<0.8, 'Low', np.where(result_grid['Significance']<0.95,'Mid','High'))
result_grid['category'] = 'Week17_Blackout_Test_Items'
result_grid['text_color'] = np.where(result_grid['Lift'] >= 0, 'black','red' )
 

# create trend charts for sales, units and margin
trends = []
ASP_trend_components = []
for i in xrange(3):
    filtered_metric = metric_list[i][metric_list[i].iloc[:,0].isin(store_list_metric[i])]
    pre_df = pivot_list[i].iloc[:,[5,8]]
    add_pre = pd.merge(filtered_metric, pre_df, left_on = 'Test site ID', right_index = True, how = 'left' )
    add_pre['Control post-pre ratio'] = add_pre.iloc[:,3]/add_pre.iloc[:,6]
    add_pre['Test expected'] = add_pre.iloc[:,5]*add_pre['Control post-pre ratio']
    pivot_trend = add_pre.groupby(['WK_NBR','period']).mean().reset_index()
    pivot_trend['Estimated impact'] = pivot_trend.iloc[:,3] - pivot_trend.iloc[:,8]
    pivot_trend['Lift'] = pivot_trend['Estimated impact']/pivot_trend.iloc[:,8]
    trends.append(pivot_trend)
    ASP_filtered_metric = metric_list[i][metric_list[i].iloc[:,0].isin(store_list_metric[3])]    
    ASP_add_pre = pd.merge(ASP_filtered_metric, pre_df, left_on = 'Test site ID', right_index = True, how = 'left' )    
    ASP_add_pre['Control post-pre ratio'] = ASP_add_pre.iloc[:,3]/ASP_add_pre.iloc[:,6]
    ASP_add_pre['Test expected'] = ASP_add_pre.iloc[:,5]*ASP_add_pre['Control post-pre ratio']
    ASP_pivot_trend = ASP_add_pre.groupby(['WK_NBR','period']).mean().reset_index()
    ASP_trend_components.append(ASP_pivot_trend )
    
# create trend chart for ASP
trend_ASP = pd.DataFrame(data = None, columns = trends[0].columns 
                        , index = trends[0].index )
trend_ASP['WK_NBR'] = ASP_trend_components[0]['WK_NBR']
trend_ASP['period'] = ASP_trend_components[0]['period']
trend_ASP.iloc[:,3] = ASP_trend_components[0].iloc[:,3]/ASP_trend_components[1].iloc[:,3]
trend_ASP.iloc[:,4] = ASP_trend_components[0].iloc[:,4]/ASP_trend_components[1].iloc[:,4]
trend_ASP.iloc[:,5] = ASP_trend_components[0].iloc[:,5]/ASP_trend_components[1].iloc[:,5]    
trend_ASP.iloc[:,6] = ASP_trend_components[0].iloc[:,6]/ASP_trend_components[1].iloc[:,6]
trend_ASP.iloc[:,8] = ASP_trend_components[0].iloc[:,8]/ASP_trend_components[1].iloc[:,8]
trend_ASP.iloc[:,9] = trend_ASP.iloc[:,3] - trend_ASP.iloc[:,8]
trend_ASP.iloc[:,10] = trend_ASP.iloc[:,9] / trend_ASP.iloc[:,8]
trend_ASP.rename(columns = {'SEARSTOTALSALES':'Test post ASP',
                            'Control Avg Sales': 'Control post ASP',
                            'SEARSTOTALSALES Pre':'Test pre ASP',
                            'Control Avg Sales Pre':'Control pre ASP'},inplace = True)
trends.append(trend_ASP)

# plot result grid
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show, output_file, vplot
#from bokeh.models import Span

fill_colormap = {
    "Low"         : "#f8d8d8",
    "Mid"         : "#fdfcaa",
    "High"        : "#a0c096"
}

frame_colormap = {
    "Low"         : "#9b0303",
    "Mid"         : "#eae72e",
    "High"        : "#478532"
} 

source = ColumnDataSource(
    data=dict(
        metric=result_grid["metric"],
        cat = result_grid['category'],
        order=[str(x) for x in result_grid["order"]],
        lift =['Lift: '+"{:.2%}".format(x) for x in result_grid["Lift"]],
        impact_site=['Impact: '+ "{0:.2f}".format(x) for x in result_grid["Est. Impact (per site)"]],
        expected_site=['Expected: '+"{0:.2f}".format(x) for x in result_grid["Test expected (per site)"]],
        actual_site=['Actual: ' + "{0:.2f}".format(x) for x in result_grid["Test post-period (per site)"]],
        site_count=[str(x)+' Test Count' for x in result_grid["Test Count"]],
        significance =  ['Significance: '+"{:.2%}".format(x) for x in result_grid["Significance"]],            
        lifty = [str(x)+":0.8" for x in result_grid["category"]],
        impact_sitey = [str(x)+":0.65" for x in result_grid["category"]],
        expected_sitey = [str(x)+":0.55" for x in result_grid["category"]],
        actual_sitey = [str(x)+":0.45" for x in result_grid["category"]],
        site_county = [str(x)+":0.35" for x in result_grid["category"]],
        signify = [str(x)+":0.25" for x in result_grid["category"]],
        impact_tot=["{0:.2f}".format(x) for x in result_grid["Est. Impact (total)"]],
        expected_tot=["{0:.2f}".format(x) for x in result_grid["Test expected (total)"]],
        actual_tot=["{0:.2f}".format(x) for x in result_grid["Test post-period (total)"]],
        symx=[str(x)+":0.5" for x in result_grid["metric"]],
        color = result_grid['text_color'],
        fill_color = [fill_colormap[x] for x in result_grid['significance level']],
        frame_color = [frame_colormap[x] for x in result_grid['significance level']]
        
    )
)

metric_index2 = [ 'Item - Average Selling Price' , 'Item - Net Quantity - All',
                'Item - Net Sales - All','Item - SPRS Margin - All']

group_range = [str(x) for x in range(1, 5)]
grid_plot = figure(title="Result Grid", tools="resize,hover,save",
       x_range= metric_index2 , y_range=['Week17_Blackout_Test_Items'])
grid_plot.plot_width = 1200
grid_plot.plot_height = 300
grid_plot.toolbar_location = None
grid_plot.outline_line_color = None

grid_plot.rect("metric","cat",0.9, 0.9, source=source, fill_color = "fill_color",
       fill_alpha=0.6, color="frame_color", line_width = 5)

text_props = {
    "source": source,
    "angle": 0,
    "text_align": "center",
    "text_baseline": "middle",
    "text_color": "color"
}

grid_plot.text(x="symx", y="lifty", text="lift",
       text_font_style="bold", text_font_size="20pt", **text_props)

grid_plot.text(x="symx", y="impact_sitey", text="impact_site",
       text_font_size="15pt", **text_props)

grid_plot.text(x="symx", y="expected_sitey", text="expected_site",
       text_font_size="12pt",  **text_props)
       
grid_plot.text(x="symx", y="actual_sitey", text="actual_site",
       text_font_size="12pt",  **text_props)

grid_plot.text(x="symx", y="site_county", text="site_count",
       text_font_size="12pt", **text_props)   
       
grid_plot.text(x="symx", y="signify", text="significance",
       text_font_size="12pt", **text_props)   

grid_plot.select_one(HoverTool).tooltips = [
    ("total impact", "@impact_tot"),
    ("total expected", "@expected_tot"),
    ("total actual", "@actual_tot")
]

color_list = ["#a6cee3",
              "#1f78b4",
              "#fdbf6f",
              "#b2df8a",
              "#33a02c",
              "#bbbb88",
              "#baa2a6",
              "#e08e79"]
# random color : "#%02x%02x%02x" % (np.random.randint(0,255) , np.random.randint(0,255), np.random.randint(0,255))

trend_plot = figure(x_axis_type = "datetime")
trend_plot.title = "Trend Chart"
trend_plot.grid.grid_line_alpha=0.3
trend_plot.xaxis.axis_label = 'Date'
trend_plot.yaxis.axis_label = 'Lift (test performance relative to control)'
trend_plot.plot_width = 1200


for i in xrange(len(metric_index)):
    date = trends[i].WK_NBR+3
    date = map(lambda x : datetime.datetime.strptime(str(x) + '-0','%Y%W-%w'), date)
    trend_plot.line(date , trends[i].Lift, color = color_list[i],
    legend = metric_index[i], line_width = 1.5, line_cap = 'round')

# Disruption period
"""
left_lim = Span(location=datetime.datetime.strptime(str(pre_period_to + 4) + '-0','%Y%W-%w'),
                              dimension='height', line_color='#dbdbdb',
                              line_dash='dashed', line_width=3)
                              
right_lim = Span(location=datetime.datetime.strptime(str(post_period_from + 2) + '-0','%Y%W-%w'),
                            dimension='height', line_color='#dbdbdb',
                            line_dash='dashed', line_width=3)                        

trend_plot.renderers.extend([left_lim, right_lim])

"""

#left = 
#right = 
#mid = (left + right)/2
#width = mid -left
#trend_plot.rect( left ,  500 , color = "#dbdbdb", alpha = 0.5, width_units="screen" )

output_file("grid_plot.html", title="grid plot")

show(vplot(grid_plot,trend_plot))
