#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:01:57 2021

@author: larsmaurath
"""
from os import listdir
import pandas as pd
import numpy as np

import cv2

from narya.tracker.full_tracker import FootballTracker

from mplsoccer.pitch import Pitch

import imageio
import progressbar

#%%

template = cv2.imread('world_cup_template.png')
template = cv2.resize(template, (512,512))/255.



#%%
img_list = []

cap = cv2.VideoCapture('clip.mp4')

cnt = 0

while cnt < 2*23:
#while cnt < 2:
    
    res, frame = cap.read()
    
    frame = cv2.resize(frame, (1024, 1024))
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img_list.append(frame)
    cnt += 1


#%% 

tracker = FootballTracker(pretrained=True, 
                          frame_rate=23,
                          track_buffer = 60,
                          ctx=None)

#%%

trajectories = tracker(img_list,
                       split_size = 512, 
                       save_tracking_folder = 'narya_output/', 
                       template = template, 
                       skip_homo = [])


#%%

trajectories_clean = [[key] + list(i) for key, value in trajectories.items() for i in value] 
trajectories_df = pd.DataFrame(trajectories_clean, columns=['player','x','y','frame', 'color'])
trajectories_df['x'] = trajectories_df['x']/320*120
trajectories_df['y'] = trajectories_df['y']/320*80
    
trajectories_df = trajectories_df.loc[trajectories_df['x'] >= 0]
trajectories_df = trajectories_df.loc[trajectories_df['y'] >= 0]

#%%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,label,center=cv2.kmeans(np.float32(trajectories_df['color'].to_list()),2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

trajectories_df['label'] = label

test = trajectories_df.groupby(['player']).apply(lambda x: pd.Series({'group': x.label.mode()[0]}))
test = test.reset_index()

trajectories_df = pd.merge(trajectories_df, test)

d = {0: 'blue', 1: 'green', 2: 'red', 3: 'black'}

trajectories_df['color_plot'] = trajectories_df['group'].map(d)

#%%

for frame in range(1, 2*23):

    trajectories_filter = trajectories_df.loc[trajectories_df['frame'] == frame]
    
    # colors = trajectories_filter['color']
    # colors = [tuple([item / 255 for item in subl]) for subl in colors]
    
    plot = Pitch(figsize=(6.8, 10.5))
    fig, ax = plot.draw()
    
    plot.scatter(trajectories_filter['x'], trajectories_filter['y'], ax=ax, c=trajectories_filter['color_plot'], s=150)
    
    fig.savefig(f'plot_output/plot_{frame}.png')

#%%

with imageio.get_writer('plot_output/movie.mp4', mode='I',fps=23) as writer:
    for frame in progressbar.progressbar(range(1, 2*23)):
        image = imageio.imread(f'plot_output/plot_{frame}.png')
        writer.append_data(image)
        
        
#%%
files = listdir('narya_output/')
files = [x for x in files if x != ".DS_Store"]
files = sorted(files)

with imageio.get_writer('narya_output/movie.mp4', mode='I',fps=23) as writer:
    for frame in progressbar.progressbar(files):
        image = imageio.imread('narya_output/' + frame)
        writer.append_data(image)        