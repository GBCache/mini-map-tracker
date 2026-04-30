# mini-map-tracker
Track global position of a player using mini-map image and a global map high resolution image

youtube video
https://www.youtube.com/watch?v=M_gLQX5ssgg

Script to track player position from minimap of any game to global map of the game.

Subscribe to my youtube channel https://www.youtube.com/@GBCache

Can download genshin map high resolution image from
https://www.hoyolab.com/article/21013856
https://drive.google.com/file/d/1o0K73x9cMqjkpLav1My387ItJzO9sfdX/view?usp=drive_link

1- Use any high resolution full map image from any game and name the image as map.png in the same folder as script
2- Run this script python map_viewer.py
The script have 3 phases
a- Breaking full map in smaller chunks
b- Extracting features of those chunks
c- Loading those chunks to construct minimap and full map, and using ORB to track the position of player in minimap to global map
