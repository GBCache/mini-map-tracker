'''
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

'''

import os
import math
from PIL import Image
import cv2
import numpy as np
import glob
import re
import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QColor


# Disable decompression bomb safety limit
Image.MAX_IMAGE_PIXELS = None

IMAGE_PATH = "map.png"   
TILES_DIR = "map_blocks" 
TILE_SIZE = 1000              
OVERLAP = 200                 # 20% overlap

NPZ_DIR = "map_blocks_npz"
STEP = TILE_SIZE - OVERLAP
FEATURES_PER_TILE = 1500 # ORB is lightning fast, so we can extract many features!

MINIMAP_VIEW_SIZE = 200     # The UI Box size
MINIMAP_CROP_AREA = 400     # The actual map area we capture
MOVE_SPEED = 20


def parse_coords_from_filename(filename, step_size):
    name = os.path.basename(filename).lower()
    m = re.search(r'_x(\d+)_y(\d+)', name)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r'_c(\d+)_r(\d+)', name)
    if m: return int(m.group(1)) * step_size, int(m.group(2)) * step_size
    nums = re.findall(r'\d+', name)
    if len(nums) >= 2: return int(nums[0]), int(nums[1])
    return None, None

def parse_coords(filename, step_size):
    name = os.path.basename(filename).lower()
    m = re.search(r'_x(\d+)_y(\d+)', name)
    if m: return int(m.group(1)), int(m.group(2))
    nums = re.findall(r'\d+', name)
    if len(nums) >= 2: return int(nums[0]), int(nums[1])
    return None, None

##################################################################################################33

def slice_map_with_overlap(image_path, output_folder, tile_size=1000, overlap=200):
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading huge map from '{image_path}'...")
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find {image_path}")
        return

    width, height = img.size
    print(f"Map Loaded! Dimensions: {width}x{height} px")
    
    # The step size is how much we move forward before cutting the next tile
    step_size = tile_size - overlap
    
    # Calculate approximate number of tiles for progress tracking
    cols = math.ceil((width - overlap) / step_size)
    rows = math.ceil((height - overlap) / step_size)
    total_tiles = cols * rows
    print(f"Generating approx {total_tiles} overlapping tiles...\n")

    count = 0
    y1 = 0
    while y1 < height:
        y2 = min(y1 + tile_size, height)
        
        x1 = 0
        while x1 < width:
            x2 = min(x1 + tile_size, width)
            
            # Crop the bounding box
            block = img.crop((x1, y1, x2, y2))
            
            filename = f"tile_{x1}_{y1}_{x2}_{y2}.jpg"
            output_path = os.path.join(output_folder, filename)
            
            # Save the tile
            block.save(output_path, quality=95)
            
            count += 1
            print(f"Saved {filename} ({count}/{total_tiles})".ljust(60), end="\r")
            
            # Move X forward by step_size
            if x2 == width: break
            x1 += step_size
            
        # Move Y forward by step_size
        if y2 == height: break
        y1 += step_size

    print("\n\n✅ Map slicing complete!")

##################################################################################################33

def extract_orb_to_npz():
    os.makedirs(NPZ_DIR, exist_ok=True)
    
    # Initialize ORB (Built specifically for Real-Time AR Tracking)
    orb = cv2.ORB_create(nfeatures=FEATURES_PER_TILE, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    
    tile_files = glob.glob(os.path.join(TILES_DIR, "*.*"))
    print(f"Extracting blazing-fast ORB features for {len(tile_files)} tiles...")
    
    for filepath in tile_files:
        if not filepath.lower().endswith(('.jpg', '.png')): continue
        x1, y1 = parse_coords_from_filename(filepath, STEP)
        if x1 is None: continue
            
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
            
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is None or len(keypoints) == 0: continue
            
        # Extract Global X, Y coordinates
        global_kps = np.array([[kp.pt[0] + x1, kp.pt[1] + y1] for kp in keypoints], dtype=np.float32)
        
        save_path = os.path.join(NPZ_DIR, f"data_x{x1}_y{y1}.npz")
        np.savez_compressed(save_path, kp=global_kps, des=descriptors)
        print(f"Saved {len(keypoints)} ORB features -> {os.path.basename(save_path)}")
        
    print("✅ Fast ORB generation complete! You can now run the viewer.")

##################################################################################################33


class RealTimeOrbWorker(QObject):
    resultReady = pyqtSignal(dict)
    
    def __init__(self, npz_dir, step_size):
        super().__init__()
        self.is_busy = False
        self.step_size = step_size
        
        self.orb = cv2.ORB_create(nfeatures=1000, edgeThreshold=15)
        # NORM_HAMMING is exactly what makes ORB 100x faster than SIFT
        self.warm_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.cold_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.all_data = {}
        self.cold_kps_list, self.cold_des_list = [],[]
        self.warm_kps, self.warm_des = None, None
        self.current_warm_base = (None, None)
        
        self._load_and_build_caches(npz_dir)

    def _load_and_build_caches(self, npz_dir):
        print("Worker: Loading ORB Database into RAM...")
        for filepath in glob.glob(os.path.join(npz_dir, "*.npz")):
            x, y = parse_coords(filepath, self.step_size)
            if x is not None:
                data = np.load(filepath)
                self.all_data[(x, y)] = {'kp': data['kp'], 'des': data['des']}
                
        # --- THE FIX FOR THE OpenCV IMGIDX_ONE CRASH ---
        # We chunk the massive map into safe 50,000 feature arrays to bypass the OpenCV bug
        cur_kps, cur_des, count = [],[], 0
        for data in self.all_data.values():
            cur_kps.append(data['kp'])
            cur_des.append(data['des'])
            count += len(data['kp'])
            if count >= 50000:
                self.cold_kps_list.append(np.vstack(cur_kps))
                self.cold_des_list.append(np.vstack(cur_des))
                cur_kps, cur_des, count = [],[], 0
                
        if cur_kps:
            self.cold_kps_list.append(np.vstack(cur_kps))
            self.cold_des_list.append(np.vstack(cur_des))
            
        if self.cold_des_list:
            self.cold_matcher.add(self.cold_des_list)
            
        print(f"Worker: Ready! Database chunked successfully to prevent crashes.")

    def _update_warm_cache(self, base_x, base_y):
        if self.current_warm_base == (base_x, base_y): return True
        kps, des = [],[]
        for dx in [-1, 0, 1]:
            for dy in[-1, 0, 1]:
                k = (base_x + (dx * self.step_size), base_y + (dy * self.step_size))
                if k in self.all_data:
                    kps.append(self.all_data[k]['kp'])
                    des.append(self.all_data[k]['des'])
                    
        if not kps: return False
        self.warm_kps = np.vstack(kps)
        self.warm_des = np.vstack(des)
        self.current_warm_base = (base_x, base_y)
        return True

    def run_analysis(self, packet):
        if self.is_busy: return
        self.is_busy = True
        start_time = time.time()
        
        minimap_img, base_x, base_y = packet.values()
        mini_gray = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
        mini_kp, mini_des = self.orb.detectAndCompute(mini_gray, None)

        if mini_des is None or len(mini_kp) < 5:
            self._emit_result(None, "Not enough terrain details", start_time)
            return

        is_warm = base_x is not None and self._update_warm_cache(base_x, base_y)
        status_prefix = "Warm Start" if is_warm else "Cold Start (Whole Map)"

        # Fast Hardware matching
        if is_warm:
            matches = self.warm_matcher.knnMatch(mini_des, self.warm_des, k=2)
        else:
            matches = self.cold_matcher.knnMatch(mini_des, k=2)
            
        good_matches = [m for m, n in matches if len(matches[0]) == 2 and m.distance < 0.75 * n.distance]

        if len(good_matches) >= 5:
            src_pts = np.float32([mini_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            if is_warm:
                dst_pts = np.float32([self.warm_kps[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
            else:
                dst_pts = np.float32([self.cold_kps_list[m.imgIdx][m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and mask.sum() >= 4:
                h, w = mini_gray.shape
                center = np.float32([[[w / 2.0, h / 2.0]]])
                global_pos = cv2.perspectiveTransform(center, M)
                gx, gy = global_pos[0][0]
                
                new_base_x = (int(gx) // self.step_size) * self.step_size
                new_base_y = (int(gy) // self.step_size) * self.step_size
                self._emit_result((gx, gy, new_base_x, new_base_y), f"{status_prefix}: Locked", start_time)
                return
                
        self._emit_result(None, f"{status_prefix}: Lost Tracking", start_time)

    def _emit_result(self, result, status, start_time):
        self.resultReady.emit({'result': result, 'status': status, 'time': time.time() - start_time})
        self.is_busy = False

class MainWindow(QMainWindow):
    requestAnalysis = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORB Real-Time Tracker (60 FPS Engine)")
        self.resize(1300, 750)
        
        self.tracked_base_x, self.tracked_base_y = None, None
        self.loaded_image_items = {}
        self.tiles =[]
        
        # --- FIX FOR HARD DRIVE LAG (RAM Cache) ---
        self.cached_tile_path = None
        self.cached_tile_img = None
        
        self.scan_image_tiles()
        self.auto_set_start_position()
        self.init_ui()
        self.setup_worker_thread()

        # UI Movement Loop (Runs 60 FPS flawlessly)
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(16)
        self.ui_timer.timeout.connect(self.update_player_view)
        self.ui_timer.start()

        # Tracking Request Loop (Runs 10 FPS)
        self.analysis_timer = QTimer(self)
        self.analysis_timer.setInterval(100) 
        self.analysis_timer.timeout.connect(self.request_tracking_frame)
        self.analysis_timer.start()
        
        # Track movement state
        self.keys_pressed = set()

    def setup_worker_thread(self):
        self.thread = QThread()
        self.worker = RealTimeOrbWorker(NPZ_DIR, STEP)
        self.worker.moveToThread(self.thread)
        self.requestAnalysis.connect(self.worker.run_analysis)
        self.worker.resultReady.connect(self.handle_tracker_result)
        self.thread.start()

    def scan_image_tiles(self):
        for filepath in glob.glob(os.path.join(TILES_DIR, "*.*")):
            if filepath.lower().endswith(('.jpg', '.png')):
                x, y = parse_coords(filepath, STEP)
                if x is not None: self.tiles.append({'x': x, 'y': y, 'path': filepath})

    def auto_set_start_position(self):
        if not self.tiles: return
        first = sorted(self.tiles, key=lambda t: (t['x'], t['y']))[0]
        self.sim_gx = 3000
        self.sim_gy = 6500
        
    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central); layout = QHBoxLayout(central)
        left = QVBoxLayout(); left.setAlignment(Qt.AlignTop)
        
        self.minimap_lbl = QLabel(); self.minimap_lbl.setFixedSize(MINIMAP_VIEW_SIZE, MINIMAP_VIEW_SIZE)
        self.minimap_lbl.setStyleSheet("border: 2px solid cyan; background-color: black;")
        
        left.addWidget(QLabel("Minimap View (WASD to Move)")); left.addWidget(self.minimap_lbl)
        self.file_lbl = QLabel("Current File:\n..."); left.addWidget(self.file_lbl)
        self.actual_lbl = QLabel("True Coords:\nX: -- | Y: --"); left.addWidget(self.actual_lbl)
        self.guessed_lbl = QLabel("Tracked Coords:\nX: -- | Y: --"); left.addWidget(self.guessed_lbl)
        self.status_lbl = QLabel("Status: Initializing..."); left.addWidget(self.status_lbl)
        layout.addLayout(left, 1)
        
        right = QVBoxLayout(); right.addWidget(QLabel("Global Map Engine"))
        self.view = QGraphicsView(); self.scene = QGraphicsScene(); self.view.setScene(self.scene)
        self.view.setBackgroundBrush(Qt.black); right.addWidget(self.view); layout.addLayout(right, 4)
        
        self.marker = QGraphicsEllipseItem(-10, -10, 20, 20)
        self.marker.setZValue(999); self.scene.addItem(self.marker)

    def keyPressEvent(self, event):
        self.keys_pressed.add(event.key())

    def keyReleaseEvent(self, event):
        if event.key() in self.keys_pressed:
            self.keys_pressed.remove(event.key())

    def get_tile_image(self, t):
        if not t: return None
        # Never touch the hard drive if we already have the image!
        if self.cached_tile_path != t['path']:
            self.cached_tile_img = cv2.imread(t['path'])
            self.cached_tile_path = t['path']
        return self.cached_tile_img

    def get_centered_crop(self, img, center_x, center_y, t):
        """Creates a perfect crop where the player is ALWAYS dead center"""
        h, w = img.shape[:2]
        half = MINIMAP_CROP_AREA // 2
        
        x1, y1 = int(center_x - t['x'] - half), int(center_y - t['y'] - half)
        x2, y2 = x1 + MINIMAP_CROP_AREA, y1 + MINIMAP_CROP_AREA
        
        img_x1, img_y1 = max(0, x1), max(0, y1)
        img_x2, img_y2 = min(w, x2), min(h, y2)
        
        extracted = img[img_y1:img_y2, img_x1:img_x2]
        
        canvas = np.zeros((MINIMAP_CROP_AREA, MINIMAP_CROP_AREA, 3), dtype=np.uint8)
        dest_x1, dest_y1 = img_x1 - x1, img_y1 - y1
        canvas[dest_y1:dest_y1+(img_y2-img_y1), dest_x1:dest_x1+(img_x2-img_x1)] = extracted
        return canvas

    def update_player_view(self):
        if Qt.Key_W in self.keys_pressed: self.sim_gy -= MOVE_SPEED
        if Qt.Key_S in self.keys_pressed: self.sim_gy += MOVE_SPEED
        if Qt.Key_A in self.keys_pressed: self.sim_gx -= MOVE_SPEED
        if Qt.Key_D in self.keys_pressed: self.sim_gx += MOVE_SPEED

        t = next((t for t in self.tiles if t['x'] <= self.sim_gx < t['x']+TILE_SIZE and t['y'] <= self.sim_gy < t['y']+TILE_SIZE), None)
        img = self.get_tile_image(t)
        
        if img is not None:
            large_crop = self.get_centered_crop(img, self.sim_gx, self.sim_gy, t)
            visual_crop = cv2.resize(large_crop, (MINIMAP_VIEW_SIZE, MINIMAP_VIEW_SIZE), interpolation=cv2.INTER_AREA)
            
            # Draw the player dot perfectly in the center
            cv2.circle(visual_crop, (MINIMAP_VIEW_SIZE//2, MINIMAP_VIEW_SIZE//2), 5, (0, 255, 255), -1)
            filename = os.path.basename(t['path'])
        else:
            visual_crop = np.zeros((MINIMAP_VIEW_SIZE, MINIMAP_VIEW_SIZE, 3), np.uint8)
            filename = "OUT OF BOUNDS"
                
        rgb_mini = cv2.cvtColor(visual_crop, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_mini.data, MINIMAP_VIEW_SIZE, MINIMAP_VIEW_SIZE, MINIMAP_VIEW_SIZE * 3, QImage.Format_RGB888)
        self.minimap_lbl.setPixmap(QPixmap.fromImage(qimg))
        
        self.actual_lbl.setText(f"True Coords:\nX: {int(self.sim_gx)} | Y: {int(self.sim_gy)}")
        self.file_lbl.setText(f"Current File:\n{filename}")

    def request_tracking_frame(self):
        t = next((t for t in self.tiles if t['x'] <= self.sim_gx < t['x']+TILE_SIZE and t['y'] <= self.sim_gy < t['y']+TILE_SIZE), None)
        img = self.get_tile_image(t)
        if img is None: return

        clean_crop = self.get_centered_crop(img, self.sim_gx, self.sim_gy, t)
        self.requestAnalysis.emit({'img': clean_crop, 'base_x': self.tracked_base_x, 'base_y': self.tracked_base_y})

    def handle_tracker_result(self, packet):
        result, status, elapsed = packet.values()
        fps = 1 / elapsed if elapsed > 0 else 0
        
        color = "green" if "Locked" in status else ("orange" if "Cold" in status else "red")
        self.status_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.status_lbl.setText(f"Status: {status}\nAlgo Time: {elapsed:.3f}s ({fps:.0f} FPS)")
        
        if result:
            gx, gy, new_base_x, new_base_y = result
            self.guessed_lbl.setText(f"Tracked Coords:\nX: {int(gx)} | Y: {int(gy)}")

            if new_base_x != self.tracked_base_x or new_base_y != self.tracked_base_y:
                self.dynamically_load_map_tiles(new_base_x, new_base_y)
                self.tracked_base_x, self.tracked_base_y = new_base_x, new_base_y

            self.marker.setPos(gx, gy)
            self.view.centerOn(self.marker) 
        else:
            self.guessed_lbl.setText("Tracked Coords:\nSeeking...")
            self.tracked_base_x = None

    def dynamically_load_map_tiles(self, center_x, center_y):
        needed_keys, keys_to_remove = set(), []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                needed_keys.add((center_x + (dx * STEP), center_y + (dy * STEP)))
                
        for k in self.loaded_image_items:
            if k not in needed_keys: keys_to_remove.append(k)
        for k in keys_to_remove:
            self.scene.removeItem(self.loaded_image_items.pop(k))
            
        for t in self.tiles:
            key = (t['x'], t['y'])
            if key in needed_keys and key not in self.loaded_image_items:
                self.loaded_image_items[key] = self.scene.addPixmap(QPixmap(t['path']))
                self.loaded_image_items[key].setPos(t['x'], t['y'])

    def closeEvent(self, event):
        self.thread.quit(); self.thread.wait(); event.accept()

if __name__ == "__main__":

    if not os.path.exists(TILES_DIR):
        print('========CHUNKING MAP========')    
        slice_map_with_overlap(IMAGE_PATH, TILES_DIR, TILE_SIZE, OVERLAP)

    if not os.path.exists(NPZ_DIR):  
        print('========EXTRACTING FEATURES========')
        extract_orb_to_npz()

    print('========RUNNING VIEWER========')
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())