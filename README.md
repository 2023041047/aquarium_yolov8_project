# YOLOv8 Aquarium: Shark / Ray / Whale Shark

Three mutually exclusive classes: `0 shark` (excluding whale sharks), `1 ray`, `2 whale_shark`.

## 1. Prepare dataset

Export a **YOLOv8 Object Detection** dataset from Roboflow. If the source is segmentation, convert to bounding-box labels before using this project. Put it under `dataset/` with `images/{train,val,test}` and `labels/{train,val,test}` and `data.yaml`. Set its `path:` to an absolute dataset directory (recommended on Windows), e.g. `D:/aquarium_yolov8_project/dataset`. Check source class indices and relabel them to 0/1/2; renaming YAML class labels alone is insufficient. Audit visually and respect licenses.

```powershell
python -m pip install -r requirements.txt
python aquarium.py check --dataset dataset
python aquarium.py train --dataset dataset --epochs 50 --batch 8 --device cpu
python aquarium.py evaluate --dataset dataset --device cpu
python aquarium.py track --source videos/aquarium_test5.mp4 --seconds 5 --device cpu --tracker bytetrack.yaml --show --model runs/aquarium_50epochs/weights/best.pt
```

Outputs: `runs/aquarium_5epochs/weights/best.pt`, `outputs/evaluation.json`, `outputs/aquarium_tracking_5sec.mp4`, `.csv` crossing events, `.json` counts.

### Video file upload notice
The original `aquarium_test5.mp4` video used for the final analysis is approximately **270 MB**. Because this exceeds GitHub's **25 MB browser-upload limit**, only an excerpt of the original video was uploaded to the repository. The uploaded clip is **not the full source video**; the final analysis was performed using the original video.

### Counting definition
A track counts only once when its bounding-box bottom-center crosses the horizontal line. `--line 0.55` positions it at 55% of frame height. This is a *line crossing count*, not the true number of distinct animals. A change in tracker ID or occlusion may affect counts. The first observed position alone does not count as crossing. Inspect the annotated video and compare with manual counts.

### Important
The pretrained COCO YOLOv8n weights do **not** include a whale-shark class. Do not use them as final inference results. You must train `best.pt` with the three classes above. No datasets, model weights, videos, or GitHub repository are included or claimed to have been created by these scripts.

### Reporting
Report dataset provenance/licenses, per-class image and box counts, split strategy, YOLOv8n hyperparameters, training loss plot, test precision/recall/mAP, tracking video, line-crossing counts, manual counting comparison, GitHub URL, and actual debugging experiences.
