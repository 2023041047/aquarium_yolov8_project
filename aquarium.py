"""YOLOv8 aquarium detection training, evaluation and video tracking/counting."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

CLASSES = ('shark', 'ray', 'whale_shark')


def load_yaml(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_dataset(root):
    from PIL import Image
    root = Path(root).resolve()
    config = load_yaml(root / 'data.yaml')
    names = config['names']
    names = [names[k] for k in sorted(names, key=lambda k: int(k))] if isinstance(names, dict) else names
    if [str(n).lower().replace(' ', '_') for n in names] != list(CLASSES):
        raise ValueError(f'Expected classes {CLASSES}, got {names}')
    counts = {}
    errors = []
    for split in ('train', 'val', 'test'):
        image_dir = root / 'images' / split
        label_dir = root / 'labels' / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f'Missing split folders: {split}')
            continue
        imgs = [p for p in image_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
        box_counts = Counter()
        image_counts = Counter()
        for img in imgs:
            label = label_dir / img.relative_to(image_dir).with_suffix('.txt')
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as exc:
                errors.append(f'Bad image {img}: {exc}')
                continue
            if not label.exists():
                errors.append(f'Missing label {label}')
                continue
            seen = set()
            for line_no, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError('expected five values (detection bbox)')
                    cls = int(fields[0])
                    coords = [float(v) for v in fields[1:]]
                    if cls not in range(3) or not all(math.isfinite(v) for v in coords):
                        raise ValueError('invalid class or non-finite coordinate')
                    if not all(0 <= v <= 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                        raise ValueError('invalid normalized bounding box')
                    if coords[0] - coords[2]/2 < -1e-5 or coords[0] + coords[2]/2 > 1+1e-5 or coords[1] - coords[3]/2 < -1e-5 or coords[1] + coords[3]/2 > 1+1e-5:
                        raise ValueError('box exceeds image')
                    box_counts[CLASSES[cls]] += 1
                    seen.add(cls)
                except Exception as exc:
                    errors.append(f'{label}:{line_no}: {exc}')
            for cls in seen:
                image_counts[CLASSES[cls]] += 1
        counts[split] = {'images': len(imgs), 'images_by_class': dict(image_counts), 'boxes_by_class': dict(box_counts)}
    result = {'classes': list(CLASSES), 'splits': counts, 'errors': errors[:100], 'error_count': len(errors)}
    (root / 'dataset_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(f'Dataset validation failed with {len(errors)} errors')
    return root


def ensure_model_classes(model):
    names = model.names
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    normalized = [str(n).lower().replace(' ', '_') for n in names]
    if normalized != list(CLASSES):
        raise ValueError(f'Model classes {normalized} do not match {CLASSES}. Train your own 3-class model first.')


def train(args):
    from ultralytics import YOLO
    root = validate_dataset(args.dataset)
    model = YOLO('yolov8n.pt')

    project_dir = Path('runs').resolve()
    experiment_name = f'aquarium_{args.epochs}epochs'

    model.train(data=str((root / 'data.yaml').resolve()), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=args.device, project=str(project_dir), name=experiment_name, exist_ok=False, workers=0)

                
    save_dir = Path(model.trainer.save_dir)
    best_model = save_dir / 'weights' / 'best.pt'

    print(f'Training directory: {save_dir}')
    print(f'Best model: {best_model}')
    print(f'Model exists: {best_model.exists()}')


def evaluate(args):
    from ultralytics import YOLO
    root = validate_dataset(args.dataset)
    model = YOLO(args.model)
    ensure_model_classes(model)
    metrics = model.val(data=str((root / 'data.yaml').resolve()), split='test', imgsz=args.imgsz,
                        device=args.device, project=str(Path('runs').resolve()), name='aquarium_test', exist_ok=True)

    result = {'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
              'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
              'per_class_mAP50_95': dict(zip(CLASSES, [float(v) for v in metrics.box.maps]))}

    out = Path('outputs')
    out.mkdir(exist_ok=True)
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def track(args):
    import cv2
    from ultralytics import YOLO
    model = YOLO(args.model)
    ensure_model_classes(model)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(3)), int(cap.get(4))
    if width <= 0 or height <= 0:
        raise RuntimeError('Invalid video dimensions')
    line_y = int(height * args.line)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot create video: {output}')
    history = defaultdict(lambda: deque(maxlen=30))
    last_side = {}
    counted = set()
    totals = Counter()
    events = []
    frame_no = 0
    max_frames = round(args.seconds * fps) if args.seconds > 0 else None
    colors = [(60, 210, 70), (230, 170, 50), (70, 90, 240)]
    try:
        while True:
            if max_frames is not None and frame_no >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf,
                                 imgsz=args.imgsz, device=args.device, verbose=False)[0]
            current = Counter()
            if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
                for box, cls, track_id, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                                    result.boxes.cls.int().cpu().tolist(),
                                                    result.boxes.id.int().cpu().tolist(),
                                                    result.boxes.conf.cpu().tolist()):
                    cls = int(cls)
                    if cls not in range(3):
                        continue
                    current[CLASSES[cls]] += 1
                    x1, y1, x2, y2 = [int(v) for v in box]
                    # Bottom center is a consistent anchor for line crossing.
                    cx, cy = (x1 + x2)//2, y2
                    key = (cls, track_id)
                    history[key].append((cx, cy))
                    side = -1 if cy < line_y else 1
                    previous = last_side.get(key)
                    if previous is not None and previous != side and key not in counted:
                        direction = 'down' if previous < side else 'up'
                        counted.add(key)
                        totals[(CLASSES[cls], direction)] += 1
                        events.append({'time_s': round(frame_no/fps, 3), 'frame': frame_no,
                                       'class': CLASSES[cls], 'track_id': track_id, 'direction': direction})
                    last_side[key] = side
                    color = colors[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{CLASSES[cls]} #{track_id} {conf:.2f}',
                                (max(0, x1), max(18, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    points = list(history[key])
                    for p, q in zip(points, points[1:]):
                        cv2.line(frame, p, q, color, 2)
            summary = ' | '.join(f'{name}: now {current[name]} / crossed {totals[(name,"up")]+totals[(name,"down")]}' for name in CLASSES)
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.putText(frame, summary, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, min(0.7, width/1800), (255,255,255), 2)
            writer.write(frame)
            frame_no += 1
            if args.show:
                cv2.imshow('Aquarium YOLOv8 tracking', frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['time_s','frame','class','track_id','direction'])
        w.writeheader()
        w.writerows(events)
    summary = {'model': str(args.model), 'video': str(args.source), 'processed_frames': frame_no,
               'seconds': round(frame_no/fps, 3), 'line_y_fraction': args.line,
               'counts': {name: {'up': totals[(name,'up')], 'down': totals[(name,'down')],
                                 'total': totals[(name,'up')] + totals[(name,'down')]} for name in CLASSES},
               'note': 'Counts are unique track IDs crossing a horizontal line once, not distinct animals in the aquarium.'}
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Saved: {output}, {csv_path}, {output.with_suffix(".json")}')


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 aquarium 3-class project')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('check', help='Validate YOLO dataset')
    p.add_argument('--dataset', default='dataset')

    p = sub.add_parser('train', help='Train YOLOv8n')
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')

    p = sub.add_parser('evaluate', help='Evaluate on test split')
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--model', default='runs/aquarium_5epochs/weights/best.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')

    p = sub.add_parser('track', help='Track and count objects in video')
    p.add_argument('--source', required=True)
    p.add_argument('--model', default='runs/aquarium_5epochs/weights/best.pt')
    p.add_argument('--output', default='outputs/aquarium_tracking_5sec.mp4')
    p.add_argument('--tracker', default='bytetrack.yaml')
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--line', type=float, default=0.55)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    p.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.command == 'check':
        validate_dataset(args.dataset)
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        if not 0 < args.line < 1:
            parser.error('--line must be between 0 and 1')
        track(args)


if __name__ == '__main__':
    main()
