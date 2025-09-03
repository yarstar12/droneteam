import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
import folium
from PIL import Image, ImageDraw, ImageFont

# Параметры
VIDEO_PATH = "0530.mp4"
FRAME_SKIP = 2
RESOLUTION = (1920, 1080)
EDGE_WIDTH = 0.1
MAX_KEYPOINTS = 100
MAP_SIZE = (600, 600)
MIN_FLOW_THRESHOLD = 0.5
ALPHA = 0.05
MIN_FRAMES_BETWEEN_PERSONS = 40
RECENT_MOVEMENTS_SIZE = 10
INFO_WINDOW_SIZE = (600, 150)
MIN_CONFIDENCE = 0.52
MAX_FLOW_MAGNITUDE = 20.0
HIGHLIGHT_DISTANCE_THRESHOLD = 10.0

# ORB
orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS)

# Параметры Lucas–Kanade
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

map_center = np.array([MAP_SIZE[0]//2, MAP_SIZE[1]//2], dtype=np.float32)
start_pos = np.array([0.0, 0.0], dtype=np.float32)

# Функция для создания интерактивной карты с folium
def create_folium_map(trajectory, person_positions):
    m = folium.Map(location=[0, 0], zoom_start=15, tiles="OpenStreetMap")
    if trajectory:
        traj_points = [[p[0], p[1]] for p, _ in trajectory]
        folium.PolyLine(traj_points, color="blue", weight=2.5, opacity=1, popup="Траектория").add_to(m)

    folium.Marker(location=[0, 0], popup="Старт", icon=folium.Icon(color="green")).add_to(m)
    if trajectory:
        last_point = [trajectory[-1][0][0], trajectory[-1][0][1]]
        folium.Marker(location=last_point, popup="Финиш", icon=folium.Icon(color="red")).add_to(m)

    for idx, (pp, _, conf, _) in enumerate(person_positions):
        popup_text = f"Человек {idx + 1}<br>Вероятность: {conf:.2f}"
        folium.Marker(location=[pp[0], pp[1]], popup=popup_text, icon=folium.Icon(color="yellow")).add_to(m)

    m.save("map.html")

# Функция для создания информационного окна
def create_info_window(person_positions, curr_pos, frame_idx):
    info_window = Image.new('RGBA', (INFO_WINDOW_SIZE[0], INFO_WINDOW_SIZE[1]), (255, 255, 255, 0))
    draw = ImageDraw.Draw(info_window)

    for y in range(INFO_WINDOW_SIZE[1]):
        alpha = int(200 * (1 - y / INFO_WINDOW_SIZE[1]))
        draw.line((0, y, INFO_WINDOW_SIZE[0], y), fill=(240, 240, 245, alpha))

    draw.rectangle((2, 2, INFO_WINDOW_SIZE[0]-3, INFO_WINDOW_SIZE[1]-3), outline=(100, 100, 100), width=2)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            print("Ошибка: Шрифт Arial или DejaVuSans не найден.")
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

    draw.text((10, 10), "Обнаруженные люди:", font=font, fill=(0, 0, 0))
    y_offset = 40
    if person_positions:
        for idx, (pp, _, conf, _) in enumerate(person_positions[:2]):
            distance_from_start = np.linalg.norm(start_pos - pp)
            distance_from_drone = np.linalg.norm(curr_pos - pp)
            text = f"Человек {idx + 1}: От старта: {distance_from_start:.2f} м, От дрона: {distance_from_drone:.2f} м, Вер: {conf:.2f}"
            draw.text((10, y_offset), text, font=font_small, fill=(0, 0, 0))
            y_offset += 25
    else:
        draw.text((10, y_offset), "Люди не обнаружены", font=font_small, fill=(0, 0, 0))

    info_window = info_window.convert('RGB')
    return np.array(info_window)

# Функция для вычисления потока по краям
def compute_edge_flow(prev_gray, curr_gray, h, w):
    edge_w = int(EDGE_WIDTH * min(h, w))
    zones_prev = [
        prev_gray[:edge_w, :],
        prev_gray[-edge_w:, :],
        prev_gray[:, :edge_w],
        prev_gray[:, -edge_w:],
        prev_gray[h//4:3*h//4, w//4:3*w//4]
    ]
    zones_curr = [
        curr_gray[:edge_w, :],
        curr_gray[-edge_w:, :],
        curr_gray[:, :edge_w],
        curr_gray[:, -edge_w:],
        curr_gray[h//4:3*h//4, w//4:3*w//4]
    ]

    flows, stdevs = [], []
    for pz, cz in zip(zones_prev, zones_curr):
        p0 = cv2.goodFeaturesToTrack(pz, maxCorners=20, qualityLevel=0.01, minDistance=10)
        if p0 is None or len(p0) == 0:
            flows.append(np.zeros(2, np.float32))
            stdevs.append(np.full(2, np.inf, np.float32))
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(pz, cz, p0, None, **lk_params)
        if p1 is None or st is None or len(p1) != len(p0):
            flows.append(np.zeros(2, np.float32))
            stdevs.append(np.full(2, np.inf, np.float32))
            continue

        good = st.ravel() == 1
        if np.sum(good) == 0:
            flows.append(np.zeros(2, np.float32))
            stdevs.append(np.full(2, np.inf, np.float32))
            continue

        p0g, p1g = p0[good], p1[good]
        if len(p0g) == 0 or len(p1g) == 0:
            flows.append(np.zeros(2, np.float32))
            stdevs.append(np.full(2, np.inf, np.float32))
            continue

        diffs = p1g - p0g
        flow_mean = np.mean(diffs, axis=0)
        flows.append(flow_mean.astype(np.float32))
        stdevs.append(np.std(diffs, axis=0).astype(np.float32))

    flows = np.array(flows, dtype=np.float32)
    stdevs = np.array(stdevs, dtype=np.float32)
    reliab = np.nan_to_num(1.0 / (1.0 + np.linalg.norm(stdevs, axis=1)))
    if reliab.sum() > 0:
        weighted = (flows * reliab[:, None]).sum(axis=0) / reliab.sum()
    else:
        weighted = np.zeros(2, np.float32)
    return -weighted

# Функция для вычисления потока с помощью ORB
def compute_orb_flow(prev_gray, curr_gray):
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None:
        return np.zeros(2, np.float32), 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    if len(matches) < 10:
        return np.zeros(2, np.float32), 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is not None:
        in1 = pts1[mask.ravel()==1]
        in2 = pts2[mask.ravel()==1]
    else:
        in1, in2 = pts1, pts2

    if len(in1) >= 5:
        flow = np.mean(in2 - in1, axis=0)
        angle = np.arctan2(H[1,0], H[0,0]) * 180.0 / np.pi if H is not None else 0.0
    else:
        flow = np.mean(pts2 - pts1, axis=0)
        angle = 0.0

    return -flow.astype(np.float32), -angle

# Функция для отрисовки карты
def draw_map(trajectory, current_angle, person_positions):
    img = np.full((MAP_SIZE[1], MAP_SIZE[0], 3), (230, 230, 230), dtype=np.uint8)
    if len(trajectory) < 2:
        return img

    pts = np.array([p for p, _ in trajectory], np.float32)
    w = 20
    sm = np.zeros_like(pts)
    for i in range(len(pts)):
        s = max(0, i - w // 2)
        e = min(len(pts), i + w // 2 + 1)
        sm[i] = pts[s:e].mean(axis=0)
    
    mn, mx = sm.min(0), sm.max(0)
    ext = np.maximum(mx - mn, 10.0)
    scale = min(MAP_SIZE[0] / ext[0], MAP_SIZE[1] / ext[1]) * 0.9
    off = map_center - (mn + mx) / 2 * scale

    draw_pts = [(off + p * scale).astype(int) for p in sm[::2]]
    draw_pts = np.clip(draw_pts, 0, MAP_SIZE[0] - 1)
    if len(draw_pts) > 1:
        for i in range(len(draw_pts) - 1):
            cv2.line(img, tuple(draw_pts[i]), tuple(draw_pts[i + 1]), (255, 90, 0), 6, cv2.LINE_AA)

    start = tuple(np.clip((off + sm[0] * scale).astype(int), 0, MAP_SIZE[0] - 1))
    end = tuple(np.clip((off + sm[-1] * scale).astype(int), 0, MAP_SIZE[0] - 1))
    cv2.circle(img, start, 15, (30, 255, 0), -1)
    cv2.circle(img, end, 15, (4, 0, 255), -1)

    for idx, (pp, _, conf, highlighted) in enumerate(person_positions):
        draw_pp = (off + pp * scale).astype(int)
        draw_pp = np.clip(draw_pp, 0, MAP_SIZE[0] - 1)
        color = (255, 0, 0) if highlighted else (0, 255, 255)
        cv2.circle(img, tuple(draw_pp), 10, color, -1)
        cv2.putText(img, f"{idx + 1}", (draw_pp[0] + 15, draw_pp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img

# Функция для сохранения 3D-графика
def save_3d_plot(trajectory, person_positions):
    if len(trajectory) < 2:
        return

    x = [p[0] for p, _ in trajectory]
    y = [p[1] for p, _ in trajectory]
    angles = [a for _, a in trajectory]

    fig = plt.figure(figsize=(10, 8), facecolor=(230/255, 230/255, 230/255))
    ax = fig.add_subplot(111, projection='3d', facecolor=(230/255, 230/255, 230/255))
    ax.plot(x, y, angles, color='#005AFF', linewidth=4, alpha=0.8)
    ax.scatter(x[0], y[0], angles[0], c='#00FF1E', s=200)
    ax.scatter(x[-1], y[-1], angles[-1], c='#FF0004', s=200)

    if person_positions:
        px = [p[0] for p, _, _, _ in person_positions]
        py = [p[1] for p, _, _, _ in person_positions]
        pa = [a for _, a, _, _ in person_positions]
        pc = [conf for _, _, conf, _ in person_positions]
        scatter = ax.scatter(px, py, pa, c='yellow', s=100, marker='*')
        for i, conf in enumerate(pc):
            ax.text(px[i], py[i], pa[i], f"{i + 1}", color='black', fontsize=8)

    ax.grid(False)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='z', colors='black')
    plt.savefig('trajectory_3d.png', facecolor=(230/255, 230/255, 230/255), edgecolor='none', dpi=300)
    plt.close()

# Основная функция
def main():
    try:
        model = YOLO('Dlya_video.pt')
        person_class_id = [k for k, v in model.names.items() if v == 'person'][0]
    except Exception as e:
        print(f"Ошибка загрузки модели YOLO: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    ret, prev = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        cap.release()
        return
    prev = cv2.resize(prev, RESOLUTION, interpolation=cv2.INTER_CUBIC)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    traj = deque(maxlen=10000)
    person_positions = []  # (position, angle, confidence, highlighted)
    people_info = []
    recent_movements = deque(maxlen=RECENT_MOVEMENTS_SIZE)
    curr_pos = np.zeros(2, np.float32)
    curr_ang = 0.0
    smoothed_pos = np.zeros(2, np.float32)
    frame_idx = 0
    last_person_frame = -MIN_FRAMES_BETWEEN_PERSONS

    cv2.namedWindow("Полет дрона", cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('drone_flight_with_map.mp4', fourcc, 30.0 / FRAME_SKIP, (1920, 1080))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, RESOLUTION, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        ef = compute_edge_flow(prev_gray, gray, h, w)
        of, oa = compute_orb_flow(prev_gray, gray)

        if np.linalg.norm(of) > 1e-3:
            flow = (0.7*of + 0.3*ef).flatten()
            angle = oa
        else:
            flow = ef.flatten()
            angle = 0.0

        flow_magnitude = np.linalg.norm(flow)
        if flow_magnitude > MAX_FLOW_MAGNITUDE:
            flow = flow / flow_magnitude * MAX_FLOW_MAGNITUDE
        if flow_magnitude < MIN_FLOW_THRESHOLD:
            flow = np.zeros(2, np.float32)
            angle = 0.0

        smoothed_pos = (1 - ALPHA) * smoothed_pos + ALPHA * (curr_pos + flow)
        curr_pos = smoothed_pos.copy()
        curr_ang += angle
        traj.append((curr_pos.copy(), curr_ang))
        recent_movements.append(np.linalg.norm(flow))

        # Проверка, проходит ли траектория через людей
        for i, (pp, angle, conf, highlighted) in enumerate(person_positions):
            if not highlighted:
                distance = np.linalg.norm(curr_pos - pp)
                print(f"Кадр {frame_idx}, Человек {i+1}, расстояние до дрона: {distance:.2f}")  # Отладочный вывод
                if distance < HIGHLIGHT_DISTANCE_THRESHOLD:
                    person_positions[i] = (pp, angle, conf, True)
                    print(f"Человек {i+1} подсвечен на кадре {frame_idx}")

        results = model(frame)
        detections = results[0].boxes
        person_detected = False
        if len(detections) > 0:
            person_detected = (detections.cls == person_class_id).any().item() and any(detections.conf >= MIN_CONFIDENCE)

        if person_detected and (frame_idx - last_person_frame) >= MIN_FRAMES_BETWEEN_PERSONS:
            added_people = 0
            for box in detections:
                if (box.cls == person_class_id).item() and box.conf >= MIN_CONFIDENCE:
                    selected_conf = box.conf.item()
                    new_pos = curr_pos.copy()
                    person_positions.append((new_pos, curr_ang, selected_conf, False))
                    distance_from_start = np.linalg.norm(start_pos - new_pos)
                    people_info.append({
                        'frame_idx': frame_idx,
                        'position': new_pos,
                        'angle': curr_ang,
                        'distance_from_start': distance_from_start,
                        'confidence': selected_conf
                    })
                    added_people += 1
                    print(f"Человек {len(people_info)} добавлен на кадре {frame_idx}, позиция: {new_pos}, расстояние от старта: {distance_from_start:.2f} м, вероятность: {selected_conf:.2f}")
            if added_people > 0:
                last_person_frame = frame_idx

        for box in detections:
            if (box.cls == person_class_id).item() and box.conf >= MIN_CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, "person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        info_window = create_info_window(person_positions, curr_pos, frame_idx)
        frame[RESOLUTION[1]-INFO_WINDOW_SIZE[1]-30:RESOLUTION[1]-30, 30:30+INFO_WINDOW_SIZE[0]] = info_window

        map_vis = draw_map(traj, curr_ang, person_positions)
        map_small = cv2.resize(map_vis, (400, 400))
        combined = np.zeros((1080, 1920, 3), dtype=np.uint8)
        combined[:1080, :1920] = frame
        combined[1080-400:, 1920-400:] = map_small
        cv2.rectangle(combined, (1920-400-5, 1080-400-5), (1920+5, 1080+5), (255, 255, 255), 2)

        cv2.imshow("Полет дрона", combined)
        out.write(combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray.copy()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    cv2.imwrite("trajectory_map.png", map_vis)
    save_3d_plot(traj, person_positions)
    create_folium_map(traj, person_positions)
    with open("trajectory.txt", "w") as f:
        for p, a in traj:
            f.write(f"Позиция: {p}, Угол: {a}\n")
    with open("people_info.txt", "w") as f:
        for person in people_info:
            f.write(f"Человек на кадре {person['frame_idx']}: Позиция: {person['position']}, Расстояние от старта: {person['distance_from_start']:.2f} м, Угол: {person['angle']}, Вероятность: {person['confidence']:.2f}\n")

if __name__ == "__main__":
    main()