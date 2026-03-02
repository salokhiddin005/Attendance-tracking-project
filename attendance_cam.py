import os
import json
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

DB_PATH = "data/db/faces_db.npz"
PEOPLE_JSON = "people.json"
CSV_PATH = "attendance.csv"

ON_TIME_HOUR = 9
ON_TIME_MINUTE = 0

ACCEPT_THRESHOLD = 0.35
COOLDOWN_SECONDS = 30


def is_on_time(now: datetime) -> bool:
    deadline = now.replace(hour=ON_TIME_HOUR, minute=ON_TIME_MINUTE, second=0, microsecond=0)
    return now <= deadline


def ensure_csv():
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=[
            "datetime", "person_key", "id_number",
            "full_name", "status", "score", "note"
        ])
        df.to_csv(CSV_PATH, index=False)


def load_database():
    data = np.load(DB_PATH, allow_pickle=True)
    return data["embeddings"], data["labels"]


def cosine_similarity(a, b) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def recognize(embedding, db_embs, db_labels):
    sims = [cosine_similarity(embedding, e) for e in db_embs]
    best_idx = int(np.argmax(sims))
    return str(db_labels[best_idx]), float(sims[best_idx])


def sim_to_percent(sim: float) -> int:
    sim = max(0.0, min(1.0, sim))
    return int(round(sim * 100))


def log_attendance(people, person_key: str, score: float, note: str):
    now = datetime.now()
    status = "on_time" if is_on_time(now) else "late"

    full_name = people.get(person_key, {}).get("full_name", person_key)
    id_number = people.get(person_key, {}).get("id_number", "")

    row = {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "person_key": person_key,
        "id_number": id_number,
        "full_name": full_name,
        "status": status,
        "score": round(score, 4),
        "note": note
    }

    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    print("✅ Logged:", row)


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Run enroll.py first.")

    with open(PEOPLE_JSON, "r", encoding="utf-8") as f:
        people = json.load(f)

    ensure_csv()
    db_embs, db_labels = load_database()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opening.")

    last_logged = {}

    print("🎥 Attendance system started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cards = []

        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False
            )
        except:
            faces = []

        for item in faces:
            face_img = item.get("face", None)
            area = item.get("facial_area", {}) or {}

            x = int(area.get("x", 0))
            y = int(area.get("y", 0))
            w = int(area.get("w", 0))
            h = int(area.get("h", 0))

            if face_img is None or w <= 0 or h <= 0:
                continue

            try:
                rep = DeepFace.represent(
                    img_path=face_img,
                    model_name="ArcFace",
                    detector_backend="skip",
                    enforce_detection=False
                )
                emb = np.array(rep[0]["embedding"], dtype=np.float32)
            except:
                continue

            person_key, sim = recognize(emb, db_embs, db_labels)
            percent = sim_to_percent(sim)

            accepted = (sim >= ACCEPT_THRESHOLD) and (person_key in people)

            if accepted:
                full_name = people[person_key]["full_name"]
                id_number = people[person_key].get("id_number", "")
                status = "on_time" if is_on_time(datetime.now()) else "late"

                cards.append({
                    "id": id_number,
                    "name": full_name,
                    "status": status,
                    "percent": percent
                })

                now_ts = time.time()
                last_ts = last_logged.get(person_key, 0)
                if now_ts - last_ts >= COOLDOWN_SECONDS:
                    log_attendance(people, person_key, sim, note="arrival_detected")
                    last_logged[person_key] = now_ts

                color = (0, 255, 0)
                label = f"{full_name} {percent}%"
            else:
                color = (0, 0, 255)
                label = f"Unknown {percent}%"

            # Rectangle bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(20, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # -------- ULTRA COMPACT TOP-LEFT PANELS --------
        seen_ids = set()
        unique_cards = []
        for c in cards:
            if c["id"] not in seen_ids:
                unique_cards.append(c)
                seen_ids.add(c["id"])

        start_x = 8
        start_y = 15

        for idx, c in enumerate(unique_cards[:5]):
            px = start_x
            py = start_y + idx * 45
            pw = 180
            ph = 38

            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), 1)

            font_scale = 0.35
            thickness = 1

            line1 = f"{c['id']} | {c['name']}"
            line2 = f"{c['percent']}% {c['status']}"

            cv2.putText(frame, line1, (px + 5, py + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            cv2.putText(frame, line2, (px + 5, py + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        cv2.imshow("Attendence System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()