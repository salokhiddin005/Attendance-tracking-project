import os
import json
import numpy as np
from tqdm import tqdm
from deepface import DeepFace

ENROLL_DIR = "data/enroll"
DB_PATH = "data/db/faces_db.npz"
PEOPLE_JSON = "people.json"

def main():

    # Check people.json exists
    if not os.path.exists(PEOPLE_JSON):
        raise FileNotFoundError("people.json not found in main folder.")

    with open(PEOPLE_JSON, "r", encoding="utf-8") as f:
        people = json.load(f)

    # Create db folder if it doesn't exist
    os.makedirs("data/db", exist_ok=True)

    embeddings = []
    labels = []

    # Loop through each person
    for person_id in people.keys():

        person_folder = os.path.join(ENROLL_DIR, person_id)

        if not os.path.isdir(person_folder):
            print(f"⚠ Folder missing: {person_folder}")
            continue

        images = [
            os.path.join(person_folder, img)
            for img in os.listdir(person_folder)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not images:
            print(f"⚠ No images found for {person_id}")
            continue

        for img_path in tqdm(images, desc=f"Processing {person_id}"):

            try:
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=True
                )

                embedding = np.array(rep[0]["embedding"], dtype=np.float32)

                embeddings.append(embedding)
                labels.append(person_id)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if not embeddings:
        raise RuntimeError("No embeddings created. Check your images.")

    np.savez(DB_PATH,
             embeddings=np.stack(embeddings),
             labels=np.array(labels))

    print("\n✅ Face database created successfully!")
    print(f"Saved to: {DB_PATH}")
    print(f"Total samples: {len(labels)}")


if __name__ == "__main__":
    main()