import os
import cv2
import random
import numpy as np

IMAGES_PER_CLASS = 500
CANVAS_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = r"C:\Users\mahim\OneDrive\digit\merge_dataset\merged_test"

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "dataset_2digit",
    "test"
)


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for label in range(100):
    os.makedirs(
        os.path.join(OUTPUT_FOLDER, f"{label:02d}"),
        exist_ok=True
    )

print("Folders Created!")

digit_images = {}

for digit in range(10):

    folder = os.path.join(INPUT_FOLDER, str(digit))

    files = []

    for file in os.listdir(folder):

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            files.append(os.path.join(folder, file))

    digit_images[digit] = files

print("Images Loaded!")


for number in range(100):

    tens = number // 10
    ones = number % 10

    save_folder = os.path.join(
        OUTPUT_FOLDER,
        f"{number:02d}"
    )

    print(f"Creating {number:02d}")

    for i in range(IMAGES_PER_CLASS):

        img1_path = random.choice(digit_images[tens])
        img2_path = random.choice(digit_images[ones])

        img1 = cv2.imread(
            img1_path,
            cv2.IMREAD_GRAYSCALE
        )

        img2 = cv2.imread(
            img2_path,
            cv2.IMREAD_GRAYSCALE
        )

        img1 = cv2.resize(img1, (28, 28))
        img2 = cv2.resize(img2, (28, 28))

        canvas = np.zeros(
            (CANVAS_SIZE, CANVAS_SIZE),
            dtype=np.uint8
        )

        gap = random.randint(2, 8)

        total_width = 28 + gap + 28

        start_x = (CANVAS_SIZE - total_width) // 2

        y1 = random.randint(16, 22)
        y2 = random.randint(16, 22)

        x1 = start_x
        x2 = start_x + 28 + gap

        canvas[
            y1:y1+28,
            x1:x1+28
        ] = np.maximum(
            canvas[
                y1:y1+28,
                x1:x1+28
            ],
            img1
        )

        canvas[
            y2:y2+28,
            x2:x2+28
        ] = np.maximum(
            canvas[
                y2:y2+28,
                x2:x2+28
            ],
            img2
        )


        filename = os.path.join(
            save_folder,
            f"{i:06d}.png"
        )

        cv2.imwrite(
            filename,
            canvas
        )

print("\nDataset Creation Completed Successfully!")