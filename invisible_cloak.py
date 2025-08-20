import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1.0)

background = None
show_mask = False

print("[INFO] Controls:")
print("  b = capture background (stand out of frame)")
print("  m = show/hide mask window")
print("  q = quit")

def capture_smooth_background(cap, samples=60, delay=0.01):
    frames = []
    for _ in range(samples):
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.flip(f, 1)
        frames.append(f)
        time.sleep(delay)
    if not frames:
        return None
    bg = np.median(np.array(frames), axis=0).astype(np.uint8)
    return cv2.GaussianBlur(bg, (5, 5), 0)

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] Camera frame not received.")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- HSV ranges for GREEN cloak ---
    lower_green = np.array([35, 100, 100], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- Clean mask ---
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    inv = cv2.bitwise_not(mask)

    visible_part = cv2.bitwise_and(frame, frame, mask=inv)

    if background is not None:
        cloak_part = cv2.bitwise_and(background, background, mask=mask)
        final = cv2.addWeighted(visible_part, 1.0, cloak_part, 1.0, 0.0)
    else:
        final = frame

    cv2.imshow("Invisible Cloak", final)
    if show_mask:
        cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        print("[INFO] Capturing background... please step out of the frame.")
        time.sleep(0.5)
        background = capture_smooth_background(cap)
        if background is not None:
            print("✅ Background captured!")
        else:
            print("❌ Failed to capture background.")
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            try:
                cv2.destroyWindow("Mask")
            except:
                pass
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
