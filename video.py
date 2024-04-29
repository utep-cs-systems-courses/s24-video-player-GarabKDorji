import cv2
import threading
import queue
import os
import numpy as np
import base64

class BlockingQueue:
    def __init__(self, capacity):
        self.queue = queue.Queue(capacity)
        self.empty = threading.Semaphore(capacity)  # Semaphore to track empty slots
        self.full = threading.Semaphore(0)          # Semaphore to track filled slots
        self.lock = threading.Lock()                # Lock for mutual exclusion

    def put(self, item):
        self.empty.acquire()
        self.lock.acquire()
        self.queue.put(item)
        self.lock.release()
        self.full.release()

    def get(self):
        self.full.acquire()
        self.lock.acquire()
        item = self.queue.get()
        self.lock.release()
        self.empty.release()
        return item

def extract_frames(queue, video_path='clip.mp4', output_dir='output', max_frames=72):
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} didn't exist, creating")
        os.makedirs(output_dir)
    
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success, image = vidcap.read()

    while success and count < max_frames:
        # Convert to JPEG but just store the numpy array
        success, jpgImage = cv2.imencode('.jpg', image)
        if success:
            queue.put(jpgImage)  # Put the encoded image array directly into the queue
        success, image = vidcap.read()
        print(f'Reading frame {count} {success}')
        count += 1

    vidcap.release()
    queue.put(None)  # Signal that the extraction is done
    print("Frame extraction completed.")

def convert_to_grayscale(input_queue, output_queue):
    while True:
        jpgImage = input_queue.get()
        if jpgImage is None:
            output_queue.put(None)
            print("Grayscale conversion completed.")
            break

        image = cv2.imdecode(jpgImage, cv2.IMREAD_COLOR)  # Decode the image
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_queue.put(gray_frame)
        print("Converted to grayscale and processed.")

def display_frames(queue):
    count = 0
    while True:
        gray_frame = queue.get()  # Get the grayscale image array
        if gray_frame is None:
            print('Finished displaying all frames')
            cv2.destroyAllWindows()
            break

        cv2.imshow('Video', gray_frame)
        print(f'Displaying frame {count}')
        if cv2.waitKey(42) & 0xFF == ord("q"):
            break
        count += 1

def main():
    extract_queue = BlockingQueue(72)
    grayscale_queue = BlockingQueue(72)

    extract_thread = threading.Thread(target=extract_frames, args=(extract_queue, 'clip.mp4', 'frames'))
    convert_thread = threading.Thread(target=convert_to_grayscale, args=(extract_queue, grayscale_queue))
    display_thread = threading.Thread(target=display_frames, args=(grayscale_queue,))

    extract_thread.start()
    convert_thread.start()
    display_thread.start()

    extract_thread.join()
    convert_thread.join()
    display_thread.join()

if __name__ == "__main__":
    main()
