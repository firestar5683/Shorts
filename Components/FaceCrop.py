import cv2
import numpy as np
from moviepy.editor import *
from Components.Speaker import detect_faces_and_speakers, Frames
import gc
from collections import deque

global Fps

def crop_to_vertical(input_video_path, output_video_path):
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = int(original_height)
    vertical_width = int(vertical_height * 9 / 16)
    print(f"Dimensions for vertical crop: {vertical_height} x {vertical_width}")

    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        cap.release()
        return

    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    half_width = vertical_width // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    global Fps
    Fps = fps

    frame_count = 0  # Track frames written to output

    # Stabilization setup
    stabilized_centerX = original_width // 2
    stabilization_factor = 0.1  # Lower factor for smoother following
    previous_faces = deque(maxlen=10)  # Larger rolling average for stability
    min_center_deviation = 20  # Minimum deviation threshold for filtering inconsistent detections

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Filter to keep only the largest detected face
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])  # Select face based on area (w * h)
            x, y, w, h = largest_face
            current_centerX = x + (w // 2)
            
            # Filter inconsistent detections
            if abs(current_centerX - stabilized_centerX) < min_center_deviation:
                previous_faces.append(current_centerX)
            elif len(previous_faces) > 0:
                # Use the average of previous faces if current detection is inconsistent
                current_centerX = int(np.mean(previous_faces))
        elif previous_faces:
            # Use rolling average if no face detected
            current_centerX = int(np.mean(previous_faces))
        else:
            # Default to center if no face and no history
            current_centerX = original_width // 2

        # Apply stabilization using exponential moving average
        stabilized_centerX = int(stabilized_centerX * (1 - stabilization_factor) + current_centerX * stabilization_factor)

        # Calculate x_start and x_end based on stabilized centerX
        x_start = max(0, stabilized_centerX - half_width)
        x_end = min(original_width, stabilized_centerX + half_width)

        # Crop frame and ensure dimensions match the expected output size
        cropped_frame = frame[:, x_start:x_end]

        # Resize cropped_frame to ensure consistent dimensions
        if cropped_frame.shape[1] != vertical_width or cropped_frame.shape[0] != vertical_height:
            cropped_frame = cv2.resize(cropped_frame, (vertical_width, vertical_height))
            print(f"Resizing frame to maintain {vertical_width}x{vertical_height}")

        # Ensure non-empty cropped frame before writing
        if cropped_frame.size == 0:
            print("Warning: Empty cropped frame, skipping frame")
            continue

        out.write(cropped_frame)
        frame_count += 1  # Increment frame count

    print(f"Frames written to output: {frame_count}/{total_frames}")
    cap.release()
    out.release()
    print("Cropping complete. The video has been saved to", output_video_path)
    gc.collect()

def combine_videos(video_with_audio, video_without_audio, output_filename):
    try:
        clip_with_audio = VideoFileClip(video_with_audio)
        clip_without_audio = VideoFileClip(video_without_audio)

        audio = clip_with_audio.audio
        combined_clip = clip_without_audio.set_audio(audio)

        global Fps
        combined_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac', fps=Fps, preset='medium', bitrate='3000k')
        print(f"Combined video saved successfully as {output_filename}")
    
    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")
    
    finally:
        clip_with_audio.close()
        clip_without_audio.close()
        combined_clip.close()
        del clip_with_audio, clip_without_audio, combined_clip
        gc.collect()

if __name__ == "__main__":
    input_video_path = r'Out.mp4'
    output_video_path = '_output_video.mp4'
    final_video_path = 'final_video_with_audio.mp4'
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    crop_to_vertical(input_video_path, output_video_path)
    combine_videos(input_video_path, output_video_path, final_video_path)
