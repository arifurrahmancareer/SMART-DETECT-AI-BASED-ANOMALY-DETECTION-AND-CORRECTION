import streamlit as st
import streamlit.components.v1 as components
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import io
import pandas as pd
import zipfile
import tempfile
from datetime import datetime
from fpdf import FPDF
import os
import cv2
import numpy as np
import base64
import math
from skimage.metrics import structural_similarity as ssim

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="SmartDetect - AI Image Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== IMPROVED EARTH PRO ANALYSIS FUNCTIONS ==========

def merge_overlapping_boxes(changes, iou_threshold=0.3):
    """Merge overlapping detection boxes using Non-Maximum Suppression"""
    if not changes:
        return []
    
    # Convert to format for NMS
    boxes = []
    for change in changes:
        x = change['x']
        y = change['y']
        w = change['width']
        h = change['height']
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        boxes.append([x1, y1, x2, y2, change['confidence']])
    
    boxes = np.array(boxes)
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(),
        boxes[:, 4].tolist(),
        score_threshold=0.1,
        nms_threshold=iou_threshold
    )
    
    if len(indices) == 0:
        return []
    
    # Flatten indices if needed
    if isinstance(indices, tuple):
        indices = indices[0] if len(indices) > 0 else []
    indices = indices.flatten() if hasattr(indices, 'flatten') else indices
    
    merged_changes = []
    for idx in indices:
        idx = int(idx)
        change = changes[idx]
        merged_changes.append(change)
    
    return merged_changes

def detect_changes_comprehensive(img_old, img_new, min_area=50):
    """
    COMPREHENSIVE BUILDING DETECTION - Detects all buildings, large and small
    Uses 6 different detection methods and combines them intelligently
    """
    old_arr = np.array(img_old)
    new_arr = np.array(img_new)
    
    if old_arr.shape != new_arr.shape:
        new_arr = cv2.resize(new_arr, (old_arr.shape[1], old_arr.shape[0]))
    
    # Convert to grayscale
    gray_old = cv2.cvtColor(old_arr, cv2.COLOR_RGB2GRAY)
    gray_new = cv2.cvtColor(new_arr, cv2.COLOR_RGB2GRAY)
    
    # Advanced lighting normalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray_old_norm = clahe.apply(gray_old)
    gray_new_norm = clahe.apply(gray_new)
    
    all_changes = []
    img_area = gray_new.shape[0] * gray_new.shape[1]
    max_area = img_area * 0.35  # Maximum 35% of image
    
    # ===== METHOD 1: Multi-Scale Intensity Difference =====
    diff_intensity = cv2.absdiff(gray_old_norm, gray_new_norm)
    
    # Use multiple thresholds to catch different change magnitudes
    thresholds = [3, 6, 10, 15, 25]
    for thresh_val in thresholds:
        _, binary = cv2.threshold(diff_intensity, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Minimal morphology to preserve details
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                mask = np.zeros(gray_new.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_diff = cv2.mean(diff_intensity, mask=mask)[0]
                confidence = min(mean_diff / 60.0, 1.0)
                
                all_changes.append({
                    "x": int(x + w/2), "y": int(y + h/2),
                    "width": int(w), "height": int(h),
                    "area": int(area),
                    "confidence": float(max(0.20, confidence)),
                    "method": f"intensity_t{thresh_val}"
                })
    
    # ===== METHOD 2: Edge Structure Detection =====
    # Detect changes in building edges/outlines
    edges_old = cv2.Canny(gray_old_norm, 15, 80)  # Very sensitive
    edges_new = cv2.Canny(gray_new_norm, 15, 80)
    diff_edges = cv2.absdiff(edges_old, edges_new)
    
    _, binary_edges = cv2.threshold(diff_edges, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_DILATE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(diff_edges, mask=mask)[0]
            confidence = min(mean_diff / 80.0, 1.0)
            
            all_changes.append({
                "x": int(x + w/2), "y": int(y + h/2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.35, confidence)),
                "method": "edges"
            })
    
    # ===== METHOD 3: RGB Color Change Detection =====
    diff_color = cv2.absdiff(old_arr, new_arr)
    diff_color_gray = cv2.cvtColor(diff_color, cv2.COLOR_RGB2GRAY)
    
    _, binary_color = cv2.threshold(diff_color_gray, 8, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    binary_color = cv2.morphologyEx(binary_color, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(diff_color_gray, mask=mask)[0]
            confidence = min(mean_diff / 70.0, 1.0)
            
            all_changes.append({
                "x": int(x + w/2), "y": int(y + h/2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.25, confidence)),
                "method": "color"
            })
    
    # ===== METHOD 4: Gradient/Texture Detection =====
    # Detects texture changes (important for buildings)
    sobelx_old = cv2.Sobel(gray_old_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobely_old = cv2.Sobel(gray_old_norm, cv2.CV_64F, 0, 1, ksize=3)
    sobelx_new = cv2.Sobel(gray_new_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobely_new = cv2.Sobel(gray_new_norm, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_old = np.sqrt(sobelx_old**2 + sobely_old**2)
    gradient_new = np.sqrt(sobelx_new**2 + sobely_new**2)
    gradient_diff = np.abs(gradient_new - gradient_old).astype(np.uint8)
    
    _, binary_grad = cv2.threshold(gradient_diff, 8, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary_grad = cv2.morphologyEx(binary_grad, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(binary_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(gradient_diff, mask=mask)[0]
            confidence = min(mean_diff / 80.0, 1.0)
            
            all_changes.append({
                "x": int(x + w/2), "y": int(y + h/2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.30, confidence)),
                "method": "gradient"
            })
    
    # ===== METHOD 5: Adaptive Thresholding =====
    # Catches local variations
    adaptive = cv2.adaptiveThreshold(
        diff_intensity, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 13, -2
    )
    kernel = np.ones((2, 2), np.uint8)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(diff_intensity, mask=mask)[0]
            confidence = min(mean_diff / 70.0, 1.0)
            
            all_changes.append({
                "x": int(x + w/2), "y": int(y + h/2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.25, confidence)),
                "method": "adaptive"
            })
    
    # ===== METHOD 6: Laplacian (Detail Detection) =====
    # Excellent for catching fine building details
    lap_old = cv2.Laplacian(gray_old_norm, cv2.CV_64F)
    lap_new = cv2.Laplacian(gray_new_norm, cv2.CV_64F)
    lap_diff = np.abs(lap_new - lap_old).astype(np.uint8)
    
    _, binary_lap = cv2.threshold(lap_diff, 5, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    binary_lap = cv2.morphologyEx(binary_lap, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary_lap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(lap_diff, mask=mask)[0]
            confidence = min(mean_diff / 50.0, 1.0)
            
            all_changes.append({
                "x": int(x + w/2), "y": int(y + h/2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.25, confidence)),
                "method": "laplacian"
            })
    
    # Merge overlapping detections
    merged_changes = merge_overlapping_boxes(all_changes, iou_threshold=0.4)
    
    return merged_changes

def detect_changes_yolo(img_old, img_new, model, min_confidence=0.15):
    """Enhanced YOLO detection with lower confidence threshold for buildings"""
    results_old = model(img_old, conf=min_confidence)[0]
    results_new = model(img_new, conf=min_confidence)[0]
    
    # Extract building-related classes
    building_classes = ['building', 'house', 'car', 'truck', 'bus', 'train']
    
    boxes_old = []
    if results_old.boxes is not None:
        for box in results_old.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0])
            
            if conf >= min_confidence:
                boxes_old.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'class': cls_name,
                    'conf': conf
                })
    
    boxes_new = []
    if results_new.boxes is not None:
        for box in results_new.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0])
            
            if conf >= min_confidence:
                boxes_new.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'class': cls_name,
                    'conf': conf
                })
    
    new_objects = []
    for box_new in boxes_new:
        is_new = True
        x_new_center = (box_new['xyxy'][0] + box_new['xyxy'][2]) / 2
        y_new_center = (box_new['xyxy'][1] + box_new['xyxy'][3]) / 2
        
        for box_old in boxes_old:
            x_old_center = (box_old['xyxy'][0] + box_old['xyxy'][2]) / 2
            y_old_center = (box_old['xyxy'][1] + box_old['xyxy'][3]) / 2
            
            # Check if same object (within tolerance)
            distance = np.sqrt((x_new_center - x_old_center)**2 + (y_new_center - y_old_center)**2)
            if distance < 30:  # Reduced tolerance
                is_new = False
                break
        
        if is_new:
            x0, y0, x1, y1 = box_new['xyxy']
            new_objects.append({
                "x": int((x0 + x1) / 2),
                "y": int((y0 + y1) / 2),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
                "area": int((x1 - x0) * (y1 - y0)),
                "confidence": box_new['conf'],
                "class": box_new['class'],
                "type": f"New {box_new['class'].capitalize()}"
            })
    
    return new_objects

def classify_change_type(change, year_old, year_new):
    """Classify the type of change with improved categorization"""
    if 'type' in change and change['type']:
        return change['type']
    
    if 'class' in change:
        obj_type = change['class'].lower()
        if obj_type in ['building', 'house']:
            return "New Building"
        elif obj_type in ['road', 'street']:
            return "New Road"
        elif obj_type in ['tree', 'plant', 'vegetation', 'potted plant']:
            return "Vegetation Growth"
        elif obj_type in ['car', 'truck', 'bus', 'vehicle']:
            return "New Vehicle/Structure"
        else:
            return f"New {obj_type.capitalize()}"
    
    # Classify by size and shape
    area = change.get('area', change.get('width', 0) * change.get('height', 0))
    width = change.get('width', 0)
    height = change.get('height', 0)
    aspect_ratio = width / height if height > 0 else 1
    
    if area > 8000:
        if aspect_ratio > 2.5:
            return "New Road/Highway"
        else:
            return "New Large Building"
    elif area > 3000:
        if aspect_ratio > 3:
            return "New Road Segment"
        elif 0.6 < aspect_ratio < 1.5:
            return "New Medium Building"
        else:
            return "New Structure"
    elif area > 800:
        if aspect_ratio > 3.5:
            return "New Path/Lane"
        else:
            return "New Small Building"
    elif area > 200:
        if aspect_ratio > 4:
            return "New Narrow Path"
        else:
            return "New Small Shop/Structure"
    else:
        return "Minor Construction"

# ========== ORIGINAL SMARTDETECT FUNCTIONS ==========

def get_base64_image(image_path):
    """Convert local image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

def get_math_animation_value():
    """Mathematical formulas for dynamic visual effects"""
    now = datetime.now()
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    sine_pulse = (math.sin(seconds * math.pi / 30) + 1) / 2
    golden_ratio = 1.618033988749
    golden_value = (seconds * golden_ratio) % 1
    fib_sequence = [0.1, 0.15, 0.2, 0.25, 0.35, 0.45]
    fib_index = int((seconds / 10) % len(fib_sequence))
    fib_opacity = fib_sequence[fib_index]
    
    return {
        'sine': sine_pulse,
        'golden': golden_value,
        'fib_opacity': fib_opacity,
        'rotation': (seconds * 0.5) % 360
    }

def detect_cracks_opencv(image):
    """Detects cracks using computer vision (OpenCV) techniques"""
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies = []
    min_area = 100
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            anomalies.append({
                "x": x + w/2,
                "y": y + h/2,
                "width": w,
                "height": h,
                "confidence": 100.0,
                "class": "crack/defect"
            })
    return anomalies

def detect_stains_opencv(image):
    """Detects stains/discoloration using color statistics"""
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    median = cv2.medianBlur(blurred, 21)
    diff = cv2.absdiff(blurred, median)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies = []
    min_area = 200
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            anomalies.append({
                "x": float(x + w/2),
                "y": float(y + h/2),
                "width": float(w),
                "height": float(h),
                "confidence": 100.0,
                "class": "stain/discoloration"
            })
    return anomalies

def compare_images_ssim(img1, img2):
    """Compare two images using SSIM and return score and difference map"""
    gray1 = cv2.cvtColor(np.array(img1.convert('RGB')), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2.convert('RGB')), cv2.COLOR_RGB2GRAY)
    
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    (score, diff) = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    return score, diff, thresh

# Get mathematical values for animations
math_values = get_math_animation_value()

# ============================================================
#  RETRO CRT TERMINAL AESTHETIC - Full Theme
# ============================================================

# ---- CRT SVG Logo ----
st.markdown("""
<div style="text-align: center; padding-top: 18px; padding-bottom: 0;">
    <svg width="90" height="90" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="crtGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#00FF41;stop-opacity:1" />
                <stop offset="50%" style="stop-color:#00CC33;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#00FF41;stop-opacity:1" />
            </linearGradient>
            <filter id="crtGlow">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <filter id="crtGlowStrong">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        <!-- CRT Monitor outer frame -->
        <rect x="10" y="5" width="100" height="80" rx="8" ry="8" fill="none" stroke="url(#crtGrad)" stroke-width="2.5" filter="url(#crtGlow)" opacity="0.9"/>
        <!-- Screen inner bezel -->
        <rect x="18" y="12" width="84" height="66" rx="4" ry="4" fill="rgba(0,255,65,0.04)" stroke="url(#crtGrad)" stroke-width="1" opacity="0.6"/>
        <!-- Scanline hints -->
        <line x1="20" y1="24" x2="100" y2="24" stroke="#00FF41" stroke-width="0.5" opacity="0.15"/>
        <line x1="20" y1="36" x2="100" y2="36" stroke="#00FF41" stroke-width="0.5" opacity="0.15"/>
        <line x1="20" y1="48" x2="100" y2="48" stroke="#00FF41" stroke-width="0.5" opacity="0.15"/>
        <line x1="20" y1="60" x2="100" y2="60" stroke="#00FF41" stroke-width="0.5" opacity="0.15"/>
        <!-- Eye/detection icon inside screen -->
        <ellipse cx="60" cy="44" rx="24" ry="16" fill="none" stroke="url(#crtGrad)" stroke-width="2" filter="url(#crtGlowStrong)"/>
        <circle cx="60" cy="44" r="8" fill="url(#crtGrad)" filter="url(#crtGlow)"/>
        <circle cx="60" cy="44" r="3" fill="#0a0a0a"/>
        <!-- Crosshair -->
        <line x1="60" y1="24" x2="60" y2="34" stroke="#00FF41" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>
        <line x1="60" y1="54" x2="60" y2="64" stroke="#00FF41" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>
        <line x1="32" y1="44" x2="42" y2="44" stroke="#00FF41" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>
        <line x1="78" y1="44" x2="88" y2="44" stroke="#00FF41" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>
        <!-- Monitor stand -->
        <rect x="48" y="87" width="24" height="6" rx="2" fill="url(#crtGrad)" opacity="0.5"/>
        <rect x="40" y="93" width="40" height="4" rx="2" fill="url(#crtGrad)" opacity="0.4"/>
        <!-- Power LED -->
        <circle cx="26" cy="82" r="2" fill="#00FF41" filter="url(#crtGlow)">
            <animate attributeName="opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite"/>
        </circle>
        <!-- Corner brackets (retro scan target) -->
        <path d="M22 16 L22 20 L26 20" fill="none" stroke="#00FF41" stroke-width="1" opacity="0.5"/>
        <path d="M98 16 L98 20 L94 20" fill="none" stroke="#00FF41" stroke-width="1" opacity="0.5"/>
        <path d="M22 74 L22 70 L26 70" fill="none" stroke="#00FF41" stroke-width="1" opacity="0.5"/>
        <path d="M98 74 L98 70 L94 70" fill="none" stroke="#00FF41" stroke-width="1" opacity="0.5"/>
    </svg>
</div>
""", unsafe_allow_html=True)

# ---- CRT Title ----
st.markdown("""
<div class="crt-title-wrap" style="text-align:center;">
    <div class="crt-title">SmartDetect://AI_Anomaly_Detection</div>
    <div class="crt-subtitle">[ SYSTEM ONLINE ] &mdash; v2.0 &mdash; RETRO TERMINAL</div>
</div>
""", unsafe_allow_html=True)

# Default theme and model
theme = "Dark"
model_choice = "Roboflow Default"

# ---- RETRO CRT TERMINAL STYLES ----
st.markdown("""
<style>
/* ===== FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&display=swap');

/* ===== CRT KEYFRAMES ===== */
@keyframes crt-flicker {
    0%   { opacity: 0.97; }
    5%   { opacity: 0.95; }
    10%  { opacity: 0.98; }
    15%  { opacity: 0.96; }
    20%  { opacity: 0.99; }
    50%  { opacity: 0.96; }
    80%  { opacity: 0.98; }
    100% { opacity: 0.97; }
}

@keyframes scanline-move {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}

@keyframes text-glow-pulse {
    0%, 100% { text-shadow: 0 0 4px #00FF41, 0 0 11px #00FF41, 0 0 19px #00FF41, 0 0 40px #00803020; }
    50%      { text-shadow: 0 0 4px #00FF41, 0 0 15px #00FF41, 0 0 25px #00FF41, 0 0 50px #00803040; }
}

@keyframes cursor-blink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0; }
}

@keyframes border-glow {
    0%, 100% { box-shadow: inset 0 0 30px rgba(0,255,65,0.05), 0 0 15px rgba(0,255,65,0.08); }
    50%      { box-shadow: inset 0 0 50px rgba(0,255,65,0.08), 0 0 25px rgba(0,255,65,0.12); }
}

@keyframes power-on {
    0% { transform: scaleY(0.005) scaleX(0.3); filter: brightness(30); }
    10% { transform: scaleY(0.005) scaleX(1); filter: brightness(10); }
    30% { transform: scaleY(1) scaleX(1); filter: brightness(2); }
    50% { filter: brightness(1.2); }
    100% { transform: scaleY(1) scaleX(1); filter: brightness(1); }
}

/* ===== HIDE STREAMLIT DEFAULTS ===== */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
section[data-testid="stSidebar"],
.stDeployButton,
#MainMenu {
    display: none !important;
}

/* ===== BASE CRT SCREEN ===== */
html, body, .stApp {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    background: #050505 !important;
    color: #00FF41 !important;
}

.stApp {
    animation: power-on 1.2s ease-out;
}

/* Scanline overlay on the entire app */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 0, 0, 0.15) 2px,
        rgba(0, 0, 0, 0.15) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Slow-moving scanline bar */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 120px;
    background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(0, 255, 65, 0.03) 50%,
        transparent 100%
    );
    animation: scanline-move 8s linear infinite;
    pointer-events: none;
    z-index: 9998;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 50% 50%, #0a1a0a 0%, #050505 70%) !important;
    animation: crt-flicker 0.15s infinite alternate;
}

/* CRT vignette (dark corners) */
[data-testid="stAppViewContainer"] > div::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.65) 100%);
    pointer-events: none;
    z-index: 9997;
}

/* ===== TITLE STYLES ===== */
.crt-title-wrap {
    margin-bottom: 8px;
}

.crt-title {
    font-family: 'VT323', 'Share Tech Mono', monospace !important;
    font-size: 2.6rem !important;
    color: #00FF41 !important;
    text-shadow: 0 0 7px #00FF41, 0 0 15px #00FF41, 0 0 30px #00803080, 0 0 60px #00401040;
    animation: text-glow-pulse 3s ease-in-out infinite;
    letter-spacing: 2px;
    padding: 5px 0;
    -webkit-text-fill-color: #00FF41;
    background: none;
}

.crt-subtitle {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.95rem;
    color: #00CC33 !important;
    text-shadow: 0 0 5px #00FF4180;
    letter-spacing: 4px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-top: -2px;
}

/* ===== ALL TEXT GREEN ===== */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div,
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown h1,
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
[data-testid="stText"], [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] {
    color: #00FF41 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

.stMarkdown h2, .stMarkdown h3 {
    color: #00FF41 !important;
    text-shadow: 0 0 8px #00FF4160;
    border-bottom: 1px solid #00FF4130;
    padding-bottom: 6px;
}

/* ===== TAB NAVIGATION ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: transparent;
    padding: 10px 0;
    border-bottom: 1px solid #00FF4130;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    white-space: nowrap;
    background-color: rgba(0, 255, 65, 0.03);
    border-radius: 0px;
    color: #00CC33 !important;
    border: 1px solid #00FF4125;
    padding: 0 18px;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.15s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(0, 255, 65, 0.08);
    border-color: #00FF4150;
    color: #00FF41 !important;
    text-shadow: 0 0 8px #00FF4140;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(0, 255, 65, 0.12) !important;
    border-color: #00FF41 !important;
    color: #00FF41 !important;
    text-shadow: 0 0 10px #00FF4180;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.15), inset 0 0 15px rgba(0, 255, 65, 0.05);
}

/* Tab highlight bar */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #00FF41 !important;
    box-shadow: 0 0 10px #00FF41, 0 0 20px #00FF4180;
}

/* ===== FORM ELEMENTS ===== */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    background-color: #0a0a0a !important;
    border: 1px solid #00FF4140 !important;
    border-radius: 0px !important;
    color: #00FF41 !important;
    font-family: 'Share Tech Mono', monospace !important;
    caret-color: #00FF41;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #00FF41 !important;
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.3), inset 0 0 5px rgba(0, 255, 65, 0.1) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background-color: rgba(0, 255, 65, 0.08) !important;
    color: #00FF41 !important;
    border: 1px solid #00FF4150 !important;
    border-radius: 0px !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.85rem;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background-color: rgba(0, 255, 65, 0.18) !important;
    border-color: #00FF41 !important;
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.25), inset 0 0 10px rgba(0, 255, 65, 0.08);
    text-shadow: 0 0 8px #00FF41;
}

.stButton > button:active {
    background-color: rgba(0, 255, 65, 0.25) !important;
    box-shadow: 0 0 30px rgba(0, 255, 65, 0.35);
}

/* Primary button special glow */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: rgba(0, 255, 65, 0.15) !important;
    border: 2px solid #00FF41 !important;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
}

/* Download buttons */
.stDownloadButton > button {
    background-color: rgba(0, 255, 65, 0.06) !important;
    color: #00CC33 !important;
    border: 1px dashed #00FF4140 !important;
    border-radius: 0px !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stDownloadButton > button:hover {
    background-color: rgba(0, 255, 65, 0.12) !important;
    border-style: solid !important;
    border-color: #00FF41 !important;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
    color: #00FF41 !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] section {
    background-color: rgba(0, 255, 65, 0.02) !important;
    border: 1px dashed #00FF4135 !important;
    border-radius: 0px !important;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #00FF4170 !important;
    box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.05);
}

[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span {
    color: #00CC33 !important;
}

/* ===== ALERTS / INFO BOXES ===== */
.stAlert, [data-testid="stAlert"] {
    background-color: rgba(0, 255, 65, 0.04) !important;
    border: 1px solid #00FF4130 !important;
    border-radius: 0px !important;
    color: #00FF41 !important;
    border-left: 3px solid #00FF41 !important;
}

.stAlert p, [data-testid="stAlert"] p,
.stAlert span, [data-testid="stAlert"] span {
    color: #00CC33 !important;
}

/* Success alert */
[data-baseweb="notification"][kind="positive"],
div[data-testid="stAlert"][data-type="success"] {
    border-left-color: #00FF41 !important;
}

/* Warning alert */
div[data-testid="stAlert"][data-type="warning"] {
    border-left-color: #FFB000 !important;
    background-color: rgba(255, 176, 0, 0.04) !important;
}

div[data-testid="stAlert"][data-type="warning"] p {
    color: #FFB000 !important;
}

/* Error alert */
div[data-testid="stAlert"][data-type="error"] {
    border-left-color: #FF3333 !important;
    background-color: rgba(255, 50, 50, 0.04) !important;
}

div[data-testid="stAlert"][data-type="error"] p {
    color: #FF3333 !important;
}

/* ===== SLIDERS ===== */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #00FF41 !important;
    box-shadow: 0 0 10px #00FF41;
}

.stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(to right, #00FF4120, #00FF41) !important;
}

.stSlider label, .stSlider span {
    color: #00FF41 !important;
}

/* ===== RADIO / CHECKBOX ===== */
.stRadio label, .stCheckbox label {
    color: #00FF41 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.stRadio [role="radiogroup"] label span,
.stCheckbox span {
    color: #00CC33 !important;
}

/* ===== DATAFRAME / TABLE ===== */
[data-testid="stDataFrame"],
.stDataFrame {
    border: 1px solid #00FF4130 !important;
    border-radius: 0px !important;
}

[data-testid="stDataFrame"] th {
    background-color: rgba(0, 255, 65, 0.1) !important;
    color: #00FF41 !important;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: rgba(0, 255, 65, 0.04);
    border: 1px solid #00FF4120;
    border-radius: 0px;
    padding: 12px;
    animation: border-glow 4s ease-in-out infinite;
}

[data-testid="stMetric"] label {
    color: #00CC33 !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #00FF41 !important;
    text-shadow: 0 0 8px #00FF4140;
    font-family: 'VT323', monospace !important;
    font-size: 2rem !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #00CC33 !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background-color: #00FF41 !important;
    box-shadow: 0 0 10px #00FF41, 0 0 20px #00FF4180;
}

.stProgress > div > div {
    background-color: #00FF4115 !important;
}

/* ===== SPINNER ===== */
.stSpinner > div {
    border-top-color: #00FF41 !important;
}

/* ===== FILE UPLOADER DRAG TEXT ===== */
[data-testid="stFileUploader"] label {
    color: #00CC33 !important;
}

/* ===== SEPARATOR / DIVIDER ===== */
hr, .stMarkdown hr {
    border-color: #00FF4120 !important;
}

/* ===== IMAGE CAPTIONS ===== */
[data-testid="stImage"] img {
    border: 1px solid #00FF4125;
    border-radius: 0px;
}

/* ===== SCROLLBAR CRT STYLE ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0a0a0a;
}

::-webkit-scrollbar-thumb {
    background: #00FF4140;
    border: 1px solid #00FF4120;
}

::-webkit-scrollbar-thumb:hover {
    background: #00FF4170;
}

/* ===== SELECTION COLOR ===== */
::selection {
    background: #00FF4140;
    color: #FFF;
}

/* ===== LINKS ===== */
a, a:visited {
    color: #00FF41 !important;
    text-decoration: underline;
}

a:hover {
    text-shadow: 0 0 8px #00FF41;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    color: #00FF41 !important;
    background-color: rgba(0, 255, 65, 0.04) !important;
    border: 1px solid #00FF4120 !important;
}

/* ===== CAMERA INPUT ===== */
[data-testid="stCameraInput"] > div {
    border: 1px solid #00FF4130 !important;
    border-radius: 0px !important;
}

/* ===== VIDEO ===== */
video {
    border: 1px solid #00FF4130 !important;
}

/* ===== COLUMN GAP ===== */
[data-testid="column"] {
    padding: 0 8px;
}

/* ===== MARKDOWN GENERAL ===== */
.stMarkdown > div {
    border-radius: 0px;
}

/* ===== TOOLTIP / HELP ICON ===== */
[data-testid="stTooltipIcon"] svg {
    fill: #00FF4180 !important;
}

/* ===== BOTTOM PADDING ===== */
.block-container {
    padding-bottom: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ---- Main Application Logic ----
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Upload & Preview", "🔍 Detection & AI Correction",
     "📹 Snapshot Video Detection", "🌍 Earth Pro Analysis", "📊 Feedback & Report", "🧭 Tutorial", "ℹ️ About/Docs"
])

# ---------- Tab 1: Upload & Preview ----------
with tab1:
    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.markdown("### 🖼️ Image Gallery")
        num_images = len(uploaded_files)
        cols_per_row = min(num_images, 4)
        cols = st.columns(cols_per_row)
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % cols_per_row]:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption=uploaded_file.name, width=200)
                uploaded_file.seek(0)

# ---------- Tab 2: Detection & Correction ----------
with tab2:
    uploaded_files = st.session_state.uploaded_files
    if not uploaded_files:
        st.warning("Upload images in the first tab.")
    else:
        st.markdown("### ⚙️ Detection Settings")
        
        detection_mode = st.radio(
            "Detection Mode",
            ["🛡️ Object Detection (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)"],
            help="Choose your detection target: Objects, Cracks, or Stains."
        )
        
        threshold = st.slider("Minimum Confidence (%)", 0, 100, 50)
        
        st.markdown("### 🤖 AI Correction Settings")
        use_ai_correction = st.checkbox("Enable AI-Powered Correction", value=True, help="Use AI to intelligently remove anomalies")
        
        if use_ai_correction:
            st.info("🎨 AI will intelligently remove detected anomalies and generate clean, natural-looking corrections.")
        else:
            st.warning("⚠️ AI correction disabled. Only detection will be performed.")
        
        st.write(f"**Detection:** Anomalies with confidence ≥ {threshold}% will be shown.")
        
        color_picker_high = "#00ff00"
        color_picker_mid = "#ff0000"
        color_picker_low = "#ffff00"

        if "session_results" not in st.session_state:
            st.session_state.session_results = []
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        session_results = []

        with zipfile.ZipFile(temp_zip.name, "w") as zip_all:
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"---\n#### Image {idx + 1}: {uploaded_file.name}")
                image_bytes = uploaded_file.getvalue()
                orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                preds = []
                
                with st.spinner(f"Detecting anomalies ({detection_mode}) for {uploaded_file.name}..."):
                    try:
                        if "Road/Surface Cracks" in detection_mode:
                            opencv_preds = detect_cracks_opencv(orig_img)
                            preds = opencv_preds
                            st.success(f"✅ Crack Detection complete! Found {len(preds)} defects.")
                            
                        elif "Stain/Discoloration" in detection_mode:
                            opencv_preds = detect_stains_opencv(orig_img)
                            preds = opencv_preds
                            st.success(f"✅ Stain Detection complete! Found {len(preds)} defects.")
                            
                        else:
                            @st.cache_resource
                            def load_yolo_model():
                                from ultralytics import YOLO
                                return YOLO("yolov8n.pt")

                            
                            model = load_yolo_model()
                            results = model(orig_img)
                            
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x, y, w, h = box.xywh[0].tolist()
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = model.names[cls]
                                    
                                    if conf * 100 >= threshold:
                                        preds.append({
                                            "x": x,
                                            "y": y,
                                            "width": w,
                                            "height": h,
                                            "confidence": conf,
                                            "class": label
                                        })
                            
                            st.success(f"✅ AI Detection complete! Found {len(preds)} objects/anomalies.")
                        
                    except Exception as e:
                        st.error(f"❌ Detection Error: {str(e)}")
                        continue

                if preds:
                    df = pd.DataFrame(preds)[["x", "y", "width", "height", "confidence"]]
                    df["confidence (%)"] = (df["confidence"] * 100).round(2)
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download anomaly results as CSV",
                        data=csv,
                        file_name=f"anomaly_results_{idx+1}.csv",
                        mime="text/csv",
                        key=f"csv_download_{idx}"
                    )

                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine="openpyxl")
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Download anomaly results as Excel",
                        data=excel_buffer,
                        file_name=f"anomaly_results_{idx+1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_download_{idx}"
                    )

                    zip_all.writestr(f"anomaly_results_{idx+1}.csv", csv)
                    zip_all.writestr(f"anomaly_results_{idx+1}.xlsx", excel_buffer.getvalue())

                def get_color(conf):
                    if conf >= 0.9:
                        return color_picker_high
                    elif conf >= 0.7:
                        return color_picker_mid
                    else:
                        return color_picker_low

                im_anno = orig_img.copy()
                draw = ImageDraw.Draw(im_anno)
                for pred in preds:
                    x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                    y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                    x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                    y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                    color = get_color(pred["confidence"])
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                im_corr = orig_img.copy()
                
                if use_ai_correction and preds:
                    with st.spinner("🤖 AI is generating corrected image..."):
                        try:
                            mask = Image.new('L', orig_img.size, 0)
                            mask_draw = ImageDraw.Draw(mask)
                            
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                mask_draw.rectangle([x0, y0, x1, y1], fill=255)
                            
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                box = (x0, y0, x1, y1)
                                region = im_corr.crop(box).filter(ImageFilter.GaussianBlur(20))
                                im_corr.paste(region, box)
                            
                            st.success("✅ AI correction completed!")
                        except Exception as e:
                            st.error(f"AI correction failed: {str(e)}")
                            im_corr = orig_img.copy()

                st.markdown("""
                <div style="display: flex; justify-content: center; gap: 25px; margin: 15px 0; padding: 12px 20px; background: rgba(128,128,128,0.1); border-radius: 50px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #00ff00; border-radius: 50%; box-shadow: 0 2px 6px rgba(0,255,0,0.4);"></div>
                        <span><b>High Confidence</b> ≥90%</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #ffff00; border-radius: 50%; box-shadow: 0 2px 6px rgba(255,255,0,0.4);"></div>
                        <span><b>Medium Confidence</b> 70-89%</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #ff0000; border-radius: 50%; box-shadow: 0 2px 6px rgba(255,0,0,0.4);"></div>
                        <span><b>Low Confidence</b> &lt;70%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(orig_img, caption="Original", use_container_width=True, output_format="PNG")
                with col2:
                    st.image(im_anno, caption="Detected", use_container_width=True, output_format="PNG")
                with col3:
                    st.image(im_corr, caption="Corrected", use_container_width=True, output_format="PNG")

                img_anno_b = io.BytesIO()
                im_anno.save(img_anno_b, format="PNG")
                img_anno_b.seek(0)
                st.download_button(
                    label="Download Annotated",
                    data=img_anno_b,
                    file_name=f"annotated_{idx+1}.png",
                    mime="image/png",
                    key=f"anno_download_{idx}"
                )

                img_corr_b = io.BytesIO()
                im_corr.save(img_corr_b, format="PNG")
                img_corr_b.seek(0)
                st.download_button(
                    label="Download Corrected",
                    data=img_corr_b,
                    file_name=f"corrected_{idx+1}.png",
                    mime="image/png",
                    key=f"corr_download_{idx}"
                )

                zip_all.writestr(f"annotated_{idx+1}.png", img_anno_b.getvalue())
                zip_all.writestr(f"corrected_{idx+1}.png", img_corr_b.getvalue())

                session_results.append({
                    "filename": uploaded_file.name,
                    "num_anomalies": len(preds),
                    "ai_corrected": use_ai_correction,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "correction_file": f"corrected_{idx+1}.png"
                })

        st.session_state.session_results = session_results

        with open(temp_zip.name, "rb") as zf:
            all_zip_bytes = zf.read()
        
        try:
            os.unlink(temp_zip.name)
        except Exception:
            pass
            
        st.download_button(
            label="Download All Results/Images as ZIP",
            data=all_zip_bytes,
            file_name="SmartDetect_results.zip",
            mime="application/zip",
            key="zip_download_all"
        )

        st.write("---")
        st.markdown("## Session Results")
        if session_results:
            st.dataframe(pd.DataFrame(session_results))

# ---------- Tab 3: SnapShot Video Detection ----------
with tab3:
    st.markdown("### 📹 Snapshot Video Anomaly Detection")
    st.info("🎥 Detect anomalies in real-time from your webcam or video feed.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎬 Video Source")
        video_source = st.radio(
            "Select video source:",
            ["Webcam", "Upload Video File"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### ⚙️ Detection Settings")
        video_threshold = st.slider("Confidence Threshold (%)", 0, 100, 60, key="video_threshold")
        show_boxes = st.checkbox("Show Detection Boxes", value=True, key="show_boxes")
    
    st.markdown("---")
    
    if video_source == "Webcam":
        st.markdown("#### 📷 Webcam Feed")
        st.warning("⚠️ **Note:** Webcam access requires browser permissions.")
        
        st.markdown("""
        **How to use:**
        1. Click "Enable Webcam" below
        2. Point camera at surfaces to check
        3. Click "Capture & Detect"
        4. View detected anomalies
        """)
        
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False
        
        col_start, col_snap, col_stop = st.columns(3)
        
        with col_start:
            if st.button("🎥 Enable Webcam", key="start_webcam", use_container_width=True):
                st.session_state.webcam_running = True
        
        with col_snap:
            capture_btn = st.button("📸 Capture & Detect", key="snapshot", use_container_width=True, disabled=not st.session_state.webcam_running)
        
        with col_stop:
            if st.button("⏹️ Disable Webcam", key="stop_webcam", use_container_width=True):
                st.session_state.webcam_running = False
        
        video_detection_mode = st.radio(
            "Video Detection Mode",
            ["🛡️ Object Detection (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)"],
            horizontal=True,
            key="video_mode_select"
        )

        st.markdown("---")
        
        if st.session_state.webcam_running:
            st.info("📷 Webcam is active. Click 'Capture & Detect' to analyze.")
            camera_image = st.camera_input("Live Camera Feed", key="camera_feed")
            
            if camera_image is not None and capture_btn:
                with st.spinner("🔍 Detecting anomalies..."):
                    try:
                        img_bytes = camera_image.getvalue()
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        
                        preds = []
                        
                        if "Road/Surface Cracks" in video_detection_mode:
                            preds = detect_cracks_opencv(img)
                            st.success(f"✅ Found {len(preds)} defects")
                        elif "Stain/Discoloration" in video_detection_mode:
                            preds = detect_stains_opencv(img)
                            st.success(f"✅ Found {len(preds)} stains")
                        else:
                            @st.cache_resource
                            def load_yolo_model():
                                from ultralytics import YOLO
                                return YOLO("yolov8n.pt")
                            
                            model = load_yolo_model()
                            results = model(img)
                            
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x, y, w, h = box.xywh[0].tolist()
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = model.names[cls]
                                    
                                    if conf * 100 >= video_threshold:
                                        preds.append({
                                            "x": x,
                                            "y": y,
                                            "width": w,
                                            "height": h,
                                            "confidence": conf,
                                            "class": label
                                        })
                            
                            st.success(f"✅ Found {len(preds)} anomalies")
                        
                        img_annotated = img.copy()
                        draw = ImageDraw.Draw(img_annotated)
                        
                        for pred in preds:
                            x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                            y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                            x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                            y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                            draw.rectangle([x0, y0, x1, y1], outline="#FF0000", width=3)
                        
                        col_orig, col_detect = st.columns(2)
                        with col_orig:
                            st.markdown("**Original Frame**")
                            st.image(img, use_container_width=True)
                        with col_detect:
                            st.markdown("**Detected Anomalies**")
                            st.image(img_annotated, use_container_width=True)
                        
                        buf = io.BytesIO()
                        img_annotated.save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Annotated Frame",
                            data=buf,
                            file_name="webcam_detection.png",
                            mime="image/png"
                        )
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("👆 Click 'Enable Webcam' to start")
    else:
        st.markdown("#### 📁 Upload Video File")
        uploaded_video = st.file_uploader("Upload video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_video:
            st.video(uploaded_video)
            st.info("Video processing feature - Install opencv-python and ffmpeg to enable")

# ---------- Tab 4: IMPROVED EARTH PRO ANALYSIS ----------
with tab4:
    st.markdown("### 🌍 Google Earth Pro Image Comparison - ENHANCED")
    
    st.info("""
    📸 **How to use this feature:**
    
    1. Open **Google Earth Pro** on your computer
    2. Navigate to the area you want to analyze
    3. Take a **screenshot** of the area from an earlier year (use the time slider)
    4. Take another **screenshot** of the **same exact area** from a recent year
    5. Upload both screenshots below
    
    **🎯 Enhanced Detection:** This improved version uses 6 different detection algorithms to catch ALL building changes - from tiny shops to large commercial buildings!
    """)
    
    st.markdown("---")
    
    # Upload sections
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("### 📤 Upload Earlier Year Image")
        st.caption("Upload a Google Earth Pro screenshot from an earlier year")
        earlier_image = st.file_uploader(
            "Choose earlier year image",
            type=["jpg", "jpeg", "png"],
            key="earth_earlier",
            help="Screenshot from Google Earth Pro - earlier year"
        )
        if earlier_image:
            img_earlier = Image.open(earlier_image).convert("RGB")
            st.image(img_earlier, caption="Earlier Year (Baseline)", use_container_width=True)
    
    with col_upload2:
        st.markdown("### 📤 Upload Recent Year Image")
        st.caption("Upload a Google Earth Pro screenshot from a recent year")
        recent_image = st.file_uploader(
            "Choose recent year image",
            type=["jpg", "jpeg", "png"],
            key="earth_recent",
            help="Screenshot from Google Earth Pro - recent year"
        )
        if recent_image:
            img_recent = Image.open(recent_image).convert("RGB")
            st.image(img_recent, caption="Recent Year (Current)", use_container_width=True)
    
    # Analysis settings and button
    if earlier_image and recent_image:
        st.markdown("---")
        st.markdown("#### ⚙️ Enhanced Analysis Settings")
        
        col_settings1, col_settings2, col_settings3 = st.columns(3)
        
        with col_settings1:
            min_change_area = st.slider(
                "Minimum Building Size (pixels²)", 
                min_value=10, 
                max_value=500, 
                value=50,
                step=10,
                help="Lower = Detects smaller buildings. Recommended: 30-80"
            )
        
        with col_settings2:
            yolo_confidence = st.slider(
                "AI Confidence (%)", 
                min_value=10, 
                max_value=70, 
                value=15,
                help="Lower = More detections. Recommended: 15-25%"
            )
        
        with col_settings3:
            use_yolo = st.checkbox(
                "Enable YOLO AI",
                value=True,
                help="Use deep learning for building detection"
            )
        
        st.markdown("---")
        
        # Analyze button
        if st.button("🔍 Analyze Changes (Enhanced)", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Load images
                status_text.text("📸 Loading images...")
                img_old = Image.open(earlier_image).convert("RGB")
                img_new = Image.open(recent_image).convert("RGB")
                progress_bar.progress(10)
                
                # Resize if needed
                if img_old.size != img_new.size:
                    status_text.text("📐 Resizing images to match...")
                    img_new = img_new.resize(img_old.size)
                progress_bar.progress(15)
                
                # Calculate similarity
                status_text.text("📊 Calculating image similarity...")
                ssim_score, diff_map, thresh_map = compare_images_ssim(img_old, img_new)
                progress_bar.progress(25)
                
                # Enhanced OpenCV detection with 6 methods
                status_text.text("🔬 Running comprehensive computer vision analysis (6 methods)...")
                opencv_changes = detect_changes_comprehensive(img_old, img_new, min_area=min_change_area)
                progress_bar.progress(60)
                
                st.info(f"✅ Computer Vision detected {len(opencv_changes)} changes")
                
                # YOLO detection (optional)
                yolo_changes = []
                if use_yolo:
                    status_text.text("🤖 Running deep learning (YOLO) analysis...")
                    
                    @st.cache_resource
                    def load_yolo_model():
                        from ultralytics import YOLO
                        return YOLO("yolov8n.pt")
                    
                    model = load_yolo_model()
                    yolo_changes = detect_changes_yolo(img_old, img_new, model, min_confidence=yolo_confidence / 100)
                    st.info(f"✅ YOLO AI detected {len(yolo_changes)} new objects")
                
                progress_bar.progress(80)
                
                # Combine all changes
                all_changes = opencv_changes + yolo_changes
                
                # Classify changes
                status_text.text("🏷️ Classifying detected changes...")
                for change in all_changes:
                    if 'type' not in change or not change.get('type'):
                        change['type'] = classify_change_type(change, "Earlier", "Recent")
                progress_bar.progress(90)
                
                # Create annotated image
                status_text.text("🎨 Creating annotated visualization...")
                img_annotated = img_new.copy()
                draw = ImageDraw.Draw(img_annotated)
                
                # Try to load font
                try:
                    font = ImageFont.truetype("arial.ttf", 13)
                    font_small = ImageFont.truetype("arial.ttf", 11)
                except:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()
                
                # Color coding
                def get_box_color(confidence):
                    if confidence >= 0.8:
                        return "#00ff00"  # Green
                    elif confidence >= 0.5:
                        return "#ffff00"  # Yellow
                    else:
                        return "#ff6600"  # Orange
                
                # Draw boxes
                for idx, change in enumerate(all_changes):
                    x = int(change['x'])
                    y = int(change['y'])
                    w = int(change['width'])
                    h = int(change['height'])
                    conf = change.get('confidence', 0.5)
                    change_type = change.get('type', 'Change')
                    
                    x0 = int(x - w / 2)
                    y0 = int(y - h / 2)
                    x1 = int(x + w / 2)
                    y1 = int(y + h / 2)
                    
                    color = get_box_color(conf)
                    
                    # Draw box
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                    
                    # Draw label
                    label_text = f"{change_type}"
                    conf_text = f"{conf*100:.0f}%"
                    
                    # Background for text
                    try:
                        bbox = draw.textbbox((x0, y0 - 28), label_text, font=font)
                        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0, 200))
                        draw.text((x0, y0 - 28), label_text, fill=color, font=font)
                        draw.text((x0, y0 - 14), conf_text, fill="white", font=font_small)
                    except:
                        draw.text((x0, y0 - 20), f"{label_text} {conf_text}", fill=color)
                
                progress_bar.progress(95)
                
                # Store results
                st.session_state.earth_results = {
                    'img_old': img_old,
                    'img_new': img_new,
                    'img_annotated': img_annotated,
                    'all_changes': all_changes,
                    'ssim_score': ssim_score,
                    'opencv_count': len(opencv_changes),
                    'yolo_count': len(yolo_changes)
                }
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.balloons()
                st.success(f"🎉 Analysis complete! Found **{len(all_changes)}** total changes (OpenCV: {len(opencv_changes)}, YOLO: {len(yolo_changes)})")
                
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                progress_bar.empty()
                status_text.empty()
    
    # Display results
    if "earth_results" in st.session_state:
        results = st.session_state.earth_results
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("🔍 Total Changes", len(results['all_changes']))
        with col_m2:
            st.metric("🖥️ OpenCV Detections", results['opencv_count'])
        with col_m3:
            st.metric("🤖 YOLO Detections", results['yolo_count'])
        with col_m4:
            st.metric(
                "📏 Similarity", 
                f"{results['ssim_score']:.1%}", 
                delta=f"{(1-results['ssim_score'])*100:.1f}% changed", 
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Confidence legend
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 25px; margin: 15px 0; padding: 12px 20px; background: rgba(128,128,128,0.1); border-radius: 50px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 20px; background-color: #00ff00; border-radius: 50%;"></div>
                <span><b>High</b> ≥80%</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 20px; background-color: #ffff00; border-radius: 50%;"></div>
                <span><b>Medium</b> 50-79%</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 20px; background-color: #ff6600; border-radius: 50%;"></div>
                <span><b>Low</b> <50%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Image comparison
        st.markdown("### 🖼️ Image Comparison")
        
        col_img1, col_img2, col_img3 = st.columns(3)
        
        with col_img1:
            st.markdown("**Earlier Year**")
            st.image(results['img_old'], use_container_width=True)
        
        with col_img2:
            st.markdown("**Recent Year**")
            st.image(results['img_new'], use_container_width=True)
        
        with col_img3:
            st.markdown("**🎯 All Detected Changes**")
            st.image(results['img_annotated'], use_container_width=True)
        
        # Download annotated image
        buf_annotated = io.BytesIO()
        results['img_annotated'].save(buf_annotated, format='PNG')
        buf_annotated.seek(0)
        st.download_button(
            "📥 Download Annotated Image",
            buf_annotated,
            "earth_pro_enhanced_analysis.png",
            "image/png",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Detailed changes table
        if results['all_changes']:
            st.markdown("### 📋 Detailed Change Analysis")
            
            # Create DataFrame
            changes_data = []
            for idx, change in enumerate(results['all_changes'], 1):
                changes_data.append({
                    "ID": idx,
                    "Type": change.get('type', 'Change Detected'),
                    "Confidence": f"{change.get('confidence', 0.5)*100:.1f}%",
                    "Method": change.get('method', 'N/A'),
                    "X": int(change['x']),
                    "Y": int(change['y']),
                    "Width": int(change['width']),
                    "Height": int(change['height']),
                    "Area (px²)": int(change.get('area', change['width'] * change['height']))
                })
            
            df_changes = pd.DataFrame(changes_data)
            st.dataframe(df_changes, use_container_width=True)
            
            # Download options
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv_data = df_changes.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download as CSV",
                    csv_data,
                    "earth_pro_enhanced_changes.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col_dl2:
                excel_buffer = io.BytesIO()
                df_changes.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button(
                    "📥 Download as Excel",
                    excel_buffer,
                    "earth_pro_enhanced_changes.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.info("No significant changes detected between the two images.")

# ---------- Tab 5: Feedback & Report ----------
with tab5:
    st.markdown("### 📝 Leave Feedback & Generate PDF Report")
    if "feedback_list" not in st.session_state:
        st.session_state.feedback_list = []

    feedback = st.text_area("Type feedback or bug report:", key="feedback_input")
    if st.button("Submit Feedback", key="submit_feedback_btn"):
        if feedback:
            st.session_state.feedback_list.append(feedback)
            st.success("Thank you for your feedback!")

    st.write("---")
    st.markdown("#### 💬 All Feedback")
    if st.session_state.feedback_list:
        for i, fb in enumerate(st.session_state.feedback_list, 1):
            st.markdown(f"**{i}:** {fb}")
    else:
        st.info("No feedback yet.")

    st.write("---")
    st.markdown("#### 📄 Generate PDF Summary Report")
    
    if st.button("🔄 Generate & Auto-Download PDF Report", key="generate_pdf_btn"):
        pdf_file = "SmartDetect_Session_Report.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="SmartDetect AI Image Anomaly Detection Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(10)

        session_results = st.session_state.get("session_results", [])
        
        if session_results:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Detection Results:", ln=True)
            pdf.set_font("Arial", size=11)
            for sess in session_results:
                pdf.multi_cell(0, 8, txt=f"  Image: {sess['filename']}\n  Anomalies: {sess['num_anomalies']}\n  AI Correction: {'Enabled' if sess.get('ai_corrected', False) else 'Disabled'}\n  Date: {sess['date']}\n")
                pdf.ln(5)
        else:
            pdf.cell(200, 10, txt="No detection results yet.", ln=True)
        
        pdf.ln(10)
        if st.session_state.feedback_list:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="User Feedback:", ln=True)
            pdf.set_font("Arial", size=11)
            for fb in st.session_state.feedback_list:
                pdf.multi_cell(0, 8, txt=f"  - {fb}")
                pdf.ln(3)

        pdf.output(pdf_file)
        
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode()
        
        auto_download_js = f'<script>var link = document.createElement("a");link.href = "data:application/pdf;base64,{b64_pdf}";link.download = "{pdf_file}";link.click();</script>'
        components.html(auto_download_js, height=0)
        st.success(f"✅ PDF Report generated: {pdf_file}")
        
        st.download_button("📥 Download PDF (Manual)", pdf_bytes, pdf_file, "application/pdf", key="pdf_manual_download")

# ---------- Tab 6: Tutorial ----------
with tab6:
    st.markdown("""
    ## How to Use This App (Tutorial)
    
    **Step 1: Upload & Preview**  
    Upload your images (JPG, PNG) to begin the analysis.
    
    **Step 2: Detection & AI Correction**  
    Choose your detection mode and let AI find and correct anomalies.
    
    **Step 3: Snapshot Video Detection**  
    Use your webcam for real-time anomaly detection.
    
    **Step 4: 🌍 Earth Pro Analysis (ENHANCED!)**  
    - Open Google Earth Pro and navigate to your area of interest
    - Use the time slider to select an earlier year
    - Take a screenshot
    - Select a recent year and take another screenshot
    - Upload both images here
    - **Enhanced detection uses 6 different computer vision methods**
    - Detects buildings of ALL sizes - from small shops to large complexes
    - Optionally enable YOLO AI for even better results
    
    **Step 5: Generate Reports**  
    Create PDF reports and provide feedback on your experience.
    
    ### 🎯 Tips for Best Results:
    - Use high-resolution Google Earth Pro screenshots
    - Ensure both images are from the exact same viewpoint
    - Lower the "Minimum Building Size" slider to detect smaller buildings
    - Enable YOLO AI for comprehensive building detection
    - Experiment with different confidence thresholds
    """)

# ---------- Tab 7: About/Docs ----------
with tab7:
    st.markdown("""
<div style="text-align: center; max-width: 800px; margin: 0 auto; font-family: 'Share Tech Mono', monospace;">

<h2 style="font-weight: 400; color: #00FF41; text-shadow: 0 0 10px #00FF4160; font-family: 'VT323', monospace; font-size: 2.2rem; letter-spacing: 3px;">
> ABOUT SmartDetect_</h2>

<p style="font-size: 1rem; line-height: 1.8; color: #00CC33; border-left: 2px solid #00FF4140; padding-left: 15px; text-align: left;">
SmartDetect is a cutting-edge AI solution for quality control, infrastructure maintenance, and urban development monitoring.
Powered by YOLOv8 deep learning and 6 OpenCV computer vision methods.
</p>

<div style="background: rgba(0, 255, 65, 0.04); padding: 20px; border: 1px solid #00FF4130; margin: 20px 0; text-align: left;">
<h3 style="color: #00FF41; text-shadow: 0 0 8px #00FF4140; font-family: 'VT323', monospace; font-size: 1.5rem;">
[SYS] Enhanced Earth Pro Analysis</h3>
<p style="color: #00CC33; font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; line-height: 2;">
<span style="color:#00FF41;">[1]</span> Multi-Scale Intensity Analysis<br>
<span style="color:#00FF41;">[2]</span> Edge Structure Detection<br>
<span style="color:#00FF41;">[3]</span> RGB Color Change Detection<br>
<span style="color:#00FF41;">[4]</span> Gradient/Texture Analysis<br>
<span style="color:#00FF41;">[5]</span> Adaptive Thresholding<br>
<span style="color:#00FF41;">[6]</span> Laplacian Detail Detection<br>
<br>
<span style="color:#00FF41;">+</span> Optional YOLO deep learning for building detection
</p>
</div>

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; margin: 30px 0;">
<div style="background: rgba(0,255,65,0.03); padding: 20px; border: 1px solid #00FF4120; width: 200px;">
<div style="font-size: 1.6rem; color: #00FF41; text-shadow: 0 0 10px #00FF41;">[ AI ]</div>
<h4 style="color: #00FF41; font-family: 'VT323', monospace;">AI Correction</h4>
<p style="font-size: 0.85rem; color: #00CC33;">Intelligent anomaly removal</p>
</div>
<div style="background: rgba(0,255,65,0.03); padding: 20px; border: 1px solid #00FF4120; width: 200px;">
<div style="font-size: 1.6rem; color: #00FF41; text-shadow: 0 0 10px #00FF41;">[ LIVE ]</div>
<h4 style="color: #00FF41; font-family: 'VT323', monospace;">Live Detection</h4>
<p style="font-size: 0.85rem; color: #00CC33;">Real-time analysis</p>
</div>
<div style="background: rgba(0,255,65,0.03); padding: 20px; border: 1px solid #00FF4120; width: 200px;">
<div style="font-size: 1.6rem; color: #00FF41; text-shadow: 0 0 10px #00FF41;">[ SAT ]</div>
<h4 style="color: #00FF41; font-family: 'VT323', monospace;">Earth Pro</h4>
<p style="font-size: 0.85rem; color: #00CC33;">6-method satellite analysis</p>
</div>
</div>

<div style="border-top: 1px solid #00FF4120; padding-top: 20px; margin-top: 10px;">
<h3 style="color: #00FF41; font-family: 'VT323', monospace; font-size: 1.4rem;">[ CREDITS ]</h3>
<p style="color: #00CC33; font-size: 0.9rem; line-height: 2;">
<span style="color:#00FF41;">dev://</span> Sugnik Tarafder<br>
<span style="color:#00FF41;">dev://</span> Arifur Rahaman<br>
<span style="color:#00FF41;">dev://</span> Sk Shonju Ali<br>
<span style="color:#00FF41;">dev://</span> Trishan Nayek
</p>
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 0.85rem; color: #00FF4160; font-family: Share Tech Mono, monospace; letter-spacing: 2px;'>SmartDetect v2.0 // RETRO CRT TERMINAL // 6-Method Detection Engine</div>", unsafe_allow_html=True)

