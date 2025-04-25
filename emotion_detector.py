import tkinter as tk
from tkinter import filedialog, ttk, Canvas
from PIL import Image, ImageTk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
from datetime import datetime

# Mock imports for demo purposes - replace with actual imports in your implementation
try:
    from keras.models import load_model
    from keras.preprocessing import image
    # Load the model
    model = load_model('emotion_recognition_model.h5')
    model_loaded = True
except:
    # For demonstration when model isn't available
    model_loaded = False
    
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_colors = {
    'angry': '#E53935',    # Red
    'disgust': '#8BC34A',  # Green
    'fear': '#9C27B0',     # Purple
    'happy': '#FFB300',    # Amber
    'neutral': '#42A5F5',  # Blue
    'sad': '#78909C',      # Blue Grey
    'surprise': '#FF9800'  # Orange
}

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection Dashboard")
        self.root.geometry("1100x700")
        self.root.minsize(900, 650)
        self.root.configure(bg="#f5f5f7")
        
        self.current_file = None
        self.history = []
        self.setup_ui()
        
    def setup_ui(self):
        # Custom title bar
        self.title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        self.title_frame.pack(fill=tk.X)
        
        # Logo and title
        title_label = tk.Label(self.title_frame, text="EMOTION DETECTION", 
                              font=("Montserrat", 18, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Version info
        version_label = tk.Label(self.title_frame, text="v1.0", 
                               font=("Montserrat", 10), fg="#bdc3c7", bg="#2c3e50")
        version_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Main content area - split into left and right frames
        self.content_frame = tk.Frame(self.root, bg="#f5f5f7")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for image and controls
        self.left_panel = tk.Frame(self.content_frame, bg="#f5f5f7", width=500)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right panel for results and charts
        self.right_panel = tk.Frame(self.content_frame, bg="#f5f5f7", width=500)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Control panel
        self.setup_control_panel()
        
        # Image display panel
        self.setup_image_panel()
        
        # Results panel
        self.setup_results_panel()
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg="#ecf0f1", height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", fg="#34495e", bg="#ecf0f1",
                                   font=("Segoe UI", 9))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Current time
        self.time_label = tk.Label(self.status_bar, text=self.get_current_time(), 
                                 fg="#34495e", bg="#ecf0f1", font=("Segoe UI", 9))
        self.time_label.pack(side=tk.RIGHT, padx=10)
        self.root.after(1000, self.update_time)
        
    def setup_control_panel(self):
        control_frame = tk.LabelFrame(self.left_panel, text="Controls", 
                                     font=("Segoe UI", 12), bg="#f5f5f7", fg="#34495e",
                                     padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons row
        btn_frame = tk.Frame(control_frame, bg="#f5f5f7")
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Upload Button
        upload_icon = "📁"  # Simple folder icon using unicode
        self.upload_btn = tk.Button(btn_frame, text=f"{upload_icon} Upload Image", 
                                  command=self.upload_image, bg="#3498db", fg="white",
                                  font=("Segoe UI", 11), padx=15, pady=8,
                                  activebackground="#2980b9", relief=tk.FLAT)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Camera Button (for future implementation)
        camera_icon = "📷"  # Simple camera icon using unicode
        self.camera_btn = tk.Button(btn_frame, text=f"{camera_icon} Camera", 
                                  bg="#7f8c8d", fg="white", font=("Segoe UI", 11),
                                  padx=15, pady=8, activebackground="#95a5a6", relief=tk.FLAT)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze Button
        analyze_icon = "🔍"  # Simple search icon using unicode
        self.analyze_btn = tk.Button(btn_frame, text=f"{analyze_icon} Analyze", 
                                   command=self.analyze_image, bg="#27ae60", fg="white",
                                   font=("Segoe UI", 11), padx=15, pady=8,
                                   activebackground="#2ecc71", relief=tk.FLAT,
                                   state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear Button
        clear_icon = "🗑️"  # Simple trash icon using unicode
        self.clear_btn = tk.Button(btn_frame, text=f"{clear_icon} Clear", 
                                 command=self.clear_display, bg="#e74c3c", fg="white",
                                 font=("Segoe UI", 11), padx=15, pady=8,
                                 activebackground="#c0392b", relief=tk.FLAT,
                                 state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
    def setup_image_panel(self):
        self.image_frame = tk.LabelFrame(self.left_panel, text="Image Preview", 
                                       font=("Segoe UI", 12), bg="#f5f5f7", fg="#34495e",
                                       padx=10, pady=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame to hold the actual image with a nice border
        display_frame = tk.Frame(self.image_frame, bg="white", bd=1, relief=tk.SUNKEN)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Default image or placeholder
        self.image_placeholder = tk.Label(display_frame, text="No image selected", 
                                        font=("Segoe UI", 14), fg="#95a5a6", bg="white")
        self.image_placeholder.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(display_frame, bg="white")
        
        # Image info frame
        self.info_frame = tk.Frame(self.image_frame, bg="#f5f5f7", height=30)
        self.info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.image_info_label = tk.Label(self.info_frame, text="", font=("Segoe UI", 9), 
                                       fg="#7f8c8d", bg="#f5f5f7", anchor=tk.W)
        self.image_info_label.pack(side=tk.LEFT)
        
    def setup_results_panel(self):
        # Results panel
        self.results_frame = tk.LabelFrame(self.right_panel, text="Analysis Results", 
                                         font=("Segoe UI", 12), bg="#f5f5f7", fg="#34495e",
                                         padx=10, pady=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Primary emotion display
        self.emotion_header = tk.Label(self.results_frame, text="Detected Emotion", 
                                     font=("Segoe UI", 14, "bold"), fg="#34495e", bg="#f5f5f7")
        self.emotion_header.pack(pady=(10, 5))
        
        self.emotion_label = tk.Label(self.results_frame, text="N/A", 
                                    font=("Montserrat", 28, "bold"), fg="#3498db", bg="#f5f5f7")
        self.emotion_label.pack(pady=(0, 5))
        
        self.confidence_label = tk.Label(self.results_frame, text="", 
                                       font=("Segoe UI", 12), fg="#7f8c8d", bg="#f5f5f7")
        self.confidence_label.pack(pady=(0, 10))
        
        # Create a frame for the chart
        self.chart_frame = tk.Frame(self.results_frame, bg="#f5f5f7")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the initial empty chart
        self.create_chart()
        
        # History section
        history_frame = tk.LabelFrame(self.right_panel, text="Recent History", 
                                     font=("Segoe UI", 12), bg="#f5f5f7", fg="#34495e",
                                     padx=10, pady=10, height=150)
        history_frame.pack(fill=tk.X, pady=(0, 10))
        history_frame.pack_propagate(False)  # Force the frame to keep its size
        
        # Create a Treeview widget for history
        self.history_tree = ttk.Treeview(history_frame, columns=("Time", "Emotion", "Confidence"), 
                                       show="headings", height=4)
        self.history_tree.heading("Time", text="Time")
        self.history_tree.heading("Emotion", text="Emotion")
        self.history_tree.heading("Confidence", text="Confidence")
        
        self.history_tree.column("Time", width=100)
        self.history_tree.column("Emotion", width=100)
        self.history_tree.column("Confidence", width=100)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
    def create_chart(self, data=None):
        # Clear previous chart if it exists
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
            
        # Create a new chart
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor('#f5f5f7')
        ax.set_facecolor('#f5f5f7')
        
        if data is None:
            # Display placeholder
            emotions = class_names
            confidence = [0] * len(emotions)
            colors = [emotion_colors[emotion] for emotion in emotions]
            
            ax.bar(emotions, confidence, color=colors, alpha=0.5)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Emotion Probability Distribution', fontsize=12, pad=10)
            plt.xticks(rotation=45, ha='right')
            
        else:
            # Display actual data
            emotions = class_names
            confidence = [round(val * 100, 1) for val in data]
            colors = [emotion_colors[emotion] for emotion in emotions]
            
            bars = ax.bar(emotions, confidence, color=colors)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Emotion Probability Distribution', fontsize=12, pad=10)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 1:  # Only label bars with significant values
                    ax.annotate(f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=8)
        
        plt.tight_layout()
        
        # Embed the chart in the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.current_file = file_path
            self.show_image(file_path)
            self.analyze_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            
            # Update status
            filename = os.path.basename(file_path)
            self.status_label.config(text=f"Loaded: {filename}")
            
            # Update image info
            img = Image.open(file_path)
            width, height = img.size
            size_kb = os.path.getsize(file_path) / 1024
            self.image_info_label.config(text=f"{filename} | {width}x{height} | {size_kb:.1f} KB")
            
    def show_image(self, path):
        # Hide placeholder
        self.image_placeholder.pack_forget()
        
        # Open and resize image while maintaining aspect ratio
        img = Image.open(path)
        img = self.resize_image(img, 400)
        photo = ImageTk.PhotoImage(img)
        
        # Show image
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
    def resize_image(self, img, base_width):
        # Resize image maintaining aspect ratio
        w_percent = base_width / float(img.size[0])
        h_size = int(float(img.size[1]) * float(w_percent))
        return img.resize((base_width, h_size), Image.LANCZOS)
    
    def analyze_image(self):
        if not self.current_file:
            return
            
        if model_loaded:
            # Actual analysis
            emotion, confidence, all_confidences = self.detect_emotion(self.current_file)
        else:
            # Demo mode - generate random results
            all_confidences = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
            predicted_index = np.argmax(all_confidences)
            emotion = class_names[predicted_index]
            confidence = all_confidences[predicted_index] * 100
            
        # Update results
        self.update_results(emotion, confidence, all_confidences)
        
        # Add to history
        self.add_to_history(emotion, confidence)
        
        # Update status
        self.status_label.config(text=f"Analysis complete: {emotion} detected")
        
    def detect_emotion(self, img_path):
        # Preprocess the image
        img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_emotion = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100
        
        return predicted_emotion, confidence, prediction[0]
    
    def update_results(self, emotion, confidence, all_confidences=None):
        # Update emotion label
        self.emotion_label.config(text=emotion.upper(), fg=emotion_colors[emotion])
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Update chart
        if all_confidences is not None:
            self.create_chart(all_confidences)
            
    def add_to_history(self, emotion, confidence):
        current_time = datetime.now().strftime("%H:%M:%S")
        confidence_str = f"{confidence:.1f}%"
        
        # Insert at the top of the list
        self.history_tree.insert("", 0, values=(current_time, emotion.capitalize(), confidence_str))
        
        # Keep only the last 10 entries
        all_items = self.history_tree.get_children()
        if len(all_items) > 10:
            self.history_tree.delete(all_items[-1])
            
    def clear_display(self):
        # Clear image
        self.image_label.pack_forget()
        self.image_placeholder.pack(fill=tk.BOTH, expand=True)
        
        # Reset results
        self.emotion_label.config(text="N/A", fg="#3498db")
        self.confidence_label.config(text="")
        self.create_chart()  # Reset chart
        
        # Reset status
        self.status_label.config(text="Ready")
        self.image_info_label.config(text="")
        
        # Reset buttons
        self.analyze_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        
        # Reset current file
        self.current_file = None
        
    def get_current_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def update_time(self):
        self.time_label.config(text=self.get_current_time())
        self.root.after(1000, self.update_time)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    
    # Set app icon (optional)
    # root.iconbitmap('path/to/icon.ico')
    
    # Custom style configuration for ttk widgets
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground="#333333")
    style.configure("Treeview.Heading", font=('Segoe UI', 9, 'bold'), background="#f0f0f0")
    
    root.mainloop()