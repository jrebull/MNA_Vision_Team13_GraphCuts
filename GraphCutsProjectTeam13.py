import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QFrame, QMessageBox, QSizePolicy,
                             QStyle, QGroupBox, QSlider, QColorDialog, QFontDialog,
                             QLineEdit, QCheckBox, QSpinBox, QTabWidget)
from PyQt5.QtGui import QIcon, QPixmap, QImage, QFont, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QSize, QRect

# Importamos el archivo mejorado
import GraphMakerTeam13 as GraphMaker


class ModernCutUI:

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("Tec de Monterrey - MNA - Team 13 - Vision (Enhanced)")
        self.window.setMinimumSize(1800, 1000)
        
        self.window.setWindowIcon(self.window.style().standardIcon(QStyle.SP_TitleBarMenuButton))

        self.graph_maker = GraphMaker.GraphMaker(max_dimension=900)
        self.current_file = None

        # --- Parámetros del Sticker ---
        self.sticker_border_color = QColor(Qt.white)
        self.sticker_text_color = QColor(Qt.black)
        self.sticker_border_thickness = 10
        self.sticker_text = ""
        self.sticker_text_font = QFont("Arial", 48, QFont.Bold)
        self.sticker_text_pos_factor = 0.5

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        header_label = QLabel("Tec de Monterrey / MNA / Team 13")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont("Segoe UI", 26, QFont.Bold))
        header_label.setStyleSheet("color: white;")

        subtitle_label = QLabel("Proyecto Final: Visión Computacional - Interactive GraphCut (Enhanced)")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Segoe UI", 12))
        subtitle_label.setStyleSheet("color: #cbd5f5;")

        header_container = QVBoxLayout()
        header_container.setSpacing(4)
        header_container.addWidget(header_label)
        header_container.addWidget(subtitle_label)

        title_frame = QFrame()
        title_frame.setLayout(header_container)
        title_frame.setStyleSheet("""
            QFrame {
                background-color: #1f2937;
                border-radius: 14px;
                padding: 12px;
            }
        """)

        main_layout.addWidget(title_frame)

        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(10)

        style = self.window.style()

        load_button = QPushButton("   Cargar imagen")
        load_button.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        load_button.clicked.connect(self.load_image)
        load_button.setObjectName("primaryButton")

        segment_button = QPushButton("   Segmentar")
        segment_button.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        segment_button.clicked.connect(self.segment_image)
        segment_button.setObjectName("accentButton")

        save_button = QPushButton("   Guardar Objeto")
        save_button.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        save_button.clicked.connect(self.save_image)
        save_button.setObjectName("saveButton")
        
        clear_button = QPushButton("   Limpiar semillas")
        clear_button.setIcon(style.standardIcon(QStyle.SP_TrashIcon))
        clear_button.clicked.connect(self.clear_seeds)
        clear_button.setObjectName("secondaryButton")

        self.seed_foreground_button = QPushButton("Semilla OBJETO")
        self.seed_foreground_button.setCheckable(True)
        self.seed_foreground_button.setChecked(True)
        self.seed_foreground_button.clicked.connect(self.set_seed_foreground_mode)
        self.seed_foreground_button.setObjectName("foregroundButton")

        self.seed_background_button = QPushButton("Semilla FONDO")
        self.seed_background_button.setCheckable(True)
        self.seed_background_button.clicked.connect(self.set_seed_background_mode)
        self.seed_background_button.setObjectName("backGroundButton")

        tools_layout.addWidget(load_button)
        tools_layout.addWidget(segment_button)
        tools_layout.addWidget(save_button)
        tools_layout.addWidget(clear_button)
        tools_layout.addStretch()
        tools_layout.addWidget(self.seed_foreground_button)
        tools_layout.addWidget(self.seed_background_button)

        main_layout.addLayout(tools_layout)

        # Layout principal con tabs para parámetros
        content_main_layout = QHBoxLayout()
        content_main_layout.setSpacing(12)

        # Panel izquierdo: Parámetros
        params_tab_widget = QTabWidget()
        params_tab_widget.setMaximumWidth(350)
        params_tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #4b5563;
                background-color: #1f2937;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #374151;
                color: #e5e7eb;
                padding: 8px 12px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4f46e5;
                color: white;
            }
        """)

        # Tab 1: Parámetros de Graph Cuts
        graphcut_params_widget = QWidget()
        graphcut_params_layout = QVBoxLayout(graphcut_params_widget)
        graphcut_params_layout.setSpacing(12)

        # Smoothness K
        k_group = QGroupBox("Smoothness (K)")
        k_group.setStyleSheet("QGroupBox { color: #e5e7eb; font-weight: bold; }")
        k_layout = QVBoxLayout(k_group)
        
        k_info = QLabel("Controla la suavidad de los bordes.\nMayor = bordes más suaves")
        k_info.setStyleSheet("color: #9ca3af; font-size: 10px;")
        k_info.setWordWrap(True)
        k_layout.addWidget(k_info)
        
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setRange(1, 200)
        self.k_slider.setValue(50)
        self.k_slider.setToolTip("Ajusta la suavidad")
        self.k_value_label = QLabel("50.0")
        self.k_value_label.setStyleSheet("color: #e5e7eb;")
        
        k_h_layout = QHBoxLayout()
        k_h_layout.addWidget(QLabel("K:"))
        k_h_layout.addWidget(self.k_slider)
        k_h_layout.addWidget(self.k_value_label)
        k_layout.addLayout(k_h_layout)
        
        self.k_slider.valueChanged.connect(lambda v: self.k_value_label.setText(f"{v}.0"))
        
        graphcut_params_layout.addWidget(k_group)

        # Edge Beta
        beta_group = QGroupBox("Edge Sensitivity (Beta)")
        beta_group.setStyleSheet("QGroupBox { color: #e5e7eb; font-weight: bold; }")
        beta_layout = QVBoxLayout(beta_group)
        
        beta_info = QLabel("Sensibilidad a diferencias de color.\nMayor = más sensible a cambios")
        beta_info.setStyleSheet("color: #9ca3af; font-size: 10px;")
        beta_info.setWordWrap(True)
        beta_layout.addWidget(beta_info)
        
        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setRange(1, 50)
        self.beta_slider.setValue(10)
        self.beta_slider.setToolTip("Ajusta sensibilidad a bordes")
        self.beta_value_label = QLabel("10.0")
        self.beta_value_label.setStyleSheet("color: #e5e7eb;")
        
        beta_h_layout = QHBoxLayout()
        beta_h_layout.addWidget(QLabel("Beta:"))
        beta_h_layout.addWidget(self.beta_slider)
        beta_h_layout.addWidget(self.beta_value_label)
        beta_layout.addLayout(beta_h_layout)
        
        self.beta_slider.valueChanged.connect(lambda v: self.beta_value_label.setText(f"{v}.0"))
        
        graphcut_params_layout.addWidget(beta_group)

        # GMM Options
        gmm_group = QGroupBox("Modelo de Color")
        gmm_group.setStyleSheet("QGroupBox { color: #e5e7eb; font-weight: bold; }")
        gmm_layout = QVBoxLayout(gmm_group)
        
        self.use_gmm_checkbox = QCheckBox("Usar GMM (más robusto)")
        self.use_gmm_checkbox.setStyleSheet("color: #e5e7eb;")
        self.use_gmm_checkbox.setToolTip("GMM modela mejor distribuciones complejas de color")
        gmm_layout.addWidget(self.use_gmm_checkbox)
        
        gmm_comp_layout = QHBoxLayout()
        gmm_comp_layout.addWidget(QLabel("Componentes GMM:"))
        self.gmm_components_spinbox = QSpinBox()
        self.gmm_components_spinbox.setRange(2, 5)
        self.gmm_components_spinbox.setValue(3)
        self.gmm_components_spinbox.setEnabled(False)
        gmm_comp_layout.addWidget(self.gmm_components_spinbox)
        gmm_layout.addLayout(gmm_comp_layout)
        
        self.use_gmm_checkbox.toggled.connect(self.gmm_components_spinbox.setEnabled)
        
        graphcut_params_layout.addWidget(gmm_group)

        # Refinement Options
        refine_group = QGroupBox("Refinamiento")
        refine_group.setStyleSheet("QGroupBox { color: #e5e7eb; font-weight: bold; }")
        refine_layout = QVBoxLayout(refine_group)
        
        # Guided Filter Radius
        radius_label = QLabel("Radio Guided Filter:")
        radius_label.setStyleSheet("color: #e5e7eb;")
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(3, 51)
        self.radius_slider.setValue(0)  # 0 = auto
        self.radius_slider.setSingleStep(2)
        self.radius_value_label = QLabel("Auto")
        self.radius_value_label.setStyleSheet("color: #e5e7eb;")
        
        radius_h_layout = QHBoxLayout()
        radius_h_layout.addWidget(radius_label)
        radius_h_layout.addWidget(self.radius_slider)
        radius_h_layout.addWidget(self.radius_value_label)
        refine_layout.addLayout(radius_h_layout)
        
        def update_radius_label(v):
            if v == 0:
                self.radius_value_label.setText("Auto")
            else:
                # Asegurar que sea impar
                v = v if v % 2 == 1 else v + 1
                self.radius_value_label.setText(str(v))
        
        self.radius_slider.valueChanged.connect(update_radius_label)
        
        # GrabCut refinement
        self.use_grabcut_checkbox = QCheckBox("Refinar con GrabCut")
        self.use_grabcut_checkbox.setStyleSheet("color: #e5e7eb;")
        self.use_grabcut_checkbox.setToolTip("Post-procesamiento iterativo (más lento)")
        refine_layout.addWidget(self.use_grabcut_checkbox)
        
        graphcut_params_layout.addWidget(refine_group)
        
        graphcut_params_layout.addStretch()

        params_tab_widget.addTab(graphcut_params_widget, "Graph Cuts")

        # Tab 2: Parámetros de Sticker (original)
        sticker_params_widget = QWidget()
        sticker_params_layout = QVBoxLayout(sticker_params_widget)
        sticker_params_layout.setSpacing(8)

        border_label = QLabel("Grosor del Borde:")
        border_label.setStyleSheet("color: #e5e7eb;")
        self.border_thickness_slider = QSlider(Qt.Horizontal)
        self.border_thickness_slider.setRange(0, 50)
        self.border_thickness_slider.setValue(self.sticker_border_thickness)
        self.border_thickness_slider.setToolTip("Ajusta el grosor del borde del sticker")
        self.border_thickness_slider.valueChanged.connect(self._update_sticker_preview)
        self.border_thickness_value_label = QLabel(f"{self.sticker_border_thickness} px")
        self.border_thickness_value_label.setStyleSheet("color: #e5e7eb;")
        
        border_thickness_h_layout = QHBoxLayout()
        border_thickness_h_layout.addWidget(border_label)
        border_thickness_h_layout.addWidget(self.border_thickness_slider)
        border_thickness_h_layout.addWidget(self.border_thickness_value_label)
        self.border_thickness_slider.valueChanged.connect(lambda v: self.border_thickness_value_label.setText(f"{v} px"))
        
        sticker_params_layout.addLayout(border_thickness_h_layout)

        border_color_button = QPushButton("Color del Borde")
        border_color_button.clicked.connect(self._pick_border_color)
        border_color_button.setObjectName("colorButton")
        sticker_params_layout.addWidget(border_color_button)

        text_input_label = QLabel("Texto del Sticker:")
        text_input_label.setStyleSheet("color: #e5e7eb;")
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Escribe tu texto aquí...")
        self.text_input.textChanged.connect(self._update_sticker_preview)
        sticker_params_layout.addWidget(text_input_label)
        sticker_params_layout.addWidget(self.text_input)

        text_color_button = QPushButton("Color del Texto")
        text_color_button.clicked.connect(self._pick_text_color)
        text_color_button.setObjectName("colorButton")
        sticker_params_layout.addWidget(text_color_button)

        text_font_button = QPushButton("Fuente del Texto")
        text_font_button.clicked.connect(self._pick_text_font)
        text_font_button.setObjectName("colorButton")
        sticker_params_layout.addWidget(text_font_button)

        text_pos_label = QLabel("Posición Vertical del Texto:")
        text_pos_label.setStyleSheet("color: #e5e7eb;")
        self.text_pos_slider = QSlider(Qt.Horizontal)
        self.text_pos_slider.setRange(0, 100)
        self.text_pos_slider.setValue(int(self.sticker_text_pos_factor * 100))
        self.text_pos_slider.setToolTip("Ajusta la posición vertical del texto")
        self.text_pos_slider.valueChanged.connect(self._update_sticker_preview)
        self.text_pos_value_label = QLabel(f"{int(self.sticker_text_pos_factor * 100)}%")
        self.text_pos_value_label.setStyleSheet("color: #e5e7eb;")
        
        text_pos_h_layout = QHBoxLayout()
        text_pos_h_layout.addWidget(text_pos_label)
        text_pos_h_layout.addWidget(self.text_pos_slider)
        text_pos_h_layout.addWidget(self.text_pos_value_label)
        self.text_pos_slider.valueChanged.connect(lambda v: self.text_pos_value_label.setText(f"{v}%"))
        
        sticker_params_layout.addLayout(text_pos_h_layout)
        sticker_params_layout.addStretch()

        params_tab_widget.addTab(sticker_params_widget, "Sticker")

        content_main_layout.addWidget(params_tab_widget)

        # Panel central: Imágenes
        images_layout = QHBoxLayout()
        images_layout.setSpacing(12)

        left_layout = QVBoxLayout()
        middle_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.setSpacing(6)
        middle_layout.setSpacing(6)
        right_layout.setSpacing(6)

        def create_title_label(text):
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Segoe UI", 11, QFont.Bold))
            label.setStyleSheet("color: white;")
            return label

        left_layout.addWidget(create_title_label("1. Original + Semillas"))
        middle_layout.addWidget(create_title_label("2. Objeto Segmentado"))
        right_layout.addWidget(create_title_label("3. Fondo Segmentado"))

        self.seedLabel = ClickableLabel(self, "seed")
        self.segmentLabel = ClickableLabel(self, "segment")
        self.inverseSegmentLabel = ClickableLabel(self, "inverse")

        for lbl in [self.seedLabel, self.segmentLabel, self.inverseSegmentLabel]:
            lbl.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            lbl.setLineWidth(2)
            lbl.setStyleSheet("background-color: #111827; border-radius: 12px; border: 1px solid #4b5563;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 320)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_layout.addWidget(self.seedLabel)
        middle_layout.addWidget(self.segmentLabel)
        right_layout.addWidget(self.inverseSegmentLabel)

        images_layout.addLayout(left_layout)
        images_layout.addLayout(middle_layout)
        images_layout.addLayout(right_layout)

        content_main_layout.addLayout(images_layout, stretch=3)

        # Panel derecho: Histogramas y Sticker
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(12)

        histogram_layout = QVBoxLayout()
        histogram_layout.setSpacing(6)
        histogram_layout.addWidget(create_title_label("Histograma BGR"))

        self.histogramLabel = QLabel()
        self.histogramLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.histogramLabel.setLineWidth(2)
        self.histogramLabel.setStyleSheet("background-color: #111827; border-radius: 12px; border: 1px solid #4b5563;")
        self.histogramLabel.setAlignment(Qt.AlignCenter)
        self.histogramLabel.setMinimumSize(280, 200)
        histogram_layout.addWidget(self.histogramLabel)

        # Histograma Lab
        lab_histogram_layout = QVBoxLayout()
        lab_histogram_layout.setSpacing(6)
        lab_histogram_layout.addWidget(create_title_label("Histograma Lab"))

        self.labHistogramLabel = QLabel()
        self.labHistogramLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.labHistogramLabel.setLineWidth(2)
        self.labHistogramLabel.setStyleSheet("background-color: #111827; border-radius: 12px; border: 1px solid #4b5563;")
        self.labHistogramLabel.setAlignment(Qt.AlignCenter)
        self.labHistogramLabel.setMinimumSize(280, 200)
        lab_histogram_layout.addWidget(self.labHistogramLabel)

        sticker_layout = QVBoxLayout()
        sticker_layout.setSpacing(6)
        sticker_layout.addWidget(create_title_label("4. Sticker Preview"))

        self.stickerLabel = QLabel()
        self.stickerLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.stickerLabel.setLineWidth(2)
        self.stickerLabel.setStyleSheet("background-color: #111827; border-radius: 12px; border: 1px solid #4b5563;")
        self.stickerLabel.setAlignment(Qt.AlignCenter)
        self.stickerLabel.setMinimumSize(280, 280)
        self.stickerLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sticker_layout.addWidget(self.stickerLabel)

        right_panel_layout.addLayout(histogram_layout)
        right_panel_layout.addLayout(lab_histogram_layout)
        right_panel_layout.addLayout(sticker_layout)

        content_main_layout.addLayout(right_panel_layout)

        main_layout.addLayout(content_main_layout)

        self._apply_stylesheet()

    def _apply_stylesheet(self):
        self.window.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #e5e7eb;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #374151;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
            QPushButton:pressed {
                background-color: #1f2937;
            }
            QPushButton#primaryButton {
                background-color: #2563eb;
            }
            QPushButton#primaryButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton#accentButton {
                background-color: #10b981;
            }
            QPushButton#accentButton:hover {
                background-color: #059669;
            }
            QPushButton#saveButton {
                background-color: #8b5cf6;
            }
            QPushButton#saveButton:hover {
                background-color: #7c3aed;
            }
            QPushButton#secondaryButton {
                background-color: #ef4444;
            }
            QPushButton#secondaryButton:hover {
                background-color: #dc2626;
            }
            QPushButton#foregroundButton {
                background-color: #10b981;
                border: 2px solid transparent;
            }
            QPushButton#foregroundButton:hover {
                background-color: #059669;
            }
            QPushButton#foregroundButton:checked {
                background-color: #047857;
                border: 2px solid #d1fae5;
            }
            QPushButton#backGroundButton {
                background-color: #3b82f6;
                border: 2px solid transparent;
            }
            QPushButton#backGroundButton:hover {
                background-color: #2563eb;
            }
            QPushButton#backGroundButton:checked {
                background-color: #1e40af;
                border: 2px solid #bfdbfe;
            }
            QPushButton#colorButton {
                background-color: #6366f1;
            }
            QPushButton#colorButton:hover {
                background-color: #4f46e5;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4b5563;
                height: 8px;
                background: #1f2937;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 1px solid #4f46e5;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #818cf8;
            }
            QLineEdit {
                background-color: #1f2937;
                border: 1px solid #4b5563;
                border-radius: 6px;
                padding: 8px;
                color: #e5e7eb;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #6366f1;
            }
            QCheckBox {
                spacing: 8px;
                color: #e5e7eb;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #4b5563;
                background-color: #1f2937;
            }
            QCheckBox::indicator:checked {
                background-color: #6366f1;
                border-color: #4f46e5;
            }
            QSpinBox {
                background-color: #1f2937;
                border: 1px solid #4b5563;
                border-radius: 6px;
                padding: 6px;
                color: #e5e7eb;
            }
            QGroupBox {
                border: 1px solid #4b5563;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #1f2937;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.window, "Cargar Imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if filename:
            try:
                self.current_file = filename
                self.graph_maker.load_image(filename)
                self._update_all_displays()
                QMessageBox.information(self.window, "Éxito", "Imagen cargada correctamente.")
            except Exception as e:
                QMessageBox.critical(self.window, "Error", f"No se pudo cargar la imagen:\n{str(e)}")

    def segment_image(self):
        if self.graph_maker.image is None:
            QMessageBox.warning(self.window, "Advertencia", "Primero carga una imagen.")
            return

        if len(self.graph_maker.foreground_seeds) == 0 or len(self.graph_maker.background_seeds) == 0:
            QMessageBox.warning(self.window, "Advertencia",
                              "Debes marcar semillas de OBJETO (verde) y FONDO (azul).")
            return

        try:
            # Aplicar parámetros desde la UI
            self.graph_maker.smoothness_K = float(self.k_slider.value())
            self.graph_maker.edge_beta = float(self.beta_slider.value())
            self.graph_maker.use_gmm = self.use_gmm_checkbox.isChecked()
            self.graph_maker.gmm_components = self.gmm_components_spinbox.value()
            
            radius_val = self.radius_slider.value()
            if radius_val == 0:
                self.graph_maker.guided_filter_radius = None
            else:
                # Asegurar que sea impar
                radius_val = radius_val if radius_val % 2 == 1 else radius_val + 1
                self.graph_maker.guided_filter_radius = radius_val
            
            self.graph_maker.use_grabcut_refine = self.use_grabcut_checkbox.isChecked()
            
            self.graph_maker.create_segment(display=True)
            self._update_all_displays()
            QMessageBox.information(self.window, "Éxito",
                                  f"Segmentación completada!\n"
                                  f"K={self.graph_maker.smoothness_K:.1f}, "
                                  f"Beta={self.graph_maker.edge_beta:.1f}\n"
                                  f"GMM={'Sí' if self.graph_maker.use_gmm else 'No'}, "
                                  f"GrabCut={'Sí' if self.graph_maker.use_grabcut_refine else 'No'}")
        except Exception as e:
            QMessageBox.critical(self.window, "Error", f"Error en la segmentación:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def save_image(self):
        if self.graph_maker.mask is None:
            QMessageBox.warning(self.window, "Advertencia", "Primero realiza la segmentación.")
            return

        # Menú de opciones
        reply = QMessageBox.question(
            self.window, 
            "Guardar", 
            "¿Qué deseas guardar?\n\n"
            "YES = Sticker con borde y texto (PNG transparente)\n"
            "NO = Objeto segmentado sin overlay (PNG transparente)\n"
            "CANCEL = Cancelar",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Cancel:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self.window, "Guardar Imagen", "",
            "PNG con transparencia (*.png)"
        )
        
        if not filename:
            return
        
        try:
            if reply == QMessageBox.Yes:
                # Guardar STICKER con borde y texto
                border_thickness = self.border_thickness_slider.value()
                border_color_rgb = (self.sticker_border_color.red(),
                                  self.sticker_border_color.green(),
                                  self.sticker_border_color.blue())
                
                text = self.text_input.text()
                text_color_rgb = (self.sticker_text_color.red(),
                                self.sticker_text_color.green(),
                                self.sticker_text_color.blue())
                
                if isinstance(self.sticker_text_font, QFont):
                    font_family = self.sticker_text_font.family()
                    font_size = self.sticker_text_font.pointSize()
                    font_weight = self.sticker_text_font.weight()
                else:
                    font_family = "Arial"
                    font_size = 48
                    font_weight = QFont.Bold
                
                text_pos_factor = self.text_pos_slider.value() / 100.0
                
                sticker_full = self.graph_maker.save_sticker_image(
                    filename="",
                    border_thickness=border_thickness,
                    border_color_rgb=border_color_rgb,
                    text=text,
                    text_color_rgb=text_color_rgb,
                    font_family=font_family,
                    font_size=font_size,
                    font_weight=font_weight,
                    text_pos_factor=text_pos_factor,
                    is_preview=False
                )
                
                if sticker_full is not None:
                    cv2.imwrite(filename, sticker_full)
                    QMessageBox.information(self.window, "Éxito", 
                                          f"Sticker guardado en:\n{filename}")
            
            else:  # No = Objeto sin overlay
                # Guardar objeto recortado SIN overlay rosa, con transparencia
                h_orig, w_orig = self.graph_maker.image_bgr_original.shape[:2]
                
                if self.graph_maker.mask.shape[:2] != (h_orig, w_orig):
                    mask_full_res = cv2.resize(self.graph_maker.mask, (w_orig, h_orig), 
                                             interpolation=cv2.INTER_LINEAR)
                else:
                    mask_full_res = self.graph_maker.mask.copy()
                
                # Crear BGRA con transparencia
                b, g, r = cv2.split(self.graph_maker.image_bgr_original)
                object_bgra = cv2.merge([b, g, r, mask_full_res])
                
                cv2.imwrite(filename, object_bgra)
                QMessageBox.information(self.window, "Éxito", 
                                      f"Objeto guardado en:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self.window, "Error", f"No se pudo guardar:\n{str(e)}")

    def clear_seeds(self):
        self.graph_maker.foreground_seeds = []
        self.graph_maker.background_seeds = []
        self._update_seed_label()
        QMessageBox.information(self.window, "Limpieza", "Semillas eliminadas.")

    def set_seed_foreground_mode(self):
        self.seed_foreground_button.setChecked(True)
        self.seed_background_button.setChecked(False)

    def set_seed_background_mode(self):
        self.seed_foreground_button.setChecked(False)
        self.seed_background_button.setChecked(True)

    def _pick_border_color(self):
        color = QColorDialog.getColor(self.sticker_border_color, self.window, "Seleccionar Color del Borde")
        if color.isValid():
            self.sticker_border_color = color
            self._update_sticker_preview()

    def _pick_text_color(self):
        color = QColorDialog.getColor(self.sticker_text_color, self.window, "Seleccionar Color del Texto")
        if color.isValid():
            self.sticker_text_color = color
            self._update_sticker_preview()

    def _pick_text_font(self):
        # Asegurar que tenemos un QFont válido
        if not isinstance(self.sticker_text_font, QFont):
            self.sticker_text_font = QFont("Arial", 48, QFont.Bold)
        
        result = QFontDialog.getFont(self.sticker_text_font, self.window, "Seleccionar Fuente del Texto")
        
        # Verificar qué es cada cosa
        if isinstance(result, tuple) and len(result) == 2:
            font, ok = result  # Orden correcto
            if ok and isinstance(font, QFont):
                self.sticker_text_font = font
                print(f"Fuente cambiada a: {font.family()}, Tamaño: {font.pointSize()}, Bold: {font.bold()}")
                self._update_sticker_preview()
            else:
                print("Diálogo cancelado o fuente inválida")
        else:
            print(f"Resultado inesperado del diálogo: {type(result)}")
    
    def _update_sticker_preview(self):
        if self.graph_maker.mask is None:
            self.stickerLabel.clear()
            self.stickerLabel.setText("Presiona 'Segmentar' primero")
            self.stickerLabel.setAlignment(Qt.AlignCenter)
            return
        
        border_thickness = self.border_thickness_slider.value()
        border_color_rgb = (self.sticker_border_color.red(),
                          self.sticker_border_color.green(),
                          self.sticker_border_color.blue())
        
        text = self.text_input.text()
        text_color_rgb = (self.sticker_text_color.red(),
                        self.sticker_text_color.green(),
                        self.sticker_text_color.blue())
        
        # Obtener info de la fuente
        if isinstance(self.sticker_text_font, QFont):
            font_family = self.sticker_text_font.family()
            font_size = self.sticker_text_font.pointSize()
            # Convertir weight de QFont a int
            if self.sticker_text_font.bold():
                font_weight = 75  # Bold en PIL
            else:
                font_weight = 50  # Normal en PIL
        else:
            font_family = "Arial"
            font_size = 48
            font_weight = 75
        
        text_pos_factor = self.text_pos_slider.value() / 100.0
        
        print(f"Generando preview con fuente: {font_family}, size: {font_size}")  # Debug
        
        sticker_preview = self.graph_maker.save_sticker_image(
            filename="",
            border_thickness=border_thickness,
            border_color_rgb=border_color_rgb,
            text=text,
            text_color_rgb=text_color_rgb,
            font_family=font_family,
            font_size=font_size,
            font_weight=font_weight,
            text_pos_factor=text_pos_factor,
            is_preview=True,
            preview_size=(self.stickerLabel.width(), self.stickerLabel.height())
        )
        
        if sticker_preview is not None:
            q_img = self.get_qimage_rgba(sticker_preview)
            pixmap = QPixmap.fromImage(q_img)
            
            scaled_pixmap = pixmap.scaled(
                self.stickerLabel.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.stickerLabel.setPixmap(scaled_pixmap)
            
    def _update_all_displays(self):
        self._update_seed_label()
        self._update_segment_label()
        self._update_inverse_segment_label()
        self._update_histogram()
        self._update_lab_histogram()
        self._update_sticker_preview()

    def _display_histogram(self, hist_data, label_widget, colors_dict):
        if not hist_data:
            label_widget.clear()
            return

        w = label_widget.width()
        h = label_widget.height()
        if w <= 0 or h <= 0:
            w, h = 420, 300

        pixmap = QPixmap(w, h)
        pixmap.fill(QColor("#2d3748"))

        painter = QPainter(pixmap)
        margin_left, margin_right, margin_top, margin_bottom = 15, 15, 10, 20
        hist_w = w - margin_left - margin_right
        hist_h = h - margin_top - margin_bottom

        painter.setPen(QColor(200, 200, 200))
        painter.drawLine(margin_left, h - margin_bottom, margin_left + hist_w, h - margin_bottom)

        max_val = 0.0
        for ch_data in hist_data.values():
            max_val = max(max_val, np.max(ch_data) if len(ch_data) > 0 else 0.0)
        if max_val <= 0: max_val = 1.0

        bar_w = max(1, int(hist_w / 256.0))

        for ch_name, ch_values in hist_data.items():
            color = colors_dict.get(ch_name, QColor(255, 255, 255))
            painter.setPen(color)
            for j, v in enumerate(ch_values):
                y_val = float(v) / float(max_val)
                if y_val <= 0.0: continue
                x = int(j * (hist_w / 256.0)) + margin_left + int(bar_w / 2)
                y1 = margin_top + hist_h
                y2 = margin_top + int(hist_h * (1.0 - y_val))
                painter.drawLine(x, y1, x, y2)
        painter.end()
        label_widget.setPixmap(pixmap)

    def _update_seed_label(self):
        full_img = self.graph_maker.get_seed_overlay_image()
        if full_img is None:
            self.seedLabel.set_image_data(None, "Carga una imagen (Botón Superior Izq.)")
            return
        q_img = self.get_qimage(full_img)
        pixmap = QPixmap.fromImage(q_img)
        self.seedLabel.set_image_data(pixmap)

    def _update_segment_label(self):
        full_img = self.graph_maker.get_segmented_image_for_display()
        if full_img is None:
            self.segmentLabel.set_image_data(None, "Presiona 'Segmentar'")
            return
        q_img = self.get_qimage(full_img)
        pixmap = QPixmap.fromImage(q_img)
        self.segmentLabel.set_image_data(pixmap)

    def _update_inverse_segment_label(self):
        full_img = self.graph_maker.get_inverse_segmented_image_for_display()
        if full_img is None:
            self.inverseSegmentLabel.set_image_data(None, "Presiona 'Segmentar'")
            return
        q_img = self.get_qimage(full_img)
        pixmap = QPixmap.fromImage(q_img)
        self.inverseSegmentLabel.set_image_data(pixmap)

    def _update_histogram(self):
        hist = self.graph_maker.get_histograms()
        if hist is None:
            self.histogramLabel.clear()
            self.histogramLabel.setText("Sin histograma")
            self.histogramLabel.setAlignment(Qt.AlignCenter)
        else:
            colors = {
                'b': QColor(66, 135, 245),
                'g': QColor(86, 217, 127),
                'r': QColor(244, 112, 112)
            }
            self._display_histogram(hist, self.histogramLabel, colors)
    
    def _update_lab_histogram(self):
        hist = self.graph_maker.get_lab_histograms()
        if hist is None:
            self.labHistogramLabel.clear()
            self.labHistogramLabel.setText("Sin histograma")
            self.labHistogramLabel.setAlignment(Qt.AlignCenter)
        else:
            colors = {
                'L': QColor(200, 200, 200),
                'a': QColor(255, 100, 150),
                'b': QColor(100, 150, 255)
            }
            self._display_histogram(hist, self.labHistogramLabel, colors)

    def get_qimage(self, img_bgr):
        if img_bgr is None:
            return QImage()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        return QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def get_qimage_rgba(self, img_bgra):
        if img_bgra is None:
            return QImage()
        
        img_rgba = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
        h, w, ch = img_rgba.shape
        bytes_per_line = ch * w
        return QImage(img_rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888)

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())


class ClickableLabel(QLabel):
    def __init__(self, parent_ui, label_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_ui = parent_ui
        self.label_type = label_type
        self.drawing = False
        self.last_pos = None
        self.brush_radius = 5
        self.pixmap = None
        self.message = ""

    def set_image_data(self, pixmap, message=""):
        self.pixmap = pixmap
        self.message = message
        self.update()

    def mousePressEvent(self, event):
        if self.label_type != "seed" or self.pixmap is None:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_pos = event.pos()
            self.add_seed_under_cursor(event.pos())

    def mouseMoveEvent(self, event):
        if self.label_type != "seed" or not self.drawing or self.pixmap is None:
            return
        self.add_seed_under_cursor(event.pos())
        self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if self.label_type != "seed": return
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def add_seed_under_cursor(self, pos):
        if self.parent_ui.graph_maker.image_bgr_original is None or self.pixmap is None or self.pixmap.isNull():
            return

        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = (self.width() - scaled_pixmap.width()) / 2
        offset_y = (self.height() - scaled_pixmap.height()) / 2
        target_rect = QRect(int(offset_x), int(offset_y), scaled_pixmap.width(), scaled_pixmap.height())

        if not target_rect.contains(pos):
            return

        relative_x = pos.x() - target_rect.x()
        relative_y = pos.y() - target_rect.y()
        if target_rect.width() == 0 or target_rect.height() == 0:
            return
            
        frac_x = relative_x / target_rect.width()
        frac_y = relative_y / target_rect.height()

        img_x = int(frac_x * self.pixmap.width())
        img_y = int(frac_y * self.pixmap.height())

        if self.parent_ui.seed_foreground_button.isChecked():
            seed_type = self.parent_ui.graph_maker.foreground
        else:
            seed_type = self.parent_ui.graph_maker.background
        
        self.parent_ui.graph_maker.add_seed(img_x, img_y, seed_type)
        self.parent_ui._update_seed_label()

    def wheelEvent(self, event):
        if self.label_type != "seed": return
        delta = event.angleDelta().y()
        if delta > 0:
            self.brush_radius = min(self.brush_radius + 1, 50)
        else:
            self.brush_radius = max(self.brush_radius - 1, 1)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.pixmap is None or self.pixmap.isNull():
            painter.setPen(QColor("#e5e7eb"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
        else:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled_pixmap.width()) / 2
            y = (self.height() - scaled_pixmap.height()) / 2
            painter.drawPixmap(int(x), int(y), scaled_pixmap)
        painter.end()


if __name__ == "__main__":
    modernUI = ModernCutUI()
    modernUI.run()