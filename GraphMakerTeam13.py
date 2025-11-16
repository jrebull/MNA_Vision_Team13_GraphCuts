import cv2
import numpy as np
import maxflow
try:
    import cv2.ximgproc
    HAS_XIMGPROC = True
except ImportError:
    print("Advertencia: cv2.ximgproc no encontrado. Instale 'opencv-contrib-python' para un mejor refinamiento de bordes.")
    print("Usando desenfoque Gaussiano como alternativa.")
    HAS_XIMGPROC = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
    print("Pillow (PIL) encontrado. La función de texto para stickers está habilitada.")
except ImportError:
    print("Advertencia: Pillow (PIL) no encontrado. Instale 'Pillow' para habilitar la función de texto en stickers.")
    HAS_PIL = False

from sklearn.mixture import GaussianMixture

class GraphMaker:
    """
    Motor de GraphCut mejorado con parámetros ajustables y GMM.
    """
    def __init__(self, max_dimension=1000):
        self.image_bgr_original = None
        self.image_bgr = None
        self.image = None
        self.image_lab = None
        self.edge_mag = None
        self.mask = None
        
        self.foreground = 1
        self.background = 0
        self.background_seeds = []
        self.foreground_seeds = []
        
        self.MAX_DIMENSION = max_dimension
        self.downsample_scale = 1.0
        
        self.MASK_COLOR_RGB = (255, 105, 180)  # Rosa
        self.MASK_ALPHA = 0.6
        
        # Parámetros ajustables
        self.smoothness_K = 50.0
        self.edge_beta = 10.0
        self.use_gmm = False
        self.gmm_components = 3
        self.guided_filter_radius = None  # None = auto
        self.use_grabcut_refine = False

    def load_image(self, filename):
        image_bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise IOError(f"No se pudo cargar la imagen: {filename}")
        self.load_image_data(image_bgr)

    def load_image_data(self, image_data):
        self.image_bgr_original = image_data.copy()
        
        h, w = image_data.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.MAX_DIMENSION:
            self.downsample_scale = self.MAX_DIMENSION / max_dim
            print(f"📉 Downsampling imagen: {w}x{h} → {int(w * self.downsample_scale)}x{int(h * self.downsample_scale)}")
            self.image_bgr = cv2.resize(image_data, 
                                       (int(w * self.downsample_scale), int(h * self.downsample_scale)),
                                       interpolation=cv2.INTER_AREA)
        else:
            self.downsample_scale = 1.0
            self.image_bgr = image_data.copy()

        self.image = self.image_bgr
        self.image_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        self.mask = None
        self.background_seeds = []
        self.foreground_seeds = []
        self.nodes = []
        self.edges = []

    def add_seed(self, x, y, type):
        if self.image_bgr_original is None:
            return
        
        x_scaled = int(x * self.downsample_scale)
        y_scaled = int(y * self.downsample_scale)
            
        if y_scaled >= self.image.shape[0] or x_scaled >= self.image.shape[1] or y_scaled < 0 or x_scaled < 0:
            return

        if type == self.background:
            if (x_scaled, y_scaled) not in self.background_seeds:
                self.background_seeds.append((x_scaled, y_scaled))
        elif type == self.foreground:
            if (x_scaled, y_scaled) not in self.foreground_seeds:
                self.foreground_seeds.append((x_scaled, y_scaled))

    def _build_color_models(self):
        print(f"Construyendo modelos de color (Lab) - {'GMM' if self.use_gmm else 'Gaussiana simple'}...")
        
        if len(self.foreground_seeds) == 0:
            raise ValueError("No hay píxeles de objeto")
        if len(self.background_seeds) == 0:
            raise ValueError("No hay píxeles de fondo")

        fg_indices_t = np.array(self.foreground_seeds).T
        bg_indices_t = np.array(self.background_seeds).T

        fg_intensities = self.image_lab[fg_indices_t[1], fg_indices_t[0], :]
        bg_intensities = self.image_lab[bg_indices_t[1], bg_indices_t[0], :]

        if self.use_gmm and fg_intensities.shape[0] > 10 and bg_intensities.shape[0] > 10:
            # Usar GMM para modelado más robusto
            n_components = min(self.gmm_components, fg_intensities.shape[0] // 3, bg_intensities.shape[0] // 3)
            
            self.fg_gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            self.bg_gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            
            self.fg_gmm.fit(fg_intensities)
            self.bg_gmm.fit(bg_intensities)
            print(f"   ✓ GMM: {n_components} componentes por clase")
        else:
            # Usar Gaussiana simple (original)
            self.fg_mean = np.mean(fg_intensities, axis=0)
            self.bg_mean = np.mean(bg_intensities, axis=0)

            if fg_intensities.shape[0] > 1:
                self.fg_cov = np.cov(fg_intensities.T)
            else:
                self.fg_cov = np.eye(3)

            if bg_intensities.shape[0] > 1:
                self.bg_cov = np.cov(bg_intensities.T)
            else:
                self.bg_cov = np.eye(3)

            self.fg_cov_inv = np.linalg.inv(self.fg_cov + np.eye(3) * 1e-3)
            self.bg_cov_inv = np.linalg.inv(self.bg_cov + np.eye(3) * 1e-3)

            sign_fg, logdet_fg = np.linalg.slogdet(self.fg_cov)
            self.fg_const = 0.5 * logdet_fg
            sign_bg, logdet_bg = np.linalg.slogdet(self.bg_cov)
            self.bg_const = 0.5 * logdet_bg
            print("   ✓ Gaussiana simple construida")

    def _build_edge_model(self):
        print("Construyendo modelo de bordes...")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        edges_canny = cv2.Canny(gray, threshold1=50, threshold2=150)
        edges_float = edges_canny.astype(np.float32) / 255.0
        edges_blur = cv2.GaussianBlur(edges_float, (3, 3), 0)
        
        self.edge_mag = edges_blur
        print("   ✓ Modelo de bordes listo.")

    def _gaussian_probability(self, x, mean, cov_inv, const):
        diff = x - mean
        return np.sum(diff * (cov_inv @ diff)) / 2 + const
    
    def _gmm_probability(self, x, gmm_model):
        # Retorna -log(P(x)) como costo
        log_prob = gmm_model.score_samples(x.reshape(1, -1))[0]
        return -log_prob

    def _build_graph(self):
        if self.image is None:
            raise ValueError("No hay imagen cargada.")
        
        print(f"Construyendo grafo (K={self.smoothness_K:.1f}, Beta={self.edge_beta:.1f})...")

        h, w = self.image.shape[:2]
        num_nodes = w * h

        g = maxflow.Graph[float](num_nodes, 4 * num_nodes)
        nodes = g.add_nodes(num_nodes)
        
        L = self.image_lab.reshape((-1, 3)).astype(np.float32)

        self._build_color_models()
        self._build_edge_model()

        max_weight = 0

        print("Agregando términos de datos (data term)...")
        for y in range(h):
            for x in range(w):
                idx = self.get_node_num(x, y, (h, w))
                color = L[idx]
                
                # Calcular costos según el modelo
                if self.use_gmm and hasattr(self, 'fg_gmm'):
                    df = self._gmm_probability(color, self.fg_gmm)
                    db = self._gmm_probability(color, self.bg_gmm)
                else:
                    df = self._gaussian_probability(color, self.fg_mean, self.fg_cov_inv, self.fg_const)
                    db = self._gaussian_probability(color, self.bg_mean, self.bg_cov_inv, self.bg_const)

                cap_source = df
                cap_sink = db

                # Semillas duras
                if (x, y) in self.foreground_seeds:
                    cap_source = 0
                    cap_sink = 1e9
                elif (x, y) in self.background_seeds:
                    cap_source = 1e9
                    cap_sink = 0

                g.add_tedge(nodes[idx], cap_source, cap_sink)
                max_weight = max(max_weight, cap_source, cap_sink)

        print("Agregando términos de borde (smoothness)...")
        
        K = self.smoothness_K
        beta = self.edge_beta
        
        L_norm = L.reshape((h, w, 3))
        
        diff_y = np.sum((L_norm[1:, :, :] - L_norm[:-1, :, :])**2, axis=2)
        diff_x = np.sum((L_norm[:, 1:, :] - L_norm[:, :-1, :])**2, axis=2)
        
        if diff_x.size > 0 and diff_y.size > 0:
            mean_diff = (np.mean(diff_x) + np.mean(diff_y)) / 2.0
            if mean_diff > 1e-6:
                beta = beta / mean_diff
            else:
                beta = 10.0

        scale_factor = K
        
        for y in range(h):
            for x in range(w):
                idx_c = self.get_node_num(x, y, (h, w))
                
                if x > 0:
                    idx_l = self.get_node_num(x-1, y, (h, w))
                    diff = np.sum((L[idx_c] - L[idx_l])**2)
                    edge_c = self.edge_mag[y, x] if x < w else 0.0
                    edge_l = self.edge_mag[y, x-1] if x-1 >= 0 else 0.0
                    edge_penalty = (edge_c + edge_l) / 2.0
                    weight = scale_factor * np.exp(-beta * diff) * (1.0 - 0.8 * edge_penalty)
                    weight = max(weight, 0.0)
                    g.add_edge(nodes[idx_c], nodes[idx_l], weight, weight)

                if y > 0:
                    idx_u = self.get_node_num(x, y-1, (h, w))
                    diff = np.sum((L[idx_c] - L[idx_u])**2)
                    edge_c = self.edge_mag[y, x] if y < h else 0.0
                    edge_u = self.edge_mag[y-1, x] if y-1 >= 0 else 0.0
                    edge_penalty = (edge_c + edge_u) / 2.0
                    weight = scale_factor * np.exp(-beta * diff) * (1.0 - 0.8 * edge_penalty)
                    weight = max(weight, 0.0)
                    g.add_edge(nodes[idx_c], nodes[idx_u], weight, weight)

                if x < w-1 and y < h-1:
                    idx_d_r = self.get_node_num(x+1, y+1, (h, w))
                    diff = np.sum((L[idx_c] - L[idx_d_r])**2)
                    weight_diag = scale_factor * np.exp(-beta * diff) * 0.5
                    weight_diag = max(weight_diag, 0.0)
                    g.add_edge(nodes[idx_c], nodes[idx_d_r], weight_diag, weight_diag)

                if x > 0 and y < h-1:
                    idx_d_l = self.get_node_num(x-1, y+1, (h, w))
                    diff = np.sum((L[idx_c] - L[idx_d_l])**2)
                    weight_diag = scale_factor * np.exp(-beta * diff) * 0.5
                    weight_diag = max(weight_diag, 0.0)
                    g.add_edge(nodes[idx_c], nodes[idx_d_l], weight_diag, weight_diag)

        print("   ✓ Grafo construido.")
        return g, nodes

    def create_segment(self, display=True):
        print("\n=== INICIANDO SEGMENTACIÓN ===")
        if self.image is None:
            raise ValueError("No se ha cargado ninguna imagen.")

        g, nodes = self._build_graph()

        print("Ejecutando maxflow/mincut...")
        flow = g.maxflow()
        print(f"   Flow computado: {flow}")

        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                idx = self.get_node_num(x, y, (h, w))
                seg = g.get_segment(nodes[idx])
                if seg == 0:
                    mask[y, x] = 0
                else:
                    mask[y, x] = 255

        print(f"   ✓ Segmentación inicial: {np.sum(mask == 255)} píxeles como objeto")

        mask_cleaned = self._clean_mask(mask)
        
        # Post-procesamiento con GrabCut si está habilitado
        if self.use_grabcut_refine:
            print("Aplicando refinamiento GrabCut...")
            mask_cleaned = self._refine_with_grabcut(mask_cleaned)

        h_orig, w_orig = self.image_bgr_original.shape[:2]
        if mask_cleaned.shape[:2] != (h_orig, w_orig):
            mask_full_res = cv2.resize(mask_cleaned, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        else:
            mask_full_res = mask_cleaned.copy()

        radius = self.guided_filter_radius
        if radius is None:
            base_dim = max(h_orig, w_orig)
            radius = max(5, int(base_dim / 100))
            if radius % 2 == 0:
                radius += 1

        mask_refined = self.refine_mask(mask_full_res, self.image_bgr_original, radius)

        self.mask = mask_refined

        print(f"   ✓ Segmentación completa: {np.sum(self.mask > 128)} píxeles finales")
        print("=== SEGMENTACIÓN TERMINADA ===\n")

    def _refine_with_grabcut(self, initial_mask):
        """Refinamiento usando GrabCut (iterativo)"""
        try:
            h, w = self.image.shape[:2]
            grabcut_mask = np.where(initial_mask == 255, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            
            # Marcar semillas como definitivas
            fg_indices_t = np.array(self.foreground_seeds).T
            bg_indices_t = np.array(self.background_seeds).T
            
            if fg_indices_t.size > 0:
                grabcut_mask[fg_indices_t[1], fg_indices_t[0]] = cv2.GC_FGD
            if bg_indices_t.size > 0:
                grabcut_mask[bg_indices_t[1], bg_indices_t[0]] = cv2.GC_BGD
            
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(self.image, grabcut_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            print(f"   ✓ GrabCut: {np.sum(refined_mask == 255)} píxeles refinados")
            
            return refined_mask
        except Exception as e:
            print(f"   ⚠ Error en GrabCut: {e}, usando máscara original")
            return initial_mask

    def get_seed_overlay_image(self):
        if self.image_bgr_original is None:
            return None

        img_copy = self.image_bgr_original.copy()

        for (x, y) in self.foreground_seeds:
            x_orig = int(x / self.downsample_scale)
            y_orig = int(y / self.downsample_scale)
            cv2.circle(img_copy, (x_orig, y_orig), 3, (0, 255, 0), -1)

        for (x, y) in self.background_seeds:
            x_orig = int(x / self.downsample_scale)
            y_orig = int(y / self.downsample_scale)
            cv2.circle(img_copy, (x_orig, y_orig), 3, (255, 0, 0), -1)

        return img_copy

    def get_segmented_image_for_display(self):
        if self.mask is None or self.image_bgr_original is None:
            return None

        h, w = self.image_bgr_original.shape[:2]
        if self.mask.shape[:2] != (h, w):
            mask_display = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_display = self.mask.copy()

        # Normalizar a 0-1
        mask_norm = mask_display.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm]*3, axis=-1)

        color_overlay = np.array(self.MASK_COLOR_RGB, dtype=np.float32) / 255.0

        img_float = self.image_bgr_original.astype(np.float32) / 255.0
        img_float_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)

        blended = (
            img_float_rgb * (1 - mask_3ch * self.MASK_ALPHA) +
            color_overlay * mask_3ch * self.MASK_ALPHA
        )
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        return blended_bgr

    def get_inverse_segmented_image_for_display(self):
        if self.mask is None or self.image_bgr_original is None:
            return None

        h, w = self.image_bgr_original.shape[:2]
        if self.mask.shape[:2] != (h, w):
            mask_display = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_display = self.mask.copy()

        mask_inv = 255 - mask_display

        mask_norm = mask_inv.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm]*3, axis=-1)

        color_overlay = np.array(self.MASK_COLOR_RGB, dtype=np.float32) / 255.0

        img_float = self.image_bgr_original.astype(np.float32) / 255.0
        img_float_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)

        blended = (
            img_float_rgb * (1 - mask_3ch * self.MASK_ALPHA) +
            color_overlay * mask_3ch * self.MASK_ALPHA
        )
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        return blended_bgr

    def get_histograms(self):
        if self.image_bgr_original is None:
            return None

        hist_dict = {}
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([self.image_bgr_original], [i], None, [256], [0, 256])
            hist_dict[color] = hist.flatten()

        return hist_dict
    
    def get_lab_histograms(self):
        """Devuelve histogramas del espacio Lab"""
        if self.image_bgr_original is None:
            return None
        
        img_lab_full = cv2.cvtColor(self.image_bgr_original, cv2.COLOR_BGR2LAB)
        
        hist_dict = {}
        for i, channel in enumerate(['L', 'a', 'b']):
            hist = cv2.calcHist([img_lab_full], [i], None, [256], [0, 256])
            hist_dict[channel] = hist.flatten()
        
        return hist_dict

    def save_sticker_image(self, filename, border_thickness, border_color_rgb,
                             text, text_color_rgb,
                             font_family, font_size, font_weight,
                             text_pos_factor,
                             is_preview=False, preview_size=(400, 400)):

        if self.mask is None or self.image_bgr_original is None:
            return None

        image_to_use = self.image_bgr_original
        h_orig, w_orig = image_to_use.shape[:2]

        if self.mask.shape[:2] != (h_orig, w_orig):
            mask_full_res = cv2.resize(self.mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        else:
            mask_full_res = self.mask.copy()

        if border_thickness > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness, border_thickness))
            border_mask = cv2.dilate(mask_full_res, kernel, iterations=2)

            border_bgr = np.zeros_like(image_to_use)
            border_bgr[:, :] = border_color_rgb[::-1]

            border_bgra = cv2.merge([border_bgr, border_mask])
        else:
            border_bgra = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)

        b, g, r = cv2.split(image_to_use)
        a = mask_full_res
        object_bgra = cv2.merge([b, g, r, a])

        obj_a = np.expand_dims(object_bgra[:, :, 3].astype(float) / 255.0, axis=2)
        bor_a = np.expand_dims(border_bgra[:, :, 3].astype(float) / 255.0, axis=2)

        obj_rgb = object_bgra[:, :, :3].astype(float)
        bor_rgb = border_bgra[:, :, :3].astype(float)

        out_a = bor_a * (1.0 - obj_a) + obj_a
        out_rgb = (bor_rgb * bor_a * (1.0 - obj_a)) + (obj_rgb * obj_a)

        sticker_bgra = np.concatenate([out_rgb, out_a * 255], axis=2).astype(np.uint8)

        if text and HAS_PIL:
            sticker_rgba_pil = Image.fromarray(cv2.cvtColor(sticker_bgra, cv2.COLOR_BGRA2RGBA))
            draw = ImageDraw.Draw(sticker_rgba_pil)

            try:
                pil_font = ImageFont.truetype(font_family, font_size)
            except IOError:
                try:
                    pil_font = ImageFont.truetype(f"{font_family}.ttf", font_size)
                except IOError:
                    print(f"Advertencia: No se pudo cargar la fuente '{font_family}'. Usando 'Arial.ttf'.")
                    try:
                        pil_font = ImageFont.truetype("Arial.ttf", font_size)
                    except IOError:
                        print("Advertencia: No se pudo cargar 'Arial.ttf'. Usando fuente por defecto.")
                        pil_font = ImageFont.load_default()

            if text:
                if hasattr(pil_font, 'getbbox'):
                    bbox = pil_font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    x_offset = bbox[0]
                    y_offset = bbox[1]
                else:
                    text_width, text_height = pil_font.getsize(text)
                    x_offset = 0
                    y_offset = 0
            else:
                text_width, text_height, x_offset, y_offset = 0, 0, 0, 0

            text_x = (w_orig - text_width) / 2
            total_y_space = h_orig - text_height
            text_y = total_y_space * text_pos_factor

            draw.text((text_x - x_offset, text_y - y_offset), text, fill=text_color_rgb, font=pil_font)

            sticker_bgra = cv2.cvtColor(np.array(sticker_rgba_pil), cv2.COLOR_RGBA2BGRA)

        if is_preview and preview_size[0] > 0 and preview_size[1] > 0:
            scale_w = preview_size[0] / w_orig
            scale_h = preview_size[1] / h_orig
            scale = min(scale_w, scale_h, 1.0)

            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)

            sticker_bgra = cv2.resize(sticker_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return sticker_bgra

    def _clean_mask(self, mask_uint8):
        if mask_uint8 is None:
            return None

        mask_binary = mask_uint8

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_binary,
            connectivity=8
        )

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                max_area = areas.max()
                area_threshold = max_area * 0.01

                cleaned = np.zeros_like(mask_binary)
                for lbl in range(1, num_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if area >= area_threshold:
                        cleaned[labels == lbl] = 255

                mask_binary = cleaned
                print(f"   ✓ Componentes: {num_labels}, área máx: {max_area}, "
                      f"umbral min: {area_threshold:.0f}")
        else:
            print("   ✓ Solo un componente, se mantiene completo")

        h, w = mask_binary.shape[:2]
        base = max(1, min(h, w))

        def _odd(x):
            x = max(3, int(x))
            return x if x % 2 == 1 else x + 1

        k_close = _odd(base / 200.0)
        k_open  = _odd(base / 300.0)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))

        mask_closed = cv2.morphologyEx(
            mask_binary,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=1
        )

        mask_opened = cv2.morphologyEx(
            mask_closed,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=1
        )

        return mask_opened

    @staticmethod
    def refine_mask(mask_uint8, guide_image_bgr, radius=None):
        if mask_uint8 is None or guide_image_bgr is None:
            return mask_uint8

        h, w = guide_image_bgr.shape[:2]
        base_dim = max(h, w)

        if radius is None:
            radius = max(5, int(base_dim / 100))
            if radius % 2 == 0:
                radius += 1

        eps = 1e-4

        print(f"   Aplicando Guided Filter (Radio: {radius}, Eps: {eps:.1e})...")

        if HAS_XIMGPROC:
            src_float = (mask_uint8.astype(np.float32) / 255.0)

            refined_mask_float = cv2.ximgproc.guidedFilter(
                guide=guide_image_bgr,
                src=src_float,
                radius=radius,
                eps=eps,
                dDepth=-1
            )

            refined_mask_float = cv2.ximgproc.guidedFilter(
                guide=guide_image_bgr,
                src=refined_mask_float,
                radius=radius // 2,
                eps=eps * 10,
                dDepth=-1
            )

            refined_mask_float = np.clip(refined_mask_float, 0.0, 1.0)
            refined_mask_float[refined_mask_float < 0.05] = 0.0

            return (refined_mask_float * 255).astype(np.uint8)
        else:
            soft_mask = cv2.GaussianBlur(mask_uint8, (radius, radius), 0)
            return soft_mask

    @staticmethod
    def get_node_num(x, y, array_shape):
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(nodenum, array_shape):
        return (nodenum % array_shape[1]), (nodenum // array_shape[1])