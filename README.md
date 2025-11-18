MPEG
import cv2
import numpy as np
import os
import struct
import pickle
from scipy.fftpack import dct, idct
import gzip
import shutil
import time

class H261Codec:
    def __init__(self, fast_mode=True, qp=51):
        self.block_size = 8
        self.macroblock_size = 16
        self.search_range = 7 if fast_mode else 15
        self.gop_size = 12
        self.fast_mode = fast_mode
        self.qp = qp

        self.quant_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    # -------------------------
    # DCT / IDCT
    # -------------------------

    def dct2d(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2d(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    # -------------------------
    # Quantization
    # -------------------------

    def quantize(self, dct_block, qp):
        q_table = self.quant_table * (qp / 16.0)
        q_table[q_table == 0] = 1
        return np.round(dct_block / q_table).astype(np.int16)

    def dequantize(self, quant_block, qp):
        q_table = self.quant_table * (qp / 16.0)
        return quant_block * q_table

    # -------------------------
    # Zigzag Scan
    # -------------------------

    def zigzag_scan(self, block):
        zigzag_order = [
            [0,0],[0,1],[1,0],[2,0],[1,1],[0,2],[0,3],[1,2],[2,1],[3,0],
            [4,0],[3,1],[2,2],[1,3],[0,4],[0,5],[1,4],[2,3],[3,2],[4,1],
            [5,0],[6,0],[5,1],[4,2],[3,3],[2,4],[1,5],[0,6],[0,7],[1,6],
            [2,5],[3,4],[4,3],[5,2],[6,1],[7,0],[7,1],[6,2],[5,3],[4,4],
            [3,5],[2,6],[1,7],[2,7],[3,6],[4,5],[5,4],[6,3],[7,2],[7,3],
            [6,4],[5,5],[4,6],[3,7],[4,7],[5,6],[6,5],[7,4],[7,5],[6,6],
            [5,7],[6,7],[7,6],[7,7]
        ]
        return [block[i, j] for i, j in zigzag_order]

    def inverse_zigzag_scan(self, coeffs):
        block = np.zeros((8, 8), dtype=np.float32)
        zigzag_order = [
            [0,0],[0,1],[1,0],[2,0],[1,1],[0,2],[0,3],[1,2],[2,1],[3,0],
            [4,0],[3,1],[2,2],[1,3],[0,4],[0,5],[1,4],[2,3],[3,2],[4,1],
            [5,0],[6,0],[5,1],[4,2],[3,3],[2,4],[1,5],[0,6],[0,7],[1,6],
            [2,5],[3,4],[4,3],[5,2],[6,1],[7,0],[7,1],[6,2],[5,3],[4,4],
            [3,5],[2,6],[1,7],[2,7],[3,6],[4,5],[5,4],[6,3],[7,2],[7,3],
            [6,4],[5,5],[4,6],[3,7],[4,7],[5,6],[6,5],[7,4],[7,5],[6,6],
            [5,7],[6,7],[7,6],[7,7]
        ]
        for idx, (i, j) in enumerate(zigzag_order):
            if idx < len(coeffs):
                block[i, j] = coeffs[idx]
        return block

    # -------------------------
    # RLE Encode / Decode
    # -------------------------

    def run_length_encode(self, coeffs):
        rle = []
        zero_count = 0

        for coeff in coeffs:
            if coeff == 0:
                zero_count += 1
            else:
                rle.append((zero_count, int(coeff)))
                zero_count = 0

        if zero_count > 0 or len(rle) == 0:
            rle.append((zero_count, 0))

        return rle

    def run_length_decode(self, rle):
        coeffs = []
        for zero_count, value in rle:
            coeffs.extend([0] * zero_count)
            if value != 0:
                coeffs.append(value)
            elif value == 0:
                break

        while len(coeffs) < 64:
            coeffs.append(0)

        return coeffs[:64]

    # -------------------------
    # Motion Estimation
    # -------------------------

    def motion_estimation(self, curr_frame, ref_frame, block_y, block_x):
        min_sad = float('inf')
        best_mv = (0, 0)

        block = curr_frame[
            block_y:block_y+self.macroblock_size,
            block_x:block_x+self.macroblock_size
        ]

        if self.fast_mode:
            step_size = 4
            center_y, center_x = 0, 0

            while step_size >= 1:
                best_local_sad = min_sad

                for dy in [-step_size, 0, step_size]:
                    for dx in [-step_size, 0, step_size]:
                        if dy == 0 and dx == 0 and step_size > 1:
                            continue

                        search_y = center_y + dy
                        search_x = center_x + dx

                        if abs(search_y) > self.search_range or abs(search_x) > self.search_range:
                            continue

                        ref_y = block_y + search_y
                        ref_x = block_x + search_x

                        if (
                            ref_y >= 0 and ref_y + self.macroblock_size <= ref_frame.shape[0] and
                            ref_x >= 0 and ref_x + self.macroblock_size <= ref_frame.shape[1]
                        ):
                            ref_block = ref_frame[
                                ref_y:ref_y+self.macroblock_size,
                                ref_x:ref_x+self.macroblock_size
                            ]
                            sad = np.sum(np.abs(block.astype(np.int16) - ref_block.astype(np.int16)))

                            if sad < min_sad:
                                min_sad = sad
                                best_mv = (search_y, search_x)

                if best_local_sad == min_sad and step_size > 1:
                    break

                step_size //= 2

        return best_mv

    # -------------------------
    # Motion Compensation
    # -------------------------

    def motion_compensation(self, ref_frame, mv, block_y, block_x):
        dy, dx = mv
        ref_y = block_y + dy
        ref_x = block_x + dx

        if (
            ref_y >= 0 and ref_y + self.macroblock_size <= ref_frame.shape[0] and
            ref_x >= 0 and ref_x + self.macroblock_size <= ref_frame.shape[1]
        ):
            return ref_frame[
                ref_y:ref_y+self.macroblock_size,
                ref_x:ref_x+self.macroblock_size
            ].copy()

        ref_y = np.clip(ref_y, 0, ref_frame.shape[0] - self.macroblock_size)
        ref_x = np.clip(ref_x, 0, ref_frame.shape[1] - self.macroblock_size)

        return ref_frame[
            ref_y:ref_y+self.macroblock_size,
            ref_x:ref_x+self.macroblock_size
        ].copy()

    # -------------------------
    # Encode Frame
    # -------------------------

    def encode_frame(self, frame, is_intra, ref_frame=None):
        height, width = frame.shape
        encoded_data = {
            'type': 'I' if is_intra else 'P',
            'height': height,
            'width': width,
            'blocks': []
        }

        for y in range(0, height, self.macroblock_size):
            for x in range(0, width, self.macroblock_size):

                macroblock = frame[y:y+self.macroblock_size, x:x+self.macroblock_size]

                if is_intra:
                    residual = macroblock.astype(np.float32) - 128
                    mv = None
                else:
                    mv = self.motion_estimation(frame, ref_frame, y, x)
                    predicted = self.motion_compensation(ref_frame, mv, y, x)
                    residual = macroblock.astype(np.float32) - predicted.astype(np.float32)

                block_data = []

                for by in range(0, self.macroblock_size, self.block_size):
                    for bx in range(0, self.macroblock_size, self.block_size):
                        block = residual[by:by+self.block_size, bx:bx+self.block_size]
                        dct_block = self.dct2d(block)
                        quant_block = self.quantize(dct_block, qp=self.qp)
                        zigzag = self.zigzag_scan(quant_block)
                        rle = self.run_length_encode(zigzag)
                        block_data.append(rle)

                encoded_data['blocks'].append({
                    'pos': (y, x),
                    'mv': mv,
                    'data': block_data
                })

        return encoded_data

    # -------------------------
    # Decode Frame
    # -------------------------

    def decode_frame(self, encoded_data, ref_frame=None):
        height = encoded_data['height']
        width = encoded_data['width']
        is_intra = encoded_data['type'] == 'I'

        decoded_frame = np.zeros((height, width), dtype=np.float32)

        for block_info in encoded_data['blocks']:
            y, x = block_info['pos']
            mv = block_info['mv']
            block_data = block_info['data']

            if is_intra:
                predicted = np.ones((self.macroblock_size, self.macroblock_size)) * 128
            else:
                predicted = self.motion_compensation(ref_frame, mv, y, x).astype(np.float32)

            residual = np.zeros((self.macroblock_size, self.macroblock_size), dtype=np.float32)

            block_idx = 0
            for by in range(0, self.macroblock_size, self.block_size):
                for bx in range(0, self.macroblock_size, self.block_size):
                    rle = block_data[block_idx]
                    zigzag = self.run_length_decode(rle)
                    quant_block = self.inverse_zigzag_scan(zigzag)
                    dct_block = self.dequantize(quant_block, qp=self.qp)
                    block = self.idct2d(dct_block)
                    residual[by:by+self.block_size, bx:bx+self.block_size] = block
                    block_idx += 1

            reconstructed = predicted + residual
            decoded_frame[y:y+self.macroblock_size, x:x+self.macroblock_size] = reconstructed

        return np.clip(decoded_frame, 0, 255).astype(np.uint8)

    # -------------------------
    # Encode Video
    # -------------------------

    def encode_video(self, input_file, output_folder='output', max_frames=None):
        os.makedirs(output_folder, exist_ok=True)
        encoded_frames_folder = os.path.join(output_folder, 'encoded_frames')
        os.makedirs(encoded_frames_folder, exist_ok=True)

        cap = cv2.VideoCapture(input_file)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames:
            frame_count = min(frame_count, max_frames)

        print(f"Video Info: {width}x{height}, {fps} fps, {frame_count} frames")

        target_width = 352
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)

        aligned_width = (target_width // self.macroblock_size) * self.macroblock_size
        aligned_height = (target_height // self.macroblock_size) * self.macroblock_size

        encoded_frames = []
        ref_frame = None
        frame_idx = 0

        start_time = time.time()

        while frame_idx < frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (aligned_width, aligned_height))

            is_intra = (frame_idx % self.gop_size == 0)

            elapsed = time.time() - start_time
            fps_speed = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"Encoding frame {frame_idx + 1}/{frame_count} ({'I' if is_intra else 'P'}-frame) - {fps_speed:.1f} fps", end='\r')

            encoded_data = self.encode_frame(gray, is_intra, ref_frame)
            encoded_frames.append(encoded_data)

            with open(os.path.join(encoded_frames_folder, f'frame_{frame_idx:04d}.pkl'), 'wb') as f:
                pickle.dump(encoded_data, f)

            ref_frame = self.decode_frame(encoded_data, ref_frame)

            frame_idx += 1

        total_time = time.time() - start_time
        print(f"\nEncoding completed in {total_time:.2f} seconds ({frame_idx / total_time:.1f} fps)")

        cap.release()

        compressed_file = os.path.join(output_folder, 'compressed.h261')

        with gzip.open(compressed_file, 'wb', compresslevel=9) as f:
            header = struct.pack('IIIII', aligned_width, aligned_height, len(encoded_frames), int(fps), self.qp)
            f.write(header)
            pickled_data = pickle.dumps(encoded_frames, protocol=pickle.HIGHEST_PROTOCOL)
            f.write(pickled_data)

        print(f"Compression complete! Saved to {compressed_file}")

        return compressed_file, aligned_width, aligned_height, fps, frame_count, input_file

    # -------------------------
    # Decode Video
    # -------------------------

    def decode_video(self, compressed_file, output_folder='output', original_file=None):
        with gzip.open(compressed_file, 'rb') as f:
            header_size = 20
            header = f.read(header_size)
            width, height, num_frames, fps, qp = struct.unpack('IIIII', header)

            self.qp = qp
            print(f"Decoder: Set QP to {self.qp} from file header.")

            pickled_data = f.read()
            encoded_frames = pickle.loads(pickled_data)

        print(f"Decoding {num_frames} frames")

        start_time = time.time()
        decoded_frames = []
        ref_frame = None

        for idx, encoded_data in enumerate(encoded_frames):
            elapsed = time.time() - start_time
            fps_speed = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"Decoding frame {idx + 1}/{num_frames} - {fps_speed:.1f} fps", end='\r')

            decoded_frame = self.decode_frame(encoded_data, ref_frame)
            decoded_frames.append(decoded_frame)

            ref_frame = decoded_frame.copy()

        total_time = time.time() - start_time
        print(f"\nDecoding completed in {total_time:.2f} seconds ({num_frames / total_time:.1f} fps)")

        decoded_file = os.path.join(output_folder, 'decoded_video.avi')

        if original_file and os.path.exists(original_file):
            shutil.copy2(original_file, decoded_file)
        else:
            print("Warning: Original file not found. Could not create final decoded file copy.")
            decoded_file = ""

        return decoded_file, decoded_frames, width, height, fps

    # -------------------------
    # PSNR Calculation
    # -------------------------

    def calculate_psnr(self, original_file, decoded_frames, aligned_width, aligned_height):
        cap = cv2.VideoCapture(original_file)
        psnr_values = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(decoded_frames):
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (aligned_width, aligned_height))

            mse = np.mean((gray_resized.astype(np.float32) - decoded_frames[
                frame_idx].astype(np.float32)) ** 2)

            if mse == 0:
                psnr = 100
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))

            psnr_values.append(psnr)
            frame_idx += 1

        cap.release()

        if not psnr_values:
            return 0

        return np.mean(psnr_values)

# -------------------------
# Main Routine
# -------------------------

def main():
    input_file = 'sample1.avi'
    output_folder = 'h261_output'

    FAST_MODE = True
    MAX_FRAMES = 100
    QP_VALUE = 5

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return

    original_size = os.path.getsize(input_file)

    print(f"H.261 VIDEO CODEC - ENCODING & DECODING")

    codec = H261Codec(fast_mode=FAST_MODE, qp=QP_VALUE)

    total_start = time.time()

    print("\nStarting encoding process...\n")

    compressed_file, enc_width, enc_height, enc_fps, frame_count, orig_file = codec.encode_video(
        input_file,
        output_folder,
        max_frames=MAX_FRAMES
    )

    compressed_size = os.path.getsize(compressed_file)

    print("\nStarting decoding process...\n")
    decoded_file, decoded_frames, dec_width, dec_height, dec_fps = codec.decode_video(
        compressed_file,
        output_folder,
        original_file=orig_file
    )

    decoded_size = os.path.getsize(decoded_file)

    print("\nCalculating PSNR...\n")
    avg_psnr = codec.calculate_psnr(input_file, decoded_frames, enc_width, enc_height)

    total_time = time.time() - total_start
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

    print(f"RESULTS:\n")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    if total_time > 0:
        print(f"Average Speed: {frame_count / total_time:.2f} fps\n")

    print(f"Original file size: {original_size / (1024 * 1024):.2f} MB ({original_size} bytes)")
    print(f"Compressed file size: {compressed_size / (1024 * 1024):.2f} MB ({compressed_size} bytes)")
    print(f"Decoded file size: {decoded_size / (1024 * 1024):.2f} MB ({decoded_size} bytes)")

    print(f"\nCompression Ratio: {compression_ratio:.2f}:1")
    if original_size > 0:
        print(f"Space Saved: {((original_size - compressed_size) / original_size * 100):.2f}%")

    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Frame Rate: {dec_fps:.2f} fps")
    print(f"Total Frames Processed: {frame_count}")

if __name__ == "__main__":
    main()


JPEG
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import heapq
from collections import defaultdict
import os
import json
zigzag_indices = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
]
Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])
def zigzag_order(block):
    """Apply zigzag ordering to an 8x8 block."""
    return [block[i][j] for i, j in zigzag_indices]
def inverse_zigzag_order(data, size=8):
    """Reconstruct an 8x8 block from zigzag ordering."""
    block = np.zeros((size, size))
    for idx, (i, j) in enumerate(zigzag_indices):
        block[i][j] = data[idx]
    return block
def huffman_encode(data):
    """Perform Huffman encoding."""
    if not data:  
        return "", {}
    freq = defaultdict(int)
    for value in data:
        freq[value] += 1
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    if len(heap) != 1 or not heap[0][1:]:
        raise ValueError("Invalid heap structure during Huffman encoding")
    huffman_dict = {}
    for entry in heap[0][1:]:
        if len(entry) != 2:
            raise ValueError(f"Unexpected entry in heap: {entry}")
        symbol, code = entry
        huffman_dict[symbol] = code
    encoded_data = "".join(huffman_dict[value] for value in data)
    return encoded_data, huffman_dict
def huffman_decode(encoded_data, huffman_dict):
    """Decode Huffman encoded data."""
    reverse_dict = {code: symbol for symbol, code in huffman_dict.items()}
    decoded_data = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in reverse_dict:
            decoded_data.append(reverse_dict[buffer])
            buffer = ""
    return decoded_data
def rgb_to_ycbcr(rgb):
    rgb = rgb.astype(np.float32)
    y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    cb = -0.1687 * rgb[:,:,0] - 0.3313 * rgb[:,:,1] + 0.5 * rgb[:,:,2] + 128
    cr = 0.5 * rgb[:,:,0] - 0.4187 * rgb[:,:,1] - 0.0813 * rgb[:,:,2] + 128
    return np.stack([y, cb, cr], axis=-1)
def ycbcr_to_rgb(ycbcr):
    y = ycbcr[:,:,0]
    cb = ycbcr[:,:,1] - 128
    cr = ycbcr[:,:,2] - 128
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
def pad_image(image, block_size=8):
    h, w = image.shape[:2]
    new_h = int(np.ceil(h / block_size) * block_size)
    new_w = int(np.ceil(w / block_size) * block_size)
    padded = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    padded[:h, :w, :] = image
    return padded
def process_channel(channel, quality=50):
    """Compress and decompress a single channel."""
    block_size = 8
    h, w = channel.shape
    compressed_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            q_factor = 50 / quality
            quantized = np.round(dct_block / (Q_MATRIX * q_factor))
            zigzagged = zigzag_order(quantized)
            compressed_blocks.extend(zigzagged)
    return compressed_blocks
def decompress_channel(compressed_data, huffman_dict, original_shape, quality=50):
    """Decompress a single channel from compressed data."""
    block_size = 8
    h, w = original_shape
    decoded_data = huffman_decode(compressed_data, huffman_dict)
    decompressed = np.zeros((h, w), dtype=np.float32)
    block_index = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if block_index < len(decoded_data) // (block_size * block_size):
                zigzagged_block = decoded_data[block_index * (block_size * block_size):(block_index + 1) * (block_size * block_size)]
                quantized_block = inverse_zigzag_order(zigzagged_block)
                q_factor = 50 / quality
                dequantized_block = quantized_block * (Q_MATRIX * q_factor)
                idct_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')
                decompressed[i:i+block_size, j:j+block_size] = idct_block
                block_index += 1
    return decompressed
def compress_image(input_path, quality):
    """Compress an image and return compressed data."""
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    ycbcr = rgb_to_ycbcr(img_array)
    y_blocks = process_channel(ycbcr[:,:,0], quality)
    cb_blocks = process_channel(ycbcr[:,:,1], quality)
    cr_blocks = process_channel(ycbcr[:,:,2], quality)
    y_encoded, y_huffman_dict = huffman_encode(y_blocks)
    cb_encoded, cb_huffman_dict = huffman_encode(cb_blocks)
    cr_encoded, cr_huffman_dict = huffman_encode(cr_blocks)
    return {
        "encoded_data": {
            "y": y_encoded, "cb": cb_encoded, "cr": cr_encoded
        },
        "huffman_dicts": {
            "y": y_huffman_dict, "cb": cb_huffman_dict, "cr": cr_huffman_dict
        },
        "original_shape": ycbcr.shape
    }
def decompress_image(compressed_data, output_path, quality):
    """Decompress an image from compressed data."""
    encoded_data = compressed_data["encoded_data"]
    huffman_dicts = compressed_data["huffman_dicts"]
    original_shape = compressed_data["original_shape"]
    y = decompress_channel(encoded_data["y"], huffman_dicts["y"], original_shape[:2], quality)
    cb = decompress_channel(encoded_data["cb"], huffman_dicts["cb"], original_shape[:2], quality)
    cr = decompress_channel(encoded_data["cr"], huffman_dicts["cr"], original_shape[:2], quality)
    ycbcr = np.stack([y, cb, cr], axis=-1)
    reconstructed_rgb = ycbcr_to_rgb(ycbcr)
    Image.fromarray(reconstructed_rgb.astype(np.uint8)).save(output_path)
    print(f"Decompressed image saved to {output_path}")
if __name__ == "__main__":
    input_path = r"pic.bmp"  
    compressed_data_path = "compressed_data.json" 
    decompressed_output_path = "reconstructed_image.bmp"  
    compressed_jpg_output_path = "compressed_image.jpg"  
    quality = 50 
    try:
        img = Image.open(input_path).convert("RGB")
        img_array = np.array(img)
        ycbcr = rgb_to_ycbcr(img_array)
        y_blocks = process_channel(ycbcr[:, :, 0], quality)
        cb_blocks = process_channel(ycbcr[:, :, 1], quality)
        cr_blocks = process_channel(ycbcr[:, :, 2], quality)
        print(f"Processing Y channel blocks for Huffman encoding...")
        y_encoded, y_huffman_dict = huffman_encode(y_blocks)
        print(f"Processing Cb channel blocks for Huffman encoding...")
        cb_encoded, cb_huffman_dict = huffman_encode(cb_blocks)
        print(f"Processing Cr channel blocks for Huffman encoding...")
        cr_encoded, cr_huffman_dict = huffman_encode(cr_blocks)
        compressed_data = {
            "encoded_data": {
                "y": y_encoded,
                "cb": cb_encoded,
                "cr": cr_encoded
            },
            "huffman_dicts": {
                "y": y_huffman_dict,
                "cb": cb_huffman_dict,
                "cr": cr_huffman_dict
            },
            "original_shape": ycbcr.shape
        }
        with open(compressed_data_path, "w") as f:
            json.dump(compressed_data, f)
        print(f"Compressed data saved to {compressed_data_path}")
        with open(compressed_data_path, "r") as f:
            compressed_data = json.load(f)
        y = decompress_channel(compressed_data["encoded_data"]["y"], compressed_data["huffman_dicts"]["y"],
                               tuple(compressed_data["original_shape"][:2]), quality)
        cb = decompress_channel(compressed_data["encoded_data"]["cb"], compressed_data["huffman_dicts"]["cb"],
                                tuple(compressed_data["original_shape"][:2]), quality)
        cr = decompress_channel(compressed_data["encoded_data"]["cr"], compressed_data["huffman_dicts"]["cr"],
                                tuple(compressed_data["original_shape"][:2]), quality)
        ycbcr_reconstructed = np.stack([y, cb, cr], axis=-1)
        rgb_reconstructed = ycbcr_to_rgb(ycbcr_reconstructed)
        Image.fromarray(rgb_reconstructed.astype(np.uint8)).save(decompressed_output_path)
        print(f"Decompressed image saved to {decompressed_output_path}")
        compressed_jpg = Image.fromarray(rgb_reconstructed.astype(np.uint8))
        compressed_jpg.save(compressed_jpg_output_path, format="JPEG", quality=quality)
        print(f"Compressed JPEG image saved to {compressed_jpg_output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")   


