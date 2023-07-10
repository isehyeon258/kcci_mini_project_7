import whisper
from pathlib import Path
from openvino.tools import mo
from openvino.runtime import serialize
from openvino.runtime import Core, Tensor
from whisper.decoding import DecodingTask, Inference, DecodingOptions, DecodingResult
import librosa
import re
import pyaudio
import wave
import sys
import torch
'''
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

if sys.platform == 'darwin':
    CHANNELS = 1

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
'''
# ===================================================================================

REPO_DIR = Path("whisper")
model = whisper.load_model("base")
model.to("cpu")
model.eval()
pass

mel = torch.zeros((1, 80, 3000))
audio_features = model.encoder(mel)
torch.onnx.export(
    model.encoder, 
    mel, 
    "whisper_encoder.onnx",
    input_names=["mel"], 
    output_names=["output_features"]
)
encoder_model = mo.convert_model("whisper_encoder.onnx", compress_to_fp16=True)
serialize(encoder_model, xml_path="whisper_encoder.xml")

from typing import Optional, Union, List, Dict
from functools import partial

positional_embeddings_size = model.decoder.positional_embedding.shape[0]


def save_to_cache(cache: Dict[str, torch.Tensor], module: str, output: torch.Tensor):
    """
    Saving cached attention hidden states for previous tokens.
    Parameters:
      cache: dictionary with cache.
      module: current attention module name.
      output: predicted hidden state.
    Returns:
      output: cached attention hidden state for specified attention module.
    """
    if module not in cache or output.shape[1] > positional_embeddings_size:
        # save as-is, for the first token or cross attention
        cache[module] = output
    else:
        cache[module] = torch.cat([cache[module], output], dim=1).detach()
    return cache[module]


def attention_forward(
        attention_module,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        idx: int = 0
):
    """
    Override for forward method of decoder attention module with storing cache values explicitly.
    Parameters:
      attention_module: current attention module
      x: input token ids.
      xa: input audio features (Optional).
      mask: mask for applying attention (Optional).
      kv_cache: dictionary with cached key values for attention modules.
      idx: idx for search in kv_cache.
    Returns:
      attention module output tensor
      updated kv_cache
    """
    q = attention_module.query(x)

    if kv_cache is None or xa is None:
        # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
        # otherwise, perform key/value projections for self- or cross-attention as usual.
        k = attention_module.key(x if xa is None else xa)
        v = attention_module.value(x if xa is None else xa)
        if kv_cache is not None:
            k = save_to_cache(kv_cache, f'k_{idx}', k)
            v = save_to_cache(kv_cache, f'v_{idx}', v)
    else:
        # for cross-attention, calculate keys and values once and reuse in subsequent calls.
        k = kv_cache.get(f'k_{idx}', save_to_cache(
            kv_cache, f'k_{idx}', attention_module.key(xa)))
        v = kv_cache.get(f'v_{idx}', save_to_cache(
            kv_cache, f'v_{idx}', attention_module.value(xa)))

    wv, qk = attention_module.qkv_attention(q, k, v, mask)
    return attention_module.out(wv), kv_cache


def block_forward(
    residual_block,
    x: torch.Tensor,
    xa: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[dict] = None,
    idx: int = 0
):
    """
    Override for residual block forward method for providing kv_cache to attention module.
      Parameters:
        residual_block: current residual block.
        x: input token_ids.
        xa: input audio features (Optional).
        mask: attention mask (Optional).
        kv_cache: cache for storing attention key values.
        idx: index of current residual block for search in kv_cache.
      Returns:
        x: residual block output
        kv_cache: updated kv_cache

    """
    x0, kv_cache = residual_block.attn(residual_block.attn_ln(
        x), mask=mask, kv_cache=kv_cache, idx=f'{idx}a')
    x = x + x0
    if residual_block.cross_attn:
        x1, kv_cache = residual_block.cross_attn(
            residual_block.cross_attn_ln(x), xa, kv_cache=kv_cache, idx=f'{idx}c')
        x = x + x1
    x = x + residual_block.mlp(residual_block.mlp_ln(x))
    return x, kv_cache


# update forward functions
for idx, block in enumerate(model.decoder.blocks):
    block.forward = partial(block_forward, block, idx=idx)
    block.attn.forward = partial(attention_forward, block.attn)
    if block.cross_attn:
        block.cross_attn.forward = partial(attention_forward, block.cross_attn)


def decoder_forward(decoder, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
    """
    Override for decoder forward method.
    Parameters:
      x: torch.LongTensor, shape = (batch_size, <= n_ctx) the text tokens
      xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
           the encoded audio features to be attended on
      kv_cache: Dict[str, torch.Tensor], attention modules hidden states cache from previous steps 
    """
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    x = decoder.token_embedding(
        x) + decoder.positional_embedding[offset: offset + x.shape[-1]]
    x = x.to(xa.dtype)

    for block in decoder.blocks:
        x, kv_cache = block(x, xa, mask=decoder.mask, kv_cache=kv_cache)

    x = decoder.ln(x)
    logits = (
        x @ torch.transpose(decoder.token_embedding.weight.to(x.dtype), 1, 0)).float()

    return logits, kv_cache


# override decoder forward
model.decoder.forward = partial(decoder_forward, model.decoder)

tokens = torch.ones((5, 3), dtype=torch.int64)

logits, kv_cache = model.decoder(tokens, audio_features, kv_cache={})
kv_cache = {k: v for k, v in kv_cache.items()}
tokens = torch.ones((5, 1), dtype=torch.int64)

outputs = [f"out_{k}" for k in kv_cache.keys()]
inputs = [f"in_{k}" for k in kv_cache.keys()]
dynamic_axes = {
    "tokens": {0: "beam_size", 1: "seq_len"},
    "audio_features": {0: "beam_size"},
    "logits": {0: "beam_size", 1: "seq_len"}}
dynamic_outs = {o: {0: "beam_size", 1: "prev_seq_len"} for o in outputs}
dynamic_inp = {i: {0: "beam_size", 1: "prev_seq_len"} for i in inputs}
dynamic_axes.update(dynamic_outs)
dynamic_axes.update(dynamic_inp)
torch.onnx.export(
    model.decoder, {'x': tokens, 'xa': audio_features, 'kv_cache': kv_cache},
    'whisper_decoder.onnx',
    input_names=["tokens", "audio_features"] + inputs,
    output_names=["logits"] + outputs,
    dynamic_axes=dynamic_axes
)

input_shapes = "tokens[1..5 1..224],audio_features[1..5 1500 512]"
for k, v in kv_cache.items():
    if k.endswith('a'):
        input_shapes += f",in_{k}[1..5 0..224 512]"
decoder_model = mo.convert_model(
    input_model="whisper_decoder.onnx",
    compress_to_fp16=True,
    input=input_shapes)
serialize(decoder_model, "whisper_decoder.xml")

class OpenVINOAudioEncoder(torch.nn.Module):
    """
    Helper for inference Whisper encoder model with OpenVINO
    """

    def __init__(self, core, model_path, device='CPU'):
        super().__init__()
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)
        self.output_blob = self.compiled_model.output(0)

    def forward(self, mel: torch.Tensor):
        """
        Inference OpenVINO whisper encoder model.

        Parameters:
          mel: input audio fragment mel spectrogram.
        Returns:
          audio_features: torch tensor with encoded audio features.
        """
        return torch.from_numpy(self.compiled_model(mel)[self.output_blob])


class OpenVINOTextDecoder(torch.nn.Module):
    """
    Helper for inference OpenVINO decoder model
    """

    def __init__(self, core: Core, model_path: Path, device: str = 'CPU'):
        super().__init__()
        self._core = core
        self.model = core.read_model(model_path)
        self._input_names = [inp.any_name for inp in self.model.inputs]
        self.compiled_model = core.compile_model(self.model, device)
        self.device = device

    def init_past_inputs(self, feed_dict):
        """
        Initialize cache input for first step.

        Parameters:
          feed_dict: Dictonary with inputs for inference
        Returns:
          feed_dict: updated feed_dict
        """
        beam_size = feed_dict['tokens'].shape[0]
        audio_len = feed_dict['audio_features'].shape[2]
        previous_seq_len = 0
        for name in self._input_names:
            if name in ['tokens', 'audio_features']:
                continue
            feed_dict[name] = Tensor(np.zeros(
                (beam_size, previous_seq_len, audio_len), dtype=np.float32))
        return feed_dict

    def preprocess_kv_cache_inputs(self, feed_dict, kv_cache):
        """
        Transform kv_cache to inputs

        Parameters:
          feed_dict: dictionary with inputs for inference
          kv_cache: dictionary with cached attention hidden states from previous step
        Returns:
          feed_dict: updated feed dictionary with additional inputs
        """
        if not kv_cache:
            return self.init_past_inputs(feed_dict)
        for k, v in kv_cache.items():
            new_k = f'in_{k}'
            if new_k in self._input_names:
                feed_dict[new_k] = Tensor(v.numpy())
        return feed_dict

    def postprocess_outputs(self, outputs):
        """
        Transform model output to format expected by the pipeline

        Parameters:
          outputs: outputs: raw inference results.
        Returns:
          logits: decoder predicted token logits
          kv_cache: cached attention hidden states
        """
        logits = None
        kv_cache = {}
        for output_t, out in outputs.items():
            if 'logits' in output_t.get_names():
                logits = torch.from_numpy(out)
            else:
                tensor_name = output_t.any_name
                kv_cache[tensor_name.replace(
                    'out_', '')] = torch.from_numpy(out)
        return logits, kv_cache

    def forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        """
        Inference decoder model.

        Parameters:
          x: torch.LongTensor, shape = (batch_size, <= n_ctx) the text tokens
          xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
             the encoded audio features to be attended on
          kv_cache: Dict[str, torch.Tensor], attention modules hidden states cache from previous steps
        Returns:
          logits: decoder predicted logits
          kv_cache: updated kv_cache with current step hidden states
        """
        feed_dict = {'tokens': Tensor(x.numpy()), 'audio_features': Tensor(xa.numpy())}
        feed_dict = (self.preprocess_kv_cache_inputs(feed_dict, kv_cache))
        res = self.compiled_model(feed_dict)
        return self.postprocess_outputs(res)
    
class OpenVINOInference(Inference):
    """
    Wrapper for inference interface
    """

    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        getting logits for given tokens sequence and audio features and save kv_cache

        Parameters:
          tokens: input tokens
          audio_features: input audio features
        Returns:
          logits: predicted by decoder logits
        """
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]
        logits, self.kv_cache = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache)
        return logits

    def cleanup_caching(self):
        """
        Reset kv_cache to initial state
        """
        self.kv_cache = {}

    def rearrange_kv_cache(self, source_indices):
        """
        Update hidden states cache for selected sequences
        Parameters:
          source_indicies: sequences indicies
        Returns:
          None
        """
        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices]


class OpenVINODecodingTask(DecodingTask):
    """
    Class for decoding using OpenVINO
    """

    def __init__(self, model: "Whisper", options: DecodingOptions):
        super().__init__(model, options)
        self.inference = OpenVINOInference(model, len(self.initial_tokens))


@torch.no_grad()
def decode(model: "Whisper", mel: torch.Tensor, options: DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    result = OpenVINODecodingTask(model, options).run(mel)

    if single:
        result = result[0]

    return result

del model.decoder
del model.encoder

from collections import namedtuple

Parameter = namedtuple('Parameter', ['device'])

core = Core()

model.encoder = OpenVINOAudioEncoder(core, 'whisper_encoder.xml')
model.decoder = OpenVINOTextDecoder(core, 'whisper_decoder.xml')
model.decode = partial(decode, model)


def parameters():
    return iter([Parameter(torch.device('cpu'))])


model.parameters = parameters


def logits(model, tokens: torch.Tensor, audio_features: torch.Tensor):
    """
    Override for logits extraction method
    Parameters:
      toekns: input tokens
      audio_features: input audio features
    Returns:
      logits: decoder predicted logits
    """
    return model.decoder(tokens, audio_features, None)[0]


model.logits = partial(logits, model)

import io
from pathlib import Path
import numpy as np
from scipy.io import wavfile


def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)


def audio_to_float(audio):
    """
    convert audio signal to floating point format
    """
    return audio.astype(np.float32) / np.iinfo(audio.dtype).max


def get_audio(audio_file):
    """
    Extract audio signal from a given video file, then convert it to float, 
    then mono-channel format and resample it to the expected sample rate

    Parameters:
        video_file: path to input video file
    Returns:
      resampled_audio: mono-channel float audio signal with 16000 Hz sample rate 
                       extracted from video  
    """
    #input_video = VideoFileClip(str(video_file))
    #input_video.audio.write_audiofile(video_file.stem + '.wav', verbose=False, logger=None)
    #input_audio_file = video_file.stem + '.wav'
    sample_rate, audio = wavfile.read(
        io.BytesIO(open(audio_file, 'rb').read()))
    audio = audio_to_float(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    resampled_audio = resample(audio, sample_rate, 16000)
    return resampled_audio

audio = get_audio("output.wav")
#audio = get_audio("downloaded_video.wav")
import ipywidgets as widgets

task = widgets.Select(
    options=["transcribe", "translate"],
    value="translate",
    description="Select task:",
    disabled=False
)
task

transcription = model.transcribe(audio, beam_size=5, best_of=5, task=task.value)


def prepare_srt(transcription):
    """
    Format transcription into srt file format
    """
    segment_lines = []
    for segment in transcription["segments"]:
        #segment_lines.append(str(segment["id"] + 1))
        #time_start = format_timestamp(segment["start"])
        #time_end = format_timestamp(segment["end"])
        #time_str = f"{time_start} --> {time_end}"

        #segment_lines.append(segment["text"])
         segment_lines.append(re.sub(r"[^a-zA-Z],", "", segment["text"]))
    return segment_lines

srt_lines = prepare_srt(transcription)

flattened_list = []

for item in srt_lines:
    flattened_list.extend(item.split(','))

modified_list = ["".join(filter(str.isalpha, item.strip().lower())) for item in flattened_list]

   

import sys
import os
import cv2
import numpy as np
import paddle
import math
import time
import collections
from PIL import Image
from pathlib import Path
import tarfile
import urllib.request

from openvino.runtime import Core
from IPython import display
import copy

sys.path.append("../utils")
import notebook_utils as utils
import pre_post_processing as processing


# Directory where model will be downloaded

#det_model_url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar"
#det_model_file_path = Path("model/ch_ppocr_mobile_v2.0_det_infer/inference.pdmodel")
model_name_det = "model/ch_ppocr_mobile_v2.0_det_infer/inference.pdmodel"

#run_model_download(det_model_url, det_model_file_path)

# initialize inference engine for text detection
core = Core()
det_model = core.read_model(model=model_name_det)
det_compiled_model = core.compile_model(model=det_model, device_name="CPU")

# get input and output nodes for text detection
det_input_layer = det_compiled_model.input(0)
det_output_layer = det_compiled_model.output(0)

#rec_model_url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar"
#rec_model_file_path = Path("model/ch_ppocr_mobile_v2.0_rec_infer/inference.pdmodel")

model_name_rec = "model/ch_ppocr_mobile_v2.0_rec_infer/inference.pdmodel"

#run_model_download(rec_model_url, rec_model_file_path)

# read the model and corresponding weights from file
rec_model = core.read_model(model=model_name_rec)

# assign dynamic shapes to every input layer on the last dimension
for input_layer in rec_model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = -1
    rec_model.reshape({input_layer: input_shape})

rec_compiled_model = core.compile_model(model=rec_model, device_name="CPU")

# get input and output nodes
rec_input_layer = rec_compiled_model.input(0)
rec_output_layer = rec_compiled_model.output(0)

# Preprocess for text detection
def image_preprocess(input_image, size):
    """
    Preprocess input image for text detection

    Parameters:
        input_image: input image
        size: value for the image to be resized for text detection model
    """
    img = cv2.resize(input_image, (size, size))
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    # NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)

# Preprocess for text recognition
def resize_norm_img(img, max_wh_ratio):
    """
    Resize input image for text recognition

    Parameters:
        img: bounding box image from text detection
        max_wh_ratio: value for the resizing for text recognition model
    """
    rec_image_shape = [3, 32, 320]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    character_type = "ch"
    if character_type == "ch":
        imgW = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def prep_for_rec(dt_boxes, frame):
    """
    Preprocessing of the detected bounding boxes for text recognition

    Parameters:
        dt_boxes: detected bounding boxes from text detection
        frame: original input frame
    """
    ori_im = frame.copy()
    img_crop_list = []
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = processing.get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)

    img_num = len(img_crop_list)
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_crop_list:
        width_list.append(img.shape[1] / float(img.shape[0]))

    # Sorting can speed up the recognition process
    indices = np.argsort(np.array(width_list))
    return img_crop_list, img_num, indices


def batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num):
    """
    Batch for text recognition

    Parameters:
        img_crop_list: processed detected bounding box images
        img_num: number of bounding boxes from text detection
        indices: sorting for bounding boxes to speed up text recognition
        beg_img_no: the beginning number of bounding boxes for each batch of text recognition inference
        batch_num: number of images for each batch
    """
    norm_img_batch = []
    max_wh_ratio = 0
    end_img_no = min(img_num, beg_img_no + batch_num)
    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    return norm_img_batch

def post_processing_detection(frame, det_results):
    """
    Postprocess the results from text detection into bounding boxes

    Parameters:
        frame: input image
        det_results: inference results from text detection model
    """
    ori_im = frame.copy()
    data = {'image': frame}
    data_resize = processing.DetResizeForTest(data)
    data_list = []
    keep_keys = ['image', 'shape']
    for key in keep_keys:
        data_list.append(data_resize[key])
    img, shape_list = data_list

    shape_list = np.expand_dims(shape_list, axis=0)
    pred = det_results[0]
    if isinstance(pred, paddle.Tensor):
        pred = pred.numpy()
    segmentation = pred > 0.3

    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        mask = segmentation[batch_index]
        boxes, scores = processing.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
        boxes_batch.append({'points': boxes})
    post_result = boxes_batch
    dt_boxes = post_result[0]['points']
    dt_boxes = processing.filter_tag_det_res(dt_boxes, ori_im.shape)
    return dt_boxes

def run_paddle_ocr(source=0, flip=False, use_popup=False, skip_first_frames=0):
    """
    Main function to run the paddleOCR inference:
    1. Create a video player to play with target fps (utils.VideoPlayer).
    2. Prepare a set of frames for text detection and recognition.
    3. Run AI inference for both text detection and recognition.
    4. Visualize the results.

    Parameters:
        source: the webcam number to feed the video stream with primary webcam set to "0", or the video path.
        flip: to be used by VideoPlayer function for flipping capture image
        use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
        skip_first_frames: Number of frames to skip at the beginning of the video.
    """
    # create video player to play with target fps
    player = None
    try:
        player = utils.VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start video capturing
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:
            # grab the frame
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # if frame larger than full HD, reduce size to improve the performance
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            # preprocess image for text detection
            test_image = image_preprocess(frame, 640)

            # measure processing time for text detection
            start_time = time.time()
            # perform the inference step
            det_results = det_compiled_model([test_image])[det_output_layer]
            stop_time = time.time()

            # Postprocessing for Paddle Detection
            dt_boxes = post_processing_detection(frame, det_results)

            processing_times.append(stop_time - start_time)
            # use processing times from last 200 frames
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000

            # Preprocess detection results for recognition
            dt_boxes = processing.sorted_boxes(dt_boxes)
            batch_num = 6
            img_crop_list, img_num, indices = prep_for_rec(dt_boxes, frame)

            # For storing recognition results, include two parts:
            # txts are the recognized text results, scores are the recognition confidence level
            rec_res = [['', 0.0]] * img_num
            txts = []
            scores = []

            for beg_img_no in range(0, img_num, batch_num):

                # Recognition starts from here
                norm_img_batch = batch_text_box(
                    img_crop_list, img_num, indices, beg_img_no, batch_num)

                # Run inference for text recognition
                rec_results = rec_compiled_model([norm_img_batch])[rec_output_layer]

                # Postprocessing recognition results
                postprocess_op = processing.build_post_process(processing.postprocess_params)
                rec_result = postprocess_op(rec_results)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]
                if rec_res:
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            #draw text recognition results beside the image
            flattened_txts_list = []
            for item in txts:
                item = re.sub(r"[^a-zA-Z],", "", item)
                flattened_txts_list.extend(item.split(','))
            txts_modified_list = ["".join(filter(str.isalpha, item.strip().lower())) for item in flattened_txts_list]


            draw_img = processing.draw_ocr_box_txt(
                image,
                boxes,
                modified_list,
                txts_modified_list,
                scores,
                drop_score=0.5)
           
            # Visualize PaddleOCR results
            f_height, f_width = draw_img.shape[:2]
            fps = 1000 / processing_time_det
            cv2.putText(img=draw_img, text=f"Inference time: {processing_time_det:.1f}ms ({fps:.1f} FPS)",
                        org=(20, 40),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1000,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            # use this workaround if there is flickering
            if use_popup:
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(winname=title, mat=draw_img)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # encode numpy array to jpg
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                _, encoded_img = cv2.imencode(ext=".jpg", img=draw_img,
                                              params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                # create IPython image
                i = display.Image(data=encoded_img)
                # display the image in this notebook
                display.clear_output(wait=True)
                display.display(i)

    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # stop capturing
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()

#run_paddle_ocr(source=0, flip=False, use_popup=False)

# Test OCR results on video file
#run_paddle_ocr(source="test.mp4", flip=False, use_popup=True, skip_first_frames=0)
run_paddle_ocr(source= 0, flip=False, use_popup=True, skip_first_frames=0)