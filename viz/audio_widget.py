# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import array

import librosa
import numpy as np
import imgui
import pyaudio
from scipy.fftpack import fft, fftfreq

import struct
import dnnlib
from gui_utils import imgui_utils


# ----------------------------------------------------------------------------

class AudioWidget:
    def __init__(self, viz):
        self.viz = viz
        self.latent = dnnlib.EasyDict(x=0, y=0, anim=False, speed=0.25)
        self.latent_def = dnnlib.EasyDict(self.latent)
        self.step_y = 100

        self.waveform = [float('nan')] * 512
        self.frequencies = [float('nan')] * 512

        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.CHUNKS = 64

        self.p = pyaudio.PyAudio()


        self.chunk_idx = 0
        self.wf_data = np.zeros((self.CHUNKS, self.CHUNK), dtype='b')


        def callback(in_data, frame_count, time_info, status):
            wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', in_data)
            wf_data = np.array(wf_data, dtype='b')[::2] + 128
            self.wf_data[self.chunk_idx, :] = wf_data
            self.chunk_idx = (self.chunk_idx + 1) % self.CHUNKS
            return (in_data, pyaudio.paContinue)

        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=False,
            frames_per_buffer=self.CHUNK,
            stream_callback=callback,
        )
    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Audio')
            imgui.same_line(viz.label_w)
            _clicked, self.latent.anim = imgui.checkbox('Capture', self.latent.anim)
            imgui.same_line(viz.label_w + viz.font_size * 9)
            with imgui_utils.item_width(viz.font_size * 8):
                imgui.plot_lines('##wave', array.array('f', self.waveform), scale_min=0)

            imgui.same_line(viz.label_w + viz.font_size * 18)
            with imgui_utils.item_width(viz.font_size * 8):
                imgui.plot_lines('##freq', array.array('f', self.frequencies), scale_min=0)

        if self.latent.anim:
            self.update()

    def update(self):
        idxs = list(range(self.chunk_idx - self.CHUNKS, self.chunk_idx))
        wf_data =self.wf_data[idxs].reshape(-1)

        x = range(0, len(self.waveform))
        self.waveform[:] = np.interp(x, range(0, len(wf_data)), wf_data)

        x = range(0, len(self.frequencies))
        fft_data = np.abs(librosa.stft(y=(wf_data - 128.0) / 255.0, n_fft=2048, center=False))[
                   8:]  # cut lower frequencies
        fft_data = np.interp(x, range(0, fft_data.shape[0]), fft_data.mean(axis=1))

        self.frequencies[:] = (fft_data  - fft_data.mean()) / fft_data.std()
        self.viz.args.update(dict(latent_offset=self.frequencies))

        # magn = np.array(wf_data, dtype='float64') - 128.0
        # magn = np.abs(fft(magn))
        # return fftfreq(len(magn), 1.0 / self.RATE)

# ----------------------------------------------------------------------------
