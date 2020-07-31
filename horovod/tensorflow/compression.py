# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import tensorflow as tf


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating:
            tensor_decompressed = tf.cast(tensor, dtype=dtype)
        return tensor_decompressed

class INT8Compressor(Compressor):
    """Compress all floating point gradients to 8-bit."""
    #def compress(tensor, sigma):
    def compress(tensor, horovod_size, global_max, global_min):
        """Downcasts the tensor to 8-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating:
            
            valid_max = (2**(8-1)/horovod_size - 1)
            valid_min = (2**(8-1)/horovod_size - 1) * -1
            
            """
            tensor_compressed = 0.5*(1+tf.math.erf(tensor_compressed/(sigma*(2**0.5))))
            tensor_compressed = tensor_compressed-0.5
            tensor_compressed = tensor_compressed*(2**8)
            tensor_compressed = tf.clip_by_value(tensor_compressed, clip_value_min=-127.0, clip_value_max=127.0)
            tensor_compressed = tf.cast(tensor_compressed, dtype=tf.int8)
            """
            
            tensor_compressed = (tensor_compressed - global_min) / (global_max - global_min)
            tensor_compressed = tensor_compressed * (valid_max - valid_min) + valid_min
            tensor_compressed = tf.math.round(tensor_compressed)
            tensor_compressed = tf.clip_by_value(tensor_compressed, clip_value_min=valid_min, clip_value_max=valid_max)
            tensor_compressed = tf.cast(tensor_compressed, dtype=tf.int8)
            
        return tensor_compressed, tensor.dtype

    #def decompress(tensor, ctx, sigma):
    def decompress(tensor, ctx, global_max, global_min):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating:
            
            valid_max = (2**(8-1) - 1)
            valid_min = (2**(8-1) - 1) * -1
            """
            tensor_decompressed = tf.cast(tensor_decompressed, dtype=dtype)
            tensor_decompressed = tf.clip_by_value(tensor_decompressed, clip_value_min=-127.0, clip_value_max=127.0)
            tensor_decompressed = tensor_decompressed/(2**8)
            tensor_decompressed = tensor_decompressed+0.5
            tensor_decompressed = sigma*(2**0.5)*tf.math.erfinv(2*tensor_decompressed-1)
            """
            
            tensor_decompressed = tf.cast(tensor_decompressed, dtype=dtype)
            tensor_decompressed = (tensor_decompressed - valid_min) / (valid_max - valid_min)
            tensor_decompressed = tensor_decompressed * (global_max - global_min) + global_min
            
            
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor
    
    """Compress all floating point gradients to 8-bit."""
    int8 = INT8Compressor
