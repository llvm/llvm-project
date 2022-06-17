#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Supports the PyTACO API with the MLIR-PyTACO implementation.

See http://tensor-compiler.org/ for TACO tensor compiler.

This module exports the MLIR-PyTACO implementation through the language defined
by PyTACO. In particular, it defines the function and type aliases and constants
needed for the PyTACO API to support the execution of PyTACO programs using the
MLIR-PyTACO implementation.
"""

from . import mlir_pytaco
from . import mlir_pytaco_io

# Functions defined by PyTACO API.
ceil = mlir_pytaco.ceil
floor = mlir_pytaco.floor
get_index_vars = mlir_pytaco.get_index_vars
from_array = mlir_pytaco.Tensor.from_array
read = mlir_pytaco_io.read
write = mlir_pytaco_io.write

# Classes defined by PyTACO API.
dtype = mlir_pytaco.DType
mode_format = mlir_pytaco.ModeFormat
mode_ordering = mlir_pytaco.ModeOrdering
mode_format_pack = mlir_pytaco.ModeFormatPack
format = mlir_pytaco.Format
index_var = mlir_pytaco.IndexVar
tensor = mlir_pytaco.Tensor
index_expression = mlir_pytaco.IndexExpr
access = mlir_pytaco.Access

# Data type constants defined by PyTACO API.
int8 = mlir_pytaco.DType(mlir_pytaco.Type.INT8)
int16 = mlir_pytaco.DType(mlir_pytaco.Type.INT16)
int32 = mlir_pytaco.DType(mlir_pytaco.Type.INT32)
int64 = mlir_pytaco.DType(mlir_pytaco.Type.INT64)
float32 = mlir_pytaco.DType(mlir_pytaco.Type.FLOAT32)
float64 = mlir_pytaco.DType(mlir_pytaco.Type.FLOAT64)
complex64 = mlir_pytaco.DType(mlir_pytaco.Type.COMPLEX64)
complex128 = mlir_pytaco.DType(mlir_pytaco.Type.COMPLEX128)

# Storage format constants defined by the PyTACO API. In PyTACO, each storage
# format constant has two aliasing names.
compressed = mlir_pytaco.ModeFormat.COMPRESSED
Compressed = mlir_pytaco.ModeFormat.COMPRESSED
dense = mlir_pytaco.ModeFormat.DENSE
Dense = mlir_pytaco.ModeFormat.DENSE
