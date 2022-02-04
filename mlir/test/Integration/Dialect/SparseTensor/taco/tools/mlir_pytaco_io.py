#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Experimental MLIR-PyTACO with sparse tensor support.

See http://tensor-compiler.org/ for TACO tensor compiler.

This module implements the PyTACO API for writing a tensor to a file or reading
a tensor from a file.

See the following links for Matrix Market Exchange (.mtx) format and FROSTT
(.tns) format:
  https://math.nist.gov/MatrixMarket/formats.html
  http://frostt.io/tensors/file-formats.html
"""

from typing import List, TextIO

from . import mlir_pytaco

# Define the type aliases so that we can write the implementation here as if
# it were part of mlir_pytaco.py.
Tensor = mlir_pytaco.Tensor
Format = mlir_pytaco.Format
DType = mlir_pytaco.DType
Type = mlir_pytaco.Type

# Constants used in the implementation.
_MTX_FILENAME_SUFFIX = ".mtx"
_TNS_FILENAME_SUFFIX = ".tns"


def read(filename: str, fmt: Format) -> Tensor:
  """Inputs a tensor from a given file.

  The name suffix of the file specifies the format of the input tensor. We
  currently only support .mtx format for support sparse tensors.

  Args:
    filename: A string input filename.
    fmt: The storage format of the tensor.

  Raises:
    ValueError: If filename doesn't end with .mtx or .tns, or fmt is not an
    instance of Format or fmt is not a sparse tensor.
  """
  if (not isinstance(filename, str) or
      (not filename.endswith(_MTX_FILENAME_SUFFIX) and
       not filename.endswith(_TNS_FILENAME_SUFFIX))):
    raise ValueError("Expected string filename ends with "
                     f"{_MTX_FILENAME_SUFFIX} or {_TNS_FILENAME_SUFFIX}: "
                     f"{filename}.")
  if not isinstance(fmt, Format) or fmt.is_dense():
    raise ValueError(f"Expected a sparse Format object: {fmt}.")

  return Tensor.from_file(filename, fmt, DType(Type.FLOAT64))


def write(filename: str, tensor: Tensor) -> None:
  """Outputs a tensor to a given file.

  The name suffix of the file specifies the format of the output. We currently
  only support .tns format.

  Args:
    filename: A string output filename.
    tensor: The tensor to output.

  Raises:
    ValueError: If filename doesn't end with .tns or tensor is not a Tensor.
  """
  if (not isinstance(filename, str) or
      not filename.endswith(_TNS_FILENAME_SUFFIX)):
    raise ValueError("Expected string filename ends with"
                     f" {_TNS_FILENAME_SUFFIX}: {filename}.")
  if not isinstance(tensor, Tensor):
    raise ValueError(f"Expected a Tensor object: {tensor}.")

  tensor.to_file(filename)
