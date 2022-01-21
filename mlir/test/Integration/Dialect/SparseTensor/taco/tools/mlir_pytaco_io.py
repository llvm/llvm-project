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

_MTX_HEAD = "%%MatrixMarket"
_MTX_MATRIX = "matrix"
_MTX_COORDINATE = "coordinate"
_MTX_REAL = "real"
_MTX_SYMMETRY = "symmetric"
_MTX_GENERAL = "general"
_SYMMETRY_FIELD_ID = 4

# The TACO supported header for .mtx has the following five fields:
# . %%MatrixMarket
# . matrix | tensor
# . coordinate | array
# . real
# . symmetric | general
#
# This is what we support currently.
_SUPPORTED_HEADER_FIELDS = ((_MTX_HEAD,), (_MTX_MATRIX,), (_MTX_COORDINATE,),
                            (_MTX_REAL,), (_MTX_GENERAL, _MTX_SYMMETRY))

_A_SPACE = " "
_MTX_COMMENT = "%"
_TNS_COMMENT = "#"


def _coordinate_from_strings(strings: List[str]) -> List[int]:
  """"Return the coordinate represented by the input strings."""
  # Coordinates are 1-based in the text file and 0-based in memory.
  return [int(s) - 1 for s in strings]


def _read_coordinate_format(file: TextIO, tensor: Tensor,
                            is_symmetric: bool) -> None:
  """Reads tensor values in coordinate format."""
  rank = tensor.order
  # Process the data for the tensor.
  for line in file:
    if not line:
      continue

    fields = line.split(_A_SPACE)
    if rank != len(fields) - 1:
      raise ValueError("The format and data have mismatched ranks: "
                       f"{rank} vs {len(fields)-1}.")
    coordinate = _coordinate_from_strings(fields[:-1])
    value = float(fields[-1])
    tensor.insert(coordinate, value)
    if is_symmetric and coordinate[0] != coordinate[-1]:
      coordinate.reverse()
      tensor.insert(coordinate, value)


def _read_mtx(file: TextIO, fmt: Format) -> Tensor:
  """Inputs tensor from a text file with .mtx format."""
  # The first line should have this five fields:
  #   head tensor-kind format data-type symmetry
  fields = file.readline().rstrip("\n").split(_A_SPACE)
  tuple_to_str = lambda x: "|".join(x)
  if len(fields) != len(_SUPPORTED_HEADER_FIELDS):
    raise ValueError(
        "Expected first line with theses fields "
        f"{' '.join(map(tuple_to_str, _SUPPORTED_HEADER_FIELDS))}: "
        f"{' '.join(fields)}")

  for i, values in enumerate(_SUPPORTED_HEADER_FIELDS):
    if fields[i] not in values:
      raise ValueError(f"The {i}th field can only be one of these values "
                       f"{tuple_to_str(values)}: {fields[i]}")

  is_symmetric = (fields[_SYMMETRY_FIELD_ID] == _MTX_SYMMETRY)
  # Skip leading empty lines or comment lines.
  line = file.readline()
  while not line or line[0] == _MTX_COMMENT:
    line = file.readline()

  # Process the first data line with dimensions and number of non-zero values.
  fields = line.split(_A_SPACE)
  rank = fmt.rank()
  if rank != len(fields) - 1:
    raise ValueError("The format and data have mismatched ranks: "
                     f"{rank} vs {len(fields)-1}.")
  shape = fields[:-1]
  shape = [int(s) for s in shape]
  num_non_zero = float(fields[-1])

  # Read the tensor values in coordinate format.
  tensor = Tensor(shape, fmt)
  _read_coordinate_format(file, tensor, is_symmetric)
  return tensor


def _read_tns(file: TextIO, fmt: Format) -> Tensor:
  """Inputs tensor from a text file with .tns format."""
  rank = fmt.rank()
  coordinates = []
  values = []
  dtype = DType(Type.FLOAT64)

  for line in file:
    # Skip empty lines and comment lines.
    if not line or line[0] == _TNS_COMMENT:
      continue

    # Process each line with a coordinate and the value at the coordinate.
    fields = line.split(_A_SPACE)
    if rank != len(fields) - 1:
      raise ValueError("The format and data have mismatched ranks: "
                       f"{rank} vs {len(fields)-1}.")
    coordinates.append(tuple(_coordinate_from_strings(fields[:-1])))
    values.append(dtype.value(fields[-1]))

  return Tensor.from_coo(coordinates, values, fmt, dtype)


def _write_tns(file: TextIO, tensor: Tensor) -> None:
  """Outputs a tensor to a file using .tns format."""
  coords, non_zeros = tensor.get_coordinates_and_values()
  assert len(coords) == len(non_zeros)
  # Output a coordinate and the corresponding value in a line.
  for c, v in zip(coords, non_zeros):
    # The coordinates are 1-based in the text file and 0-based in memory.
    plus_one_to_str = lambda x: str(x + 1)
    file.write(f"{' '.join(map(plus_one_to_str,c))} {v}\n")


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

  with open(filename, "r") as file:
    return (_read_mtx(file, fmt) if filename.endswith(_MTX_FILENAME_SUFFIX) else
            _read_tns(file, fmt))


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

  with open(filename, "w") as file:
    return _write_tns(file, tensor)
