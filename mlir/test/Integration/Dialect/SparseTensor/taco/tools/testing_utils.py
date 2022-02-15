#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains the utilities to support testing.

import numpy as np


def compare_sparse_tns(expected: str, actual: str, rtol: float = 0.0001) -> bool:
  """Compares sparse tensor actual output file with expected output file.

  This routine assumes the input files are in FROSTT format. See
  http://frostt.io/tensors/file-formats.html for FROSTT (.tns) format.

  It also assumes the first line in the output file is a comment line.

  """
  with open(actual, "r") as actual_f:
    with open(expected, "r") as expected_f:
      # Skip the first comment line.
      _ = actual_f.readline()
      _ = expected_f.readline()

      # Compare the two lines of meta data
      if (actual_f.readline() != expected_f.readline() or
          actual_f.readline() != expected_f.readline()):
        return FALSE

  actual_data = np.loadtxt(actual, np.float64, skiprows=3)
  expected_data = np.loadtxt(expected, np.float64, skiprows=3)
  return np.allclose(actual_data, expected_data, rtol=rtol)


def file_as_string(file: str) -> str:
  """Returns contents of file as string."""
  with open(file, "r") as f:
    return f.read()


def run_test(f):
  """Prints the test name and runs the test."""
  print(f.__name__)
  f()
  return f
