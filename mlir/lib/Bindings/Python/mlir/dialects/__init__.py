#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Re-export the parent _cext so that every level of the API can get it locally.
from .. import _cext

def _segmented_accessor(elements, raw_segments, idx):
  """
  Returns a slice of elements corresponding to the idx-th segment.

    elements: a sliceable container (operands or results).
    raw_segments: an mlir.ir.Attribute, of DenseIntElements subclass containing
        sizes of the segments.
    idx: index of the segment.
  """
  segments = _cext.ir.DenseIntElementsAttr(raw_segments)
  start = sum(segments[i] for i in range(idx))
  end = start + segments[idx]
  return elements[start:end]


def _equally_sized_accessor(elements, n_variadic, n_preceding_simple,
                            n_preceding_variadic):
  """
  Returns a starting position and a number of elements per variadic group
  assuming equally-sized groups and the given numbers of preceding groups.

    elements: a sequential container.
    n_variadic: the number of variadic groups in the container.
    n_preceding_simple: the number of non-variadic groups preceding the current
        group.
    n_preceding_variadic: the number of variadic groups preceding the current
        group.
  """

  total_variadic_length = len(elements) - n_variadic + 1
  # This should be enforced by the C++-side trait verifier.
  assert total_variadic_length % n_variadic == 0

  elements_per_group = total_variadic_length // n_variadic
  start = n_preceding_simple + n_preceding_variadic * elements_per_group
  return start, elements_per_group
