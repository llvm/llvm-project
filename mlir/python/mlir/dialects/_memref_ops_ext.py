#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ._ods_common import get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union


class LoadOp:
  """Specialization for the MemRef load operation."""

  def __init__(self,
               memref: Union[Operation, OpView, Value],
               indices: Optional[Union[Operation, OpView,
                                       Sequence[Value]]] = None,
               *,
               loc=None,
               ip=None):
    """Creates a memref load operation.

    Args:
      memref: the buffer to load from.
      indices: the list of subscripts, may be empty for zero-dimensional
        buffers.
      loc: user-visible location of the operation.
      ip: insertion point.
    """
    memref_resolved = _get_op_result_or_value(memref)
    indices_resolved = [] if indices is None else _get_op_results_or_values(
        indices)
    return_type = MemRefType(memref_resolved.type).element_type
    super().__init__(return_type, memref, indices_resolved, loc=loc, ip=ip)
