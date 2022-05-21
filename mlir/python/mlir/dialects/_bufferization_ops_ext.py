#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from typing import Sequence, Union
  from ..ir import *
  from ._ods_common import get_default_loc_context as _get_default_loc_context

  from typing import Any, List, Union
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e


class AllocTensorOp:
  """Extends the bufferization.alloc_tensor op."""

  def __init__(self,
               sizes: Union[Sequence[int], Sequence[Value]],
               element_type: Type,
               *,
               loc=None,
               ip=None):
    """Constructs an `alloc_tensor` with either static or dynamic sizes."""
    context = get_default_loc_context(loc)
    operands = []
    attributes = {}
    # TODO: Refactor the AllocTensorOp to take an element type attribute and
    # then use normal result type inference, unifying the Python and C++ side
    # with a standard mechanism (versus stashing that in builders).
    if sizes and isinstance(sizes[0], Value):
      # Dynamic sizes.
      operands.extend(sizes)
      static_size_ints = [-1] * len(sizes)
      result_type = RankedTensorType.get(static_size_ints, element_type)
    else:
      # Static sizes.
      result_type = RankedTensorType.get(sizes, element_type)
      static_size_ints = sizes

    i64_type = IntegerType.get_signless(64)
    attributes["static_sizes"] = ArrayAttr.get(
        [IntegerAttr.get(i64_type, s) for s in static_size_ints],
        context=context)
    op = self.build_generic(
        results=[result_type],
        operands=operands,
        attributes=attributes,
        loc=loc,
        ip=ip)
    OpView.__init__(self, op)
