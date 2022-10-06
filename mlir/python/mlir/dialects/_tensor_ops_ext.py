#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from typing import Any, Optional, Sequence, Union
from ._ods_common import get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values


class EmptyOp:
  """Extends the tensor.empty op."""

  def __init__(self,
               sizes: Sequence[Union[int, Value]],
               element_type: Type,
               *,
               loc=None,
               ip=None):
    """Constructs an `empty` with mixed static/dynamic sizes."""
    # TODO: Refactor the EmptyOp to take an element type attribute and
    # then use normal result type inference, unifying the Python and C++ side
    # with a standard mechanism (versus stashing that in builders).
    dynamic_sizes = []
    static_sizes = []
    for s in sizes:
      if isinstance(s, int):
        static_sizes.append(s)
      else:
        static_sizes.append(-1)
        dynamic_sizes.append(s)
    result_type = RankedTensorType.get(static_sizes, element_type)
    op = self.build_generic(
        results=[result_type],
        operands=dynamic_sizes,
        attributes={},
        loc=loc,
        ip=ip)
    OpView.__init__(self, op)
