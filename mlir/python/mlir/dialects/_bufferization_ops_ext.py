#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from typing import Sequence, Union
  from ..ir import *
  from ._ods_common import get_default_loc_context

  from typing import Any, List, Union
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e


class AllocTensorOp:
  """Extends the bufferization.alloc_tensor op."""

  def __init__(self,
               tensor_type: Type,
               dynamic_sizes: Sequence[Value],
               *,
               loc=None,
               ip=None):
    """Constructs an `alloc_tensor` with static and/or dynamic sizes."""
    context = get_default_loc_context(loc)
    op = self.build_generic(
        results=[tensor_type],
        operands=dynamic_sizes,
        attributes={},
        loc=loc,
        ip=ip)
    OpView.__init__(self, op)
