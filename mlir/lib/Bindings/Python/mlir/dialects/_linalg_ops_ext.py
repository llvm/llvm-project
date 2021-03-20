#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence, Union
from ..ir import *
from ._ods_common import get_default_loc_context


class InitTensorOp:
  """Extends the linalg.init_tensor op."""

  def __init__(self,
               sizes: Union[Sequence[int], Sequence[Value]],
               element_type: Type,
               *,
               loc=None,
               ip=None):
    """Constructs an `init_tensor` with either static or dynamic sizes."""
    context = get_default_loc_context(loc)
    operands = []
    attributes = {}
    # TODO: Refactor the InitTensorOp to take an element type attribute and
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

    index_type = IndexType.get(context)
    attributes["static_sizes"] = ArrayAttr.get(
        [IntegerAttr.get(index_type, s) for s in static_size_ints],
        context=context)
    op = self.build_generic(results=[result_type],
                            operands=operands,
                            attributes=attributes,
                            loc=loc,
                            ip=ip)
    OpView.__init__(self, op)


class StructuredOpMixin:
  """All structured ops use the same mixin class."""

  def __init__(self, inputs, outputs=(), results=(), loc=None, ip=None):
    if outputs and results:
      raise ValueError(
          "Structured ops must have outputs or results, but not both.")
    super().__init__(
        self.build_generic(results=list(results),
                           operands=[list(inputs), list(outputs)],
                           loc=loc,
                           ip=ip))


def select_opview_mixin(parent_opview_cls):
  # TODO: This shouldn't be a heuristic: we should have a way to annotate
  # the OpView to note that it is a structured op.
  if ("__init__" not in parent_opview_cls.__dict__ and
      hasattr(parent_opview_cls, "inputs") and
      hasattr(parent_opview_cls, "outputs") and
      hasattr(parent_opview_cls, "result_tensors")):
    return StructuredOpMixin
