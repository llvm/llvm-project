#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from typing import Optional, Sequence, Union
  from ..ir import *
  from ._ods_common import get_default_loc_context
  from .._mlir_libs._mlirDialectsLinalg import fill_builtin_region
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from ._ods_common import get_op_result_or_value as _get_op_result_or_value

def isa(cls: Type, ty: Type):
  try:
    cls(ty)
    return True
  except ValueError:
    return False


class StructuredOpMixin:
  """All structured ops use the same mixin class."""

  def __init__(self, inputs, outputs=(), results=(), loc=None, ip=None):
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
