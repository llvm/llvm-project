#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


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
