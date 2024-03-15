#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ....ir import Operation
from ...._mlir_libs import _mlirTransformInterpreter as _cextTransformInterpreter


TransformOptions = _cextTransformInterpreter.TransformOptions


def _unpack_operation(op):
    if isinstance(op, Operation):
        return op
    return op.operation


def apply_named_sequence(
    payload_root, transform_root, transform_module, transform_options=None
):
    """Applies the transformation script starting at the given transform root
    operation to the given payload operation. The module containing the
    transform root as well as the transform options should be provided.
    The transform operation must implement TransformOpInterface and the module
    must be a ModuleOp."""

    args = tuple(
        map(_unpack_operation, (payload_root, transform_root, transform_module))
    )
    if transform_options is None:
        _cextTransformInterpreter.apply_named_sequence(*args)
    else:
        _cextTransformInterpreter(*args, transform_options)
