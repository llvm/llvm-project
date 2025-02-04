#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Re-export the objects provided by pybind.
from ..._mlir_libs._mlirDialectsLinalg import *

# These are the backing OpView classes generated from the linalg tablegen
# definitions following these steps:
#   DSL -> YAML -> tblgen -> pytblgen -> build/.../_linalg_ops_gen.py.
from .._linalg_ops_gen import *
from .._linalg_enum_gen import *
from .._linalg_enum_gen import _iteratortypeenum

# These are the ground truth functions defined as:
# ```
#    @linalg_structured_op
#    def matmul(A=TensorDef(T1, S.M, S.K),
#               B=TensorDef(T2, S.K, S.N),
#               C=TensorDef(U, S.M, S.N, output=True)):
# ```
# using the linalg-py eDSL.
# The linalg-py eDSL builds a python representation (PyRepr) that is
# used in following ways:
#  1. PyRepr -> YAML to generate the C++ and Python .td files. These
#     then turn into the core C++ Op classes and Python OpView classes
#     respectively (made available in _linalg_ops_gen). The generic OpView class
#     mechanism makes the C++ classes available to python through the CAPI.
#     PyRepr -> YAML currently occurs before compiler compile time.
#     The other steps in this category occur at compiler compile time.
#  2. PyRepr -> linalg.core_named_ops calls: piggybacks on the
#     _linalg_ops_gen classes and the OpView mechanism to build IR at
#     runtime in python:
#       a. by default, the Named Op Form is emitted, e.g.:
#          `linalg.matmul(lhs, rhs, outs=[out])` creates the following IR:
#          ```
#             %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>)
#                               outs(%0 : tensor<4x8xf32>)
#                  -> tensor<4x8xf32>
#          ```
#       b. by setting emit_generic=True, the Generic Op Form is emitted, e.g.:
#           `linalg.matmul(lhs, rhs, outs=[out], emit_generic=True)` creates the following IR:
#          ```
#             %1 = linalg.generic {indexing_maps = [...], iterator_types = [...]}
#               ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>)
#              outs(%0 : tensor<4x8xf32>) {
#               ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
#                  ...
#                  linalg.yield %3 : f32
#             } -> tensor<4x8xf32>
#          ```
#  3. PyRepr -> Runtime Custom Op definitions: directly generates a
#     linalg.generic form like in 2.b.
#     !!!WARNING!!!: if one creates a runtime custom op with the same name
#     as an existing core named op, step 2. will likely take precedence.
#     TODO: guard against surprises and fail create Runtime Custom Ops with
#     the same name as existing Core Named Ops.
from .opdsl.ops.core_named_ops import *

from ...ir import *
from .._ods_common import get_op_result_or_value as _get_op_result_or_value
from ...extras.meta import region_op


def transpose(
    input: Union[Operation, OpView, Sequence[Value]],
    *,
    outs: List[Union[Operation, OpView, Sequence[Value]]],
    permutation: Union[DenseI64ArrayAttr, List[int]],
):
    input = _get_op_result_or_value(input)
    if len(outs) > 1:
        raise ValueError(f"{outs=} must have length 1.")
    init = _get_op_result_or_value(outs[0])
    result_types = [init.type] if isinstance(init.type, RankedTensorType) else []

    op = TransposeOp(
        result=result_types,
        input=input,
        init=init,
        permutation=permutation,
    )
    fill_builtin_region(op.operation)
    return op


def broadcast(
    input: Union[Operation, OpView, Sequence[Value]],
    *,
    outs: List[Union[Operation, OpView, Sequence[Value]]],
    dimensions: Union[DenseI64ArrayAttr, List[int]],
):
    input = _get_op_result_or_value(input)
    if len(outs) > 1:
        raise ValueError(f"{outs=} must have length 1.")
    init = _get_op_result_or_value(outs[0])
    result_types = [init.type] if isinstance(init.type, RankedTensorType) else []

    op = BroadcastOp(
        result=result_types,
        input=input,
        init=init,
        dimensions=dimensions,
    )
    fill_builtin_region(op.operation)
    return op


@register_attribute_builder("IteratorTypeArrayAttr")
def _IteratorTypeArrayAttr(x, context):
    return ArrayAttr.get([_iteratortypeenum(v, context) for v in x])


# The underscore is needed here so that there's no collision with opdsl generation.
class GenericOp_(GenericOp):
    def __init__(
        self,
        inputs,
        outputs,
        indexing_maps,
        iterator_types,
        *,
        doc=None,
        library_call=None,
        loc=None,
        ip=None,
    ):
        result_types = []
        if isinstance(outputs[0].type, RankedTensorType):
            result_types = [o.type for o in outputs]

        super().__init__(
            result_types,
            inputs,
            outputs,
            indexing_maps,
            iterator_types,
            doc=doc,
            library_call=library_call,
            loc=loc,
            ip=ip,
        )
        element_types = [i.type.element_type for i in inputs] + [
            o.type.element_type for o in outputs
        ]
        self.regions[0].blocks.append(*element_types)


generic = region_op(GenericOp_, terminator=YieldOp)
