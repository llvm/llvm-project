#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Re-export the objects provided by pybind.
from ..._mlir_libs._mlirDialectsLinalg import *

# These are the backing OpView classes generated from the linalg tablegen
# definitions following these steps:
#   DSL -> YAML -> tblgen -> pytblgen -> build/.../_linalg_ops_gen.py.
from .._linalg_ops_gen import *
from .._linalg_ops_gen import _Dialect
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
from .._ods_common import (
    get_op_result_or_value as _get_op_result_or_value,
    get_op_result_or_op_results as _get_op_result_or_op_results,
    _dispatch_mixed_values,
)
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


def _create_matmul_like_op(
    op_type,
    *ins: Union[Operation, OpView, Value],
    outs: Sequence[Union[Operation, OpView, Value]],
    indexing_maps: Optional[Sequence[AffineMapAttr]] = None,
    cast: Optional[Union[TypeFn, Attribute]] = None,
):
    ins = [_get_op_result_or_value(input) for input in ins]
    if len(outs) > 1:
        raise ValueError(f"{outs=} must have length 1.")
    init = _get_op_result_or_value(outs[0])
    result_types = [init.type] if isinstance(init.type, RankedTensorType) else []

    op = op_type(
        result_tensors=result_types,
        inputs=ins,
        outputs=[init],
        indexing_maps=indexing_maps,
        cast=cast,
    )
    fill_builtin_region(op.operation)
    return op


def matmul(
    *ins: Union[Operation, OpView, Value],
    outs: Sequence[Union[Operation, OpView, Value]],
    indexing_maps: Optional[Sequence[AffineMapAttr]] = None,
    cast: Optional[Union[TypeFn, Attribute]] = None,
):
    return _get_op_result_or_op_results(
        _create_matmul_like_op(
            MatmulOp, *ins, outs=outs, indexing_maps=indexing_maps, cast=cast
        )
    )


def batch_matmul(
    *ins: Union[Operation, OpView, Value],
    outs: Sequence[Union[Operation, OpView, Value]],
    indexing_maps: Optional[Sequence[AffineMapAttr]] = None,
    cast: Optional[Union[TypeFn, Attribute]] = None,
):
    return _get_op_result_or_op_results(
        _create_matmul_like_op(
            BatchMatmulOp, *ins, outs=outs, indexing_maps=indexing_maps, cast=cast
        )
    )


def batch_reduce_matmul(
    *ins: Union[Operation, OpView, Value],
    outs: Sequence[Union[Operation, OpView, Value]],
    indexing_maps: Optional[Sequence[AffineMapAttr]] = None,
    cast: Optional[Union[TypeFn, Attribute]] = None,
):
    return _get_op_result_or_op_results(
        _create_matmul_like_op(
            BatchReduceMatmulOp, *ins, outs=outs, indexing_maps=indexing_maps, cast=cast
        )
    )


def contract(
    *ins: Union[Operation, OpView, Value],
    outs: Sequence[Union[Operation, OpView, Value]],
    indexing_maps: Sequence[AffineMapAttr],
    cast: Optional[Union[TypeFn, Attribute]] = None,
):
    return _get_op_result_or_op_results(
        _create_matmul_like_op(
            ContractOp, *ins, outs=outs, indexing_maps=indexing_maps, cast=cast
        )
    )


# Extend and shadow the TableGen-derived version to make sure correct default
# indexing_maps are derived (as there is no mechanism for doing so given the
# Python API bypasses the C++-builders).
class ElementwiseOp_(ElementwiseOp):
    def __init__(
        self,
        result_tensors,
        inputs,
        outputs,
        kind,
        *,
        indexing_maps=None,
        loc=None,
        ip=None,
    ):
        if indexing_maps is None:
            inputs = [_get_op_result_or_value(in_) for in_ in inputs]
            for in0, in1 in zip(inputs[:-1], inputs[1:]):
                assert in0.type == in1.type
            output = _get_op_result_or_value(outputs[0])
            assert inputs[0].type == output.type
            num_args = len(inputs) + 1
            indexing_maps = [AffineMap.get_identity(output.type.rank)] * num_args

        super().__init__(
            result_tensors=result_tensors,
            inputs=inputs,
            outputs=outputs,
            kind=kind,
            indexing_maps=indexing_maps,
            loc=loc,
            ip=ip,
        )


ElementwiseOp = ElementwiseOp_


def elementwise(
    *ins: Union[Operation, OpView, Value],
    outs: Sequence[Union[Operation, OpView, Value]],
    kind: Union[ElementwiseKind, Attribute],
    indexing_maps: Optional[Sequence[AffineMapAttr]] = None,
):
    ins = [_get_op_result_or_value(input) for input in ins]
    if len(outs) != 1:
        raise ValueError(f"{outs=} must have length 1.")
    init = _get_op_result_or_value(outs[0])
    result_types = [init.type] if isinstance(init.type, RankedTensorType) else []

    op = ElementwiseOp(
        result_tensors=result_types,
        inputs=ins,
        outputs=[init],
        kind=kind,
        indexing_maps=indexing_maps,
    )
    fill_builtin_region(op.operation)
    return _get_op_result_or_op_results(op)


def pack(
    source,
    dest,
    inner_dims_pos,
    inner_tiles,
    *,
    padding_value=None,
    outer_dims_perm=None,
    loc=None,
    ip=None,
) -> ir.Value:
    (
        dynamic_inner_tiles,
        # packed here means %1:2 packing (results packing)
        _inner_tiles,
        static_inner_tiles,
    ) = _dispatch_mixed_values(inner_tiles)
    dest = _get_op_result_or_value(dest)
    result_type = dest.type if isinstance(dest.type, RankedTensorType) else None

    return _get_op_result_or_op_results(
        PackOp(
            result=result_type,
            source=source,
            dest=dest,
            inner_dims_pos=inner_dims_pos,
            inner_tiles=dynamic_inner_tiles,
            static_inner_tiles=static_inner_tiles,
            padding_value=padding_value,
            outer_dims_perm=outer_dims_perm,
            loc=loc,
            ip=ip,
        )
    )


def unpack(
    source,
    dest,
    inner_dims_pos,
    inner_tiles,
    *,
    outer_dims_perm=None,
    loc=None,
    ip=None,
) -> ir.Value:
    (
        dynamic_inner_tiles,
        # packed here means %1:2 packing (results packing)
        _inner_tiles,
        static_inner_tiles,
    ) = _dispatch_mixed_values(inner_tiles)
    dest = _get_op_result_or_value(dest)
    result_type = dest.type if isinstance(dest.type, RankedTensorType) else None
    return _get_op_result_or_op_results(
        UnPackOp(
            result=result_type,
            source=source,
            dest=dest,
            inner_dims_pos=inner_dims_pos,
            inner_tiles=dynamic_inner_tiles,
            static_inner_tiles=static_inner_tiles,
            outer_dims_perm=outer_dims_perm,
            loc=loc,
            ip=ip,
        )
    )


reduce = region_op(ReduceOp, terminator=YieldOp)
map = region_op(MapOp, terminator=YieldOp)
