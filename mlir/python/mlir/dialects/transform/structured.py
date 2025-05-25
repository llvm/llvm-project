#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._structured_transform_ops_gen import *
from .._structured_transform_ops_gen import _Dialect
from .._structured_transform_enum_gen import *

try:
    from ...ir import *
    from ...dialects import transform
    from .._ods_common import (
        DynamicIndexList,
        IntOrAttrList,
        MixedValues,
        OptionalBoolList,
        OptionalIntList,
        _cext as _ods_cext,
        _dispatch_dynamic_index_list,
        _dispatch_mixed_values,
        _get_int_array_array_attr,
        _get_int_array_attr,
        _get_value_list,
        _get_value_or_attribute_value,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import List, Optional, Sequence, Union, overload


@_ods_cext.register_operation(_Dialect, replace=True)
class BufferizeToAllocationOp(BufferizeToAllocationOp):
    """Specialization for BufferizeToAllocationOp class."""

    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        memory_space: Optional[Union[int, str, Attribute]] = None,
        memcpy_op: Optional[str] = None,
        alloc_op: Optional[str] = None,
        bufferize_destination_only: Optional[bool] = None,
        loc=None,
        ip=None,
    ):
        # No other types are allowed, so hard-code those here.
        allocated_buffer_type = transform.AnyValueType.get()
        new_ops_type = transform.AnyOpType.get()

        if isinstance(memory_space, int):
            memory_space = str(memory_space)
        if isinstance(memory_space, str):
            memory_space = Attribute.parse(memory_space)

        super().__init__(
            allocated_buffer_type,
            new_ops_type,
            target,
            memory_space=memory_space,
            memcpy_op=memcpy_op,
            alloc_op=alloc_op,
            bufferize_destination_only=bufferize_destination_only,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class DecomposeOp(DecomposeOp):
    """Specialization for DecomposeOp class."""

    def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
        transformed_type = transform.AnyOpType.get()
        super().__init__(transformed_type, target, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class FuseIntoContainingOp(FuseIntoContainingOp):
    """Specialization for FuseIntoContainingOp class."""

    @overload
    def __init__(
        self,
        fused_op_type: Type,
        new_containing_op_type: Type,
        producer_op: Union[Operation, OpView, Value],
        containing_op: Union[Operation, OpView, Value],
        *,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        producer_op: Union[Operation, OpView, Value],
        containing_op: Union[Operation, OpView, Value],
        *,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        fused_op_type_or_producer_op: Union[Operation, OpView, Type, Value],
        new_containing_op_type_or_containing_op: Union[Operation, OpView, Type, Value],
        producer_op_or_none: Optional[Union[Operation, OpView, Value]] = None,
        containing_op_or_none: Optional[Union[Operation, OpView, Value]] = None,
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(fused_op_type_or_producer_op, Type):
            if not isinstance(new_containing_op_type_or_containing_op, Type):
                raise TypeError(
                    "If 'fused_op_type_or_producer_op' is a type, then "
                    "'new_containing_op_type_or_containing_op' is expected "
                    "to be one as well."
                )
            fused_op_type = fused_op_type_or_producer_op
            new_containing_op_type = new_containing_op_type_or_containing_op
            producer_op = producer_op_or_none
            containing_op = containing_op_or_none
        else:
            fused_op_type = transform.AnyOpType.get()
            new_containing_op_type = transform.AnyOpType.get()
            producer_op = fused_op_type_or_producer_op
            containing_op = new_containing_op_type_or_containing_op

        super().__init__(
            fused_op_type,
            new_containing_op_type,
            producer_op,
            containing_op,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class FuseOp(FuseOp):
    """Specialization for FuseOp class."""

    @overload
    def __init__(
        self,
        loop_types: Union[Type, Sequence[Type]],
        target: Union[Operation, Value, OpView],
        *,
        tile_sizes: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        tile_interchange: OptionalIntList = None,
        apply_cleanup: Optional[bool] = False,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, Value, OpView],
        *,
        tile_sizes: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        tile_interchange: OptionalIntList = None,
        apply_cleanup: Optional[bool] = False,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        loop_types_or_target: Union[Type, Sequence[Type], Operation, OpView, Value],
        target_or_none: Optional[Union[Operation, Value, OpView]] = None,
        *,
        tile_sizes: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        tile_interchange: OptionalIntList = None,
        apply_cleanup: Optional[bool] = False,
        loc=None,
        ip=None,
    ):
        tile_sizes = tile_sizes if tile_sizes else []
        tile_interchange = tile_interchange if tile_interchange else []
        _, tile_sizes, _ = _dispatch_dynamic_index_list(tile_sizes)
        _, tile_interchange, _ = _dispatch_dynamic_index_list(tile_interchange)
        num_loops = sum(0 if v == 0 else 1 for v in tile_sizes)

        if isinstance(loop_types_or_target, (Operation, Value, OpView)):
            loop_types = [transform.AnyOpType.get()] * num_loops
            target = loop_types_or_target
            assert target_or_none is None, "Cannot construct FuseOp with two targets."
        else:
            loop_types = (
                ([loop_types_or_target] * num_loops)
                if isinstance(loop_types_or_target, Type)
                else loop_types_or_target
            )
            target = target_or_none
        super().__init__(
            target.type,
            loop_types,
            target,
            tile_sizes=tile_sizes,
            tile_interchange=tile_interchange,
            apply_cleanup=apply_cleanup,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class GeneralizeOp(GeneralizeOp):
    """Specialization for GeneralizeOp class."""

    def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
        transformed_type = transform.AnyOpType.get()
        super().__init__(transformed_type, target, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class InterchangeOp(InterchangeOp):
    """Specialization for InterchangeOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        *,
        iterator_interchange: OptionalIntList = None,
        loc=None,
        ip=None,
    ):
        transformed_type = transform.AnyOpType.get()
        super().__init__(
            transformed_type,
            target,
            iterator_interchange=iterator_interchange,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class MapCopyToThreadsOp(MapCopyToThreadsOp):
    """Specialization for MapCopyToThreadsOp class."""

    @overload
    def __init__(
        self,
        forall_op_type: Type,
        tiled_op_type: Type,
        target: Union[Operation, OpView, Value],
        *,
        total_num_threads: Union[int, IntegerAttr],
        desired_bit_alignment: Union[int, IntegerAttr],
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        total_num_threads: Union[int, IntegerAttr],
        desired_bit_alignment: Union[int, IntegerAttr],
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        forall_op_type_or_target: Union[Operation, OpView, Type, Value],
        tiled_op_type_or_none: Optional[Type] = None,
        target_or_none: Optional[Union[Operation, OpView, Value]] = None,
        *,
        total_num_threads: Union[int, IntegerAttr],
        desired_bit_alignment: Union[int, IntegerAttr],
        loc=None,
        ip=None,
    ):
        if isinstance(forall_op_type_or_target, Type):
            forall_op_type = forall_op_type_or_target
            tiled_op_type = tiled_op_type_or_none
            target = target_or_none
        else:
            forall_op_type = transform.AnyOpType.get()
            tiled_op_type = transform.AnyOpType.get()
            target = forall_op_type_or_target

        super().__init__(
            forall_op_type,
            tiled_op_type,
            target,
            total_num_threads=total_num_threads,
            desired_bit_alignment=desired_bit_alignment,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class VectorizeOp(VectorizeOp):
    """Specialization for VectorizeOp class."""

    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        vector_sizes: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        *,
        vectorize_nd_extract: Optional[bool] = None,
        scalable_sizes: OptionalBoolList = None,
        static_vector_sizes: OptionalIntList = None,
        loc=None,
        ip=None,
    ):
        if (
            scalable_sizes is None
            and static_vector_sizes is None
            and vector_sizes is None
        ):
            dynamic_vector_sizes = []
        elif scalable_sizes is None and static_vector_sizes is None:
            (
                dynamic_vector_sizes,
                static_vector_sizes,
                scalable_sizes,
            ) = _dispatch_dynamic_index_list(vector_sizes)
        elif scalable_sizes is None or static_vector_sizes is None:
            raise TypeError(
                "'scalable_sizes' and 'static_vector_sizes' must either both "
                "be given explicitly or both be given as part of 'vector_sizes'."
            )
        else:
            dynamic_vector_sizes = vector_sizes

        super().__init__(
            target,
            vector_sizes=dynamic_vector_sizes,
            static_vector_sizes=static_vector_sizes,
            scalable_sizes=scalable_sizes,
            vectorize_nd_extract=vectorize_nd_extract,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class MatchOp(MatchOp):
    """Specialization for MatchOp class."""

    @overload
    @classmethod
    def match_op_names(
        cls,
        target: Union[Operation, Value],
        names: Union[str, Sequence[str]],
        *,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    @classmethod
    def match_op_names(
        cls,
        result_type: Type,
        target: Union[Operation, Value],
        names: Union[str, Sequence[str]],
        *,
        loc=None,
        ip=None,
    ):
        ...

    @classmethod
    def match_op_names(
        cls,
        result_type_or_target: Union[Type, Operation, Value],
        target_or_names: Union[Operation, Value, Sequence[str], str],
        names_or_none: Optional[Union[Sequence[str], str]] = None,
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(result_type_or_target, Type):
            result_type = result_type_or_target
            target = target_or_names
            names = names_or_none
        else:
            result_type = transform.AnyOpType.get()
            target = result_type_or_target
            names = target_or_names

        if isinstance(names, str):
            names = [names]

        return cls(
            result_type,
            target,
            ops=ArrayAttr.get(list(map(lambda s: StringAttr.get(s), names))),
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class MultiTileSizesOp(MultiTileSizesOp):
    """Specialization for MultiTileSizesOp class."""

    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, Value],
        *,
        dimension: Union[int, IntegerAttr],
        target_size: Union[int, IntegerAttr],
        divisor: Optional[Optional[Union[int, IntegerAttr]]] = None,
        loc=None,
        ip=None,
    ):
        super().__init__(
            result_type,
            result_type,
            result_type,
            target,
            dimension=dimension,
            target_size=target_size,
            divisor=divisor,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class PadOp(PadOp):
    """Specialization for PadOp class."""

    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        pad_to_multiple_of: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        padding_values: Optional[Union[ArrayAttr, Sequence[Attribute]]] = None,
        padding_dimensions: OptionalIntList = None,
        nofold_flags: OptionalIntList = None,
        transpose_paddings: Optional[
            Union[ArrayAttr, Sequence[Union[ArrayAttr, IntOrAttrList]]]
        ] = None,
        copy_back_op: Optional[Union[str, StringAttr]] = None,
        loc=None,
        ip=None,
    ):
        if pad_to_multiple_of is None:
            dynamic_pad_to_multiple_of = []
            static_pad_to_multiple_of = None
        else:
            (
                dynamic_pad_to_multiple_of,
                static_pad_to_multiple_of,
                _,
            ) = _dispatch_dynamic_index_list(pad_to_multiple_of)

        transpose_paddings = _get_int_array_array_attr(transpose_paddings)

        any_op_type = transform.AnyOpType.get()
        super().__init__(
            any_op_type,
            any_op_type,
            any_op_type,
            target,
            pad_to_multiple_of=dynamic_pad_to_multiple_of,
            padding_values=padding_values,
            padding_dimensions=padding_dimensions,
            static_pad_to_multiple_of=static_pad_to_multiple_of,
            nofold_flags=nofold_flags,
            transpose_paddings=transpose_paddings,
            copy_back_op=copy_back_op,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class ScalarizeOp(ScalarizeOp):
    """Specialization for ScalarizeOp class."""

    def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
        result_type = transform.AnyOpType.get()
        super().__init__(result_type, target, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class SplitOp(SplitOp):
    """Specialization for SplitOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        dimension: Union[int, Attribute],
        chunk_sizes: Union[int, Operation, Value, Attribute],
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(chunk_sizes, int):
            static_chunk_sizes = chunk_sizes
            dynamic_chunk_sizes = None
        else:
            static_chunk_sizes = ShapedType.get_dynamic_size()
            dynamic_chunk_sizes = chunk_sizes

        super().__init__(
            target.type,
            target,
            dimension=dimension,
            static_chunk_sizes=static_chunk_sizes,
            dynamic_chunk_sizes=dynamic_chunk_sizes,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class TileUsingForOp(TileUsingForOp):
    """Specialization for TileUsingForOp class."""

    @overload
    def __init__(
        self,
        loop_types: Union[Type, List[Type]],
        target: Union[Operation, Value],
        *,
        sizes: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        interchange: OptionalIntList = None,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, Value, OpView],
        *,
        sizes: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        interchange: OptionalIntList = None,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        loop_types_or_target: Union[Type, List[Type], Operation, Value],
        target_or_none: Optional[Union[Operation, Value, OpView]] = None,
        *,
        sizes: Optional[Union[DynamicIndexList, ArrayAttr]] = None,
        interchange: OptionalIntList = None,
        loc=None,
        ip=None,
    ):
        (
            dynamic_sizes,
            static_sizes,
            scalable_sizes,
        ) = _dispatch_dynamic_index_list(sizes)

        num_loops = sum(v if v == 0 else 1 for v in static_sizes)

        if isinstance(loop_types_or_target, (Operation, Value, OpView)):
            loop_types = [transform.AnyOpType.get()] * num_loops
            target = loop_types_or_target
            assert (
                target_or_none is None
            ), "Cannot construct TileUsingForOp with two targets."
        else:
            loop_types = (
                ([loop_types_or_target] * num_loops)
                if isinstance(loop_types_or_target, Type)
                else loop_types_or_target
            )
            target = target_or_none

        super().__init__(
            target.type,
            loop_types,
            target,
            dynamic_sizes=dynamic_sizes,
            static_sizes=static_sizes,
            interchange=interchange,
            scalable_sizes=scalable_sizes,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class TileUsingForallOp(TileUsingForallOp):
    """Specialization for TileUsingForallOp class."""

    @overload
    def __init__(
        self,
        loops_type: Type,
        tiled_op_type: Type,
        target: Union[Operation, Value, OpView],
        *,
        num_threads: Optional[MixedValues] = None,
        tile_sizes: MixedValues = None,
        mapping=None,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, Value, OpView],
        *,
        num_threads: Optional[MixedValues] = None,
        tile_sizes: MixedValues = None,
        mapping=None,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        loops_type_or_target: Union[
            Type, Union[Operation, Value, OpView]  # loops_type
        ],  # target
        tiled_op_type_or_none: Optional[Type] = None,
        target_or_none: Optional[Union[Operation, Value, OpView]] = None,
        *,
        num_threads: MixedValues = None,
        tile_sizes: MixedValues = None,
        mapping=None,
        loc=None,
        ip=None,
    ):
        # `Type` arguments in the front are optional: add default values to front.
        if isinstance(loops_type_or_target, Type):
            # First overload: type arguments provided.
            if not isinstance(tiled_op_type_or_none, Type):
                raise TypeError(
                    "If 'loops_type_or_target' is a type, then "
                    "'tiled_op_type_or_none' is expected to be one as well."
                )
            loops_type = loops_type_or_target
            tiled_op_type = tiled_op_type_or_none
            target = target_or_none
        else:
            # Last overload: type arguments missing.
            loops_type = transform.AnyOpType.get()
            tiled_op_type = transform.AnyOpType.get()
            target = loops_type_or_target

        # Unpack mixed num_threads.
        (
            dynamic_num_threads,
            packed_num_threads,
            num_threads_attr,
        ) = _dispatch_mixed_values(num_threads)

        # Unpack mixed tile_sizes.
        (
            dynamic_tile_sizes,
            packed_tile_sizes,
            tile_sizes_attr,
        ) = _dispatch_mixed_values(tile_sizes)

        super().__init__(
            loops_type,
            tiled_op_type,
            target=target,
            tile_sizes=dynamic_tile_sizes,
            packed_tile_sizes=packed_tile_sizes,
            static_tile_sizes=tile_sizes_attr,
            num_threads=dynamic_num_threads,
            packed_num_threads=packed_num_threads,
            static_num_threads=num_threads_attr,
            mapping=mapping,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class VectorizeChildrenAndApplyPatternsOp(VectorizeChildrenAndApplyPatternsOp):
    """Specialization for VectorizeChildrenAndApplyPatternsOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        *,
        disable_multi_reduction_to_contract_patterns: bool = False,
        disable_transfer_permutation_map_lowering_patterns: bool = False,
        vectorize_nd_extract: bool = False,
        vectorize_padding: bool = False,
        loc=None,
        ip=None,
    ):
        transformed_type = transform.AnyOpType.get()
        super().__init__(
            transformed_type,
            target,
            disable_multi_reduction_to_contract_patterns=disable_multi_reduction_to_contract_patterns,
            disable_transfer_permutation_map_lowering_patterns=disable_transfer_permutation_map_lowering_patterns,
            vectorize_nd_extract=vectorize_nd_extract,
            vectorize_padding=vectorize_padding,
            loc=loc,
            ip=ip,
        )
