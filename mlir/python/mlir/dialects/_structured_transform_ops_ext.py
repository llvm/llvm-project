#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from ..ir import *
    from ._ods_common import get_op_result_or_value as _get_op_result_or_value
    from ..dialects import pdl, transform
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import List, Optional, Sequence, Tuple, Union, overload

StaticIntLike = Union[int, IntegerAttr]
ValueLike = Union[Operation, OpView, Value]
MixedInt = Union[StaticIntLike, ValueLike]

IntOrAttrList = Sequence[Union[IntegerAttr, int]]
OptionalIntList = Optional[Union[ArrayAttr, IntOrAttrList]]

BoolOrAttrList = Sequence[Union[BoolAttr, bool]]
OptionalBoolList = Optional[Union[ArrayAttr, BoolOrAttrList]]

MixedValues = Union[Sequence[Union[StaticIntLike, ValueLike]], ArrayAttr, ValueLike]

DynamicIndexList = Sequence[Union[MixedInt, Sequence[MixedInt]]]


def _dispatch_dynamic_index_list(
    indices: Union[DynamicIndexList, ArrayAttr],
) -> Tuple[List[ValueLike], Union[List[int], ArrayAttr], List[bool]]:
    """Dispatches a list of indices to the appropriate form.

    This is similar to the custom `DynamicIndexList` directive upstream:
    provided indices may be in the form of dynamic SSA values or static values,
    and they may be scalable (i.e., as a singleton list) or not. This function
    dispatches each index into its respective form. It also extracts the SSA
    values and static indices from various similar structures, respectively.
    """
    dynamic_indices = []
    static_indices = [ShapedType.get_dynamic_size()] * len(indices)
    scalable_indices = [False] * len(indices)

    # ArrayAttr: Extract index values.
    if isinstance(indices, ArrayAttr):
        indices = [idx for idx in indices]

    def process_nonscalable_index(i, index):
        """Processes any form of non-scalable index.

        Returns False if the given index was scalable and thus remains
        unprocessed; True otherwise.
        """
        if isinstance(index, int):
            static_indices[i] = index
        elif isinstance(index, IntegerAttr):
            static_indices[i] = index.value  # pytype: disable=attribute-error
        elif isinstance(index, (Operation, Value, OpView)):
            dynamic_indices.append(index)
        else:
            return False
        return True

    # Process each index at a time.
    for i, index in enumerate(indices):
        if not process_nonscalable_index(i, index):
            # If it wasn't processed, it must be a scalable index, which is
            # provided as a Sequence of one value, so extract and process that.
            scalable_indices[i] = True
            assert len(index) == 1
            ret = process_nonscalable_index(i, index[0])
            assert ret

    return dynamic_indices, static_indices, scalable_indices


# Dispatches `MixedValues` that all represents integers in various forms into
# the following three categories:
#   - `dynamic_values`: a list of `Value`s, potentially from op results;
#   - `packed_values`: a value handle, potentially from an op result, associated
#                      to one or more payload operations of integer type;
#   - `static_values`: an `ArrayAttr` of `i64`s with static values, from Python
#                      `int`s, from `IntegerAttr`s, or from an `ArrayAttr`.
# The input is in the form for `packed_values`, only that result is set and the
# other two are empty. Otherwise, the input can be a mix of the other two forms,
# and for each dynamic value, a special value is added to the `static_values`.
def _dispatch_mixed_values(
    values: MixedValues,
) -> Tuple[List[Value], Union[Operation, Value, OpView], DenseI64ArrayAttr]:
    dynamic_values = []
    packed_values = None
    static_values = None
    if isinstance(values, ArrayAttr):
        static_values = values
    elif isinstance(values, (Operation, Value, OpView)):
        packed_values = values
    else:
        static_values = []
        for size in values or []:
            if isinstance(size, int):
                static_values.append(size)
            else:
                static_values.append(ShapedType.get_dynamic_size())
                dynamic_values.append(_get_op_result_or_value(size))
        static_values = DenseI64ArrayAttr.get(static_values)

    return (dynamic_values, packed_values, static_values)


def _get_value_or_attribute_value(
    value_or_attr: Union[any, Attribute, ArrayAttr]
) -> any:
    if isinstance(value_or_attr, Attribute) and hasattr(value_or_attr, "value"):
        return value_or_attr.value
    if isinstance(value_or_attr, ArrayAttr):
        return _get_value_list(value_or_attr)
    return value_or_attr


def _get_value_list(
    sequence_or_array_attr: Union[Sequence[any], ArrayAttr]
) -> Sequence[any]:
    return [_get_value_or_attribute_value(v) for v in sequence_or_array_attr]


def _get_int_array_attr(values: Optional[Union[ArrayAttr, IntOrAttrList]]) -> ArrayAttr:
    if values is None:
        return None

    # Turn into a Python list of Python ints.
    values = _get_value_list(values)

    # Make an ArrayAttr of IntegerAttrs out of it.
    return ArrayAttr.get(
        [IntegerAttr.get(IntegerType.get_signless(64), v) for v in values]
    )


def _get_int_array_array_attr(
    values: Optional[Union[ArrayAttr, Sequence[Union[ArrayAttr, IntOrAttrList]]]]
) -> ArrayAttr:
    """Creates an ArrayAttr of ArrayAttrs of IntegerAttrs.

    The input has to be a collection of collection of integers, where any
    Python Sequence and ArrayAttr are admissible collections and Python ints and
    any IntegerAttr are admissible integers. Both levels of collections are
    turned into ArrayAttr; the inner level is turned into IntegerAttrs of i64s.
    If the input is None, an empty ArrayAttr is returned.
    """
    if values is None:
        return None

    # Make sure the outer level is a list.
    values = _get_value_list(values)

    # The inner level is now either invalid or a mixed sequence of ArrayAttrs and
    # Sequences. Make sure the nested values are all lists.
    values = [_get_value_list(nested) for nested in values]

    # Turn each nested list into an ArrayAttr.
    values = [_get_int_array_attr(nested) for nested in values]

    # Turn the outer list into an ArrayAttr.
    return ArrayAttr.get(values)


class BufferizeToAllocationOp:
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


class DecomposeOp:
    """Specialization for DecomposeOp class."""

    def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
        super().__init__(
            pdl.OperationType.get(), _get_op_result_or_value(target), loc=loc, ip=ip
        )


class FuseIntoContainingOp:
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


class GeneralizeOp:
    """Specialization for GeneralizeOp class."""

    def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
        super().__init__(
            pdl.OperationType.get(), _get_op_result_or_value(target), loc=loc, ip=ip
        )


class InterchangeOp:
    """Specialization for InterchangeOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        *,
        iterator_interchange: OptionalIntList = None,
        loc=None,
        ip=None,
    ):
        pdl_operation_type = pdl.OperationType.get()
        super().__init__(
            pdl_operation_type,
            _get_op_result_or_value(target),
            iterator_interchange=iterator_interchange,
            loc=loc,
            ip=ip,
        )


class MapCopyToThreadsOp:
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


class MaskedVectorizeOp:
    """Specialization for MaskedVectorizeOp class."""

    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        vector_sizes: Union[DynamicIndexList, ArrayAttr],
        *,
        vectorize_nd_extract: Optional[bool] = None,
        scalable_sizes: OptionalBoolList = None,
        static_vector_sizes: OptionalIntList = None,
        loc=None,
        ip=None,
    ):
        if scalable_sizes is None and static_vector_sizes is None:
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


class MatchOp:
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
            _get_op_result_or_value(target),
            ops=ArrayAttr.get(list(map(lambda s: StringAttr.get(s), names))),
            loc=loc,
            ip=ip,
        )


class MultiTileSizesOp:
    """Specialization for MultitileSizesOp class."""

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
        if divisor is None:
            divisor = 1
        super().__init__(
            result_type,
            result_type,
            result_type,
            _get_op_result_or_value(target),
            dimension=dimension,
            target_size=target_size,
            divisor=divisor,
            loc=loc,
            ip=ip,
        )


class PadOp:
    """Specialization for PadOp class."""

    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        padding_values: Optional[Union[ArrayAttr, Sequence[Attribute]]] = None,
        padding_dimensions: OptionalIntList = None,
        pad_to_multiple_of: OptionalIntList = None,
        pack_paddings: OptionalIntList = None,
        transpose_paddings: Optional[
            Union[ArrayAttr, Sequence[Union[ArrayAttr, IntOrAttrList]]]
        ] = None,
        copy_back_op: Optional[Union[str, StringAttr]] = None,
        loc=None,
        ip=None,
    ):
        transpose_paddings = _get_int_array_array_attr(transpose_paddings)

        pdl_operation_type = pdl.OperationType.get()
        super().__init__(
            pdl_operation_type,
            pdl_operation_type,
            pdl_operation_type,
            target,
            padding_values=padding_values,
            padding_dimensions=padding_dimensions,
            pad_to_multiple_of=pad_to_multiple_of,
            pack_paddings=pack_paddings,
            transpose_paddings=transpose_paddings,
            copy_back_op=copy_back_op,
            loc=loc,
            ip=ip,
        )


class ScalarizeOp:
    """Specialization for ScalarizeOp class."""

    def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
        pdl_operation_type = pdl.OperationType.get()
        super().__init__(
            pdl_operation_type, _get_op_result_or_value(target), loc=loc, ip=ip
        )


class SplitOp:
    """Specialization for SplitOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        dimension: Union[int, Attribute],
        split_point: Union[int, Operation, Value, Attribute],
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(split_point, int):
            static_split_point = split_point
            dynamic_split_point = None
        else:
            static_split_point = ShapedType.get_dynamic_size()
            dynamic_split_point = _get_op_result_or_value(split_point)

        target = _get_op_result_or_value(target)

        super().__init__(
            target.type,
            target.type,
            target,
            dimension=dimension,
            static_split_point=static_split_point,
            dynamic_split_point=dynamic_split_point,
            loc=loc,
            ip=ip,
        )


class TileOp:
  """Specialization for TileOp class."""

  @overload
  def __init__(
        self,
        loop_types: Union[Type, List[Type]],
        target: Union[Operation, Value],
        *,
        sizes: Optional[
            Union[Sequence[Union[int, IntegerAttr, Operation, Value]], ArrayAttr]
        ] = None,
        interchange: OptionalIntList = None,
        scalable_sizes: OptionalBoolList = None,
        loc=None,
        ip=None,
    ):
    ...

  @overload
  def __init__(
        self,
        target: Union[Operation, Value, OpView],
        *,
        sizes: Optional[
            Union[Sequence[Union[int, IntegerAttr, Operation, Value]], ArrayAttr]
        ] = None,
        interchange: OptionalIntList = None,
        scalable_sizes: OptionalBoolList = None,
        loc=None,
        ip=None,
    ):
    ...

  def __init__(
        self,
        loop_types_or_target: Union[Type, List[Type], Operation, Value],
        target_or_none: Optional[Union[Operation, Value, OpView]] = None,
        *,
        sizes: Optional[
            Union[Sequence[Union[int, IntegerAttr, Operation, Value]], ArrayAttr]
        ] = None,
        interchange: OptionalIntList = None,
        scalable_sizes: OptionalBoolList = None,
        loc=None,
        ip=None,
    ):
    if interchange is None:
      interchange = []
    if sizes is None:
      sizes = []

    static_sizes = []
    dynamic_sizes = []
    if isinstance(sizes, ArrayAttr):
      sizes_attr = sizes
    else:
      for size in sizes:
        if isinstance(size, int):
          static_sizes.append(size)
        else:
          static_sizes.append(ShapedType.get_dynamic_size())
          dynamic_sizes.append(_get_op_result_or_value(size))
      sizes_attr = DenseI64ArrayAttr.get(static_sizes)

    num_loops = sum(
        v if v == 0 else 1 for v in self.__extract_values(sizes_attr)
    )
    if scalable_sizes is None:
      scalable_sizes = [False] * len(self.__extract_values(sizes_attr))

    if isinstance(loop_types_or_target, (Operation, Value, OpView)):
      loop_types = [transform.AnyOpType.get()] * num_loops
      target = loop_types_or_target
      assert target_or_none is None, "Cannot construct TileOp with two targets."
    else:
      loop_types = (
          ([loop_types_or_target] * num_loops)
          if isinstance(loop_types_or_target, Type)
          else loop_types_or_target
      )
      target = target_or_none

    target = _get_op_result_or_value(target)

    super().__init__(
            target.type,
            loop_types,
            target,
            dynamic_sizes=dynamic_sizes,
            static_sizes=sizes_attr,
            interchange=interchange,
            scalable_sizes=scalable_sizes,
            loc=loc,
            ip=ip,
        )

  def __extract_values(self, attr: Optional[DenseI64ArrayAttr]) -> List[int]:
    if not attr:
      return []
    return [element for element in attr]


class TileToForallOp:
    """Specialization for TileToForallOp class."""

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


class VectorizeOp:
    """Specialization for VectorizeOp class."""

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
        pdl_operation_type = pdl.OperationType.get()
        super().__init__(
            pdl_operation_type,
            _get_op_result_or_value(target),
            disable_multi_reduction_to_contract_patterns=disable_multi_reduction_to_contract_patterns,
            disable_transfer_permutation_map_lowering_patterns=disable_transfer_permutation_map_lowering_patterns,
            vectorize_nd_extract=vectorize_nd_extract,
            vectorize_padding=vectorize_padding,
            loc=loc,
            ip=ip,
        )
