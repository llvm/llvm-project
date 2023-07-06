#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from ..ir import *
    from ._ods_common import get_op_result_or_value as _get_op_result_or_value
    from ..dialects import pdl, transform
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import List, Optional, Sequence, Union, overload

IntOrAttrList = Sequence[Union[IntegerAttr, int]]
OptionalIntList = Optional[Union[ArrayAttr, IntOrAttrList]]

BoolOrAttrList = Sequence[Union[BoolAttr, bool]]
OptionalBoolList = Optional[Union[ArrayAttr, BoolOrAttrList]]


def _get_int_int_array_attr(
    values: Optional[Union[ArrayAttr, Sequence[Union[ArrayAttr, IntOrAttrList]]]]
) -> ArrayAttr:
    """Creates an array attribute containing array attributes of integers.

    If the operand is already an array attribute, forwards it. Otherwise treats
    the operand as a list of attributes or integers, potentially interpserced, to
    create a new array-of-array attribute. Expects the thread-local MLIR context
    to have been set by the context manager.
    """
    if values is None:
        return ArrayAttr.get([])
    if isinstance(values, ArrayAttr):
        return values
    if isinstance(values, list):
        values = [
            ArrayAttr.get(
                [IntegerAttr.get(IntegerType.get_signless(64), v) for v in value]
            )
            for value in values
        ]

    return ArrayAttr.get(values)


class DecomposeOp:
    """Specialization for DecomposeOp class."""

    def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
        super().__init__(
            pdl.OperationType.get(), _get_op_result_or_value(target), loc=loc, ip=ip
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


class MatchOp:
    """Specialization for MatchOp class."""

    @classmethod
    def match_op_names(
        MatchOp,
        target: Union[Operation, Value],
        names: Sequence[str],
        loc=None,
        ip=None,
    ):
        pdl_operation_type = pdl.OperationType.get()
        return MatchOp(
            pdl_operation_type,
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
      target: Union[Operation, Value],
      *,
      padding_values: Optional[
          Optional[Union[ArrayAttr, Sequence[Attribute]]]
      ] = None,
      padding_dimensions: OptionalIntList = None,
      pack_paddings: OptionalIntList = None,
      transpose_paddings: Optional[
          Union[ArrayAttr, Sequence[Union[ArrayAttr, IntOrAttrList]]]
      ] = None,
      loc=None,
      ip=None,
  ):
    if transpose_paddings is None:
      transpose_paddings = []
    if pack_paddings is None:
      pack_paddings = []
    if padding_dimensions is None:
      padding_dimensions = []
    if padding_values is None:
      padding_values = []
    pdl_operation_type = pdl.OperationType.get()
    transpose_paddings_attr = _get_int_int_array_attr(transpose_paddings)
    super().__init__(
        pdl_operation_type,
        pdl_operation_type,
        _get_op_result_or_value(target),
        padding_values=padding_values,
        padding_dimensions=padding_dimensions,
        pack_paddings=pack_paddings,
        transpose_paddings=transpose_paddings_attr,
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


class VectorizeOp:
    """Specialization for VectorizeOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        *,
        vectorize_padding: Union[bool, BoolAttr] = False,
        loc=None,
        ip=None,
    ):
        pdl_operation_type = pdl.OperationType.get()
        if isinstance(vectorize_padding, bool):
            vectorize_padding = UnitAttr.get()
        super().__init__(
            pdl_operation_type,
            _get_op_result_or_value(target),
            vectorize_padding=vectorize_padding,
            loc=loc,
            ip=ip,
        )
