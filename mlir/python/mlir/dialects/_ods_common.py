#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from collections import defaultdict

# Provide a convenient name for sub-packages to resolve the main C-extension
# with a relative import.
from .._mlir_libs import _mlir as _cext
from typing import (
    Callable as _Callable,
    Sequence as _Sequence,
    Type as _Type,
    TypeVar as _TypeVar,
    Union as _Union,
)

__all__ = [
    "equally_sized_accessor",
    "get_default_loc_context",
    "get_op_result_or_value",
    "get_op_results_or_values",
    "get_op_result_or_op_results",
    "segmented_accessor",
]


def segmented_accessor(elements, raw_segments, idx):
    """
    Returns a slice of elements corresponding to the idx-th segment.

      elements: a sliceable container (operands or results).
      raw_segments: an mlir.ir.Attribute, of DenseI32Array subclass containing
          sizes of the segments.
      idx: index of the segment.
    """
    segments = _cext.ir.DenseI32ArrayAttr(raw_segments)
    start = sum(segments[i] for i in range(idx))
    end = start + segments[idx]
    return elements[start:end]


def equally_sized_accessor(
    elements, n_variadic, n_preceding_simple, n_preceding_variadic
):
    """
    Returns a starting position and a number of elements per variadic group
    assuming equally-sized groups and the given numbers of preceding groups.

      elements: a sequential container.
      n_variadic: the number of variadic groups in the container.
      n_preceding_simple: the number of non-variadic groups preceding the current
          group.
      n_preceding_variadic: the number of variadic groups preceding the current
          group.
    """

    total_variadic_length = len(elements) - n_variadic + 1
    # This should be enforced by the C++-side trait verifier.
    assert total_variadic_length % n_variadic == 0

    elements_per_group = total_variadic_length // n_variadic
    start = n_preceding_simple + n_preceding_variadic * elements_per_group
    return start, elements_per_group


def get_default_loc_context(location=None):
    """
    Returns a context in which the defaulted location is created. If the location
    is None, takes the current location from the stack, raises ValueError if there
    is no location on the stack.
    """
    if location is None:
        # Location.current raises ValueError if there is no current location.
        return _cext.ir.Location.current.context
    return location.context


def get_op_result_or_value(
    arg: _Union[
        _cext.ir.OpView, _cext.ir.Operation, _cext.ir.Value, _cext.ir.OpResultList
    ]
) -> _cext.ir.Value:
    """Returns the given value or the single result of the given op.

    This is useful to implement op constructors so that they can take other ops as
    arguments instead of requiring the caller to extract results for every op.
    Raises ValueError if provided with an op that doesn't have a single result.
    """
    if isinstance(arg, _cext.ir.OpView):
        return arg.operation.result
    elif isinstance(arg, _cext.ir.Operation):
        return arg.result
    elif isinstance(arg, _cext.ir.OpResultList):
        return arg[0]
    else:
        assert isinstance(arg, _cext.ir.Value)
        return arg


def get_op_results_or_values(
    arg: _Union[
        _cext.ir.OpView,
        _cext.ir.Operation,
        _Sequence[_Union[_cext.ir.OpView, _cext.ir.Operation, _cext.ir.Value]],
    ]
) -> _Union[_Sequence[_cext.ir.Value], _cext.ir.OpResultList]:
    """Returns the given sequence of values or the results of the given op.

    This is useful to implement op constructors so that they can take other ops as
    lists of arguments instead of requiring the caller to extract results for
    every op.
    """
    if isinstance(arg, _cext.ir.OpView):
        return arg.operation.results
    elif isinstance(arg, _cext.ir.Operation):
        return arg.results
    else:
        return [get_op_result_or_value(element) for element in arg]


def get_op_result_or_op_results(
    op: _Union[_cext.ir.OpView, _cext.ir.Operation],
) -> _Union[_cext.ir.Operation, _cext.ir.OpResult, _Sequence[_cext.ir.OpResult]]:
    if isinstance(op, _cext.ir.OpView):
        op = op.operation
    return (
        list(get_op_results_or_values(op))
        if len(op.results) > 1
        else get_op_result_or_value(op)
        if len(op.results) > 0
        else op
    )


U = _TypeVar("U", bound=_cext.ir.Value)
SubClassValueT = _Type[U]

TypeCasterT = _Callable[
    [_Union[_cext.ir.Value, _cext.ir.OpResult]], _Union[SubClassValueT, None]
]

_VALUE_CASTERS: defaultdict[
    _cext.ir.TypeID,
    _Sequence[TypeCasterT],
] = defaultdict(list)


def has_value_caster(typeid: _cext.ir.TypeID):
    if not isinstance(typeid, _cext.ir.TypeID):
        raise ValueError(f"{typeid=} is not a TypeID")
    if typeid in _VALUE_CASTERS:
        return True
    return False


def get_value_caster(typeid: _cext.ir.TypeID):
    if not has_value_caster(typeid):
        raise ValueError(f"no registered caster for {typeid=}")
    return _VALUE_CASTERS[typeid]


def maybe_cast(
    val: _Union[
        _cext.ir.Value,
        _cext.ir.OpResult,
        _Sequence[_cext.ir.Value],
        _Sequence[_cext.ir.OpResult],
        _cext.ir.Operation,
    ]
) -> _Union[SubClassValueT, _Sequence[SubClassValueT], _cext.ir.Operation]:
    if isinstance(val, (tuple, list)):
        return tuple(map(maybe_cast, val))

    if not isinstance(val, _cext.ir.Value) and not isinstance(val, _cext.ir.OpResult):
        return val

    if has_value_caster(val.type.typeid):
        for caster in get_value_caster(val.type.typeid):
            if casted := caster(val):
                return casted
    return val
