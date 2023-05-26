#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Provide a convenient name for sub-packages to resolve the main C-extension
# with a relative import.
from .._mlir_libs import _mlir as _cext
from typing import Sequence as _Sequence, Union as _Union

__all__ = [
    "equally_sized_accessor",
    "extend_opview_class",
    "get_default_loc_context",
    "get_op_result_or_value",
    "get_op_results_or_values",
    "segmented_accessor",
]


def extend_opview_class(ext_module):
    """Decorator to extend an OpView class from an extension module.

    Extension modules can expose various entry-points:
      Stand-alone class with the same name as a parent OpView class (i.e.
      "ReturnOp"). A name-based match is attempted first before falling back
      to a below mechanism.

      def select_opview_mixin(parent_opview_cls):
        If defined, allows an appropriate mixin class to be selected dynamically
        based on the parent OpView class. Should return NotImplemented if a
        decision is not made.

    Args:
      ext_module: A module from which to locate extensions. Can be None if not
        available.

    Returns:
      A decorator that takes an OpView subclass and further extends it as
      needed.
    """

    def class_decorator(parent_opview_cls: type):
        if ext_module is None:
            return parent_opview_cls
        mixin_cls = NotImplemented
        # First try to resolve by name.
        try:
            mixin_cls = getattr(ext_module, parent_opview_cls.__name__)
        except AttributeError:
            # Fall back to a select_opview_mixin hook.
            try:
                select_mixin = getattr(ext_module, "select_opview_mixin")
            except AttributeError:
                pass
            else:
                mixin_cls = select_mixin(parent_opview_cls)

        if mixin_cls is NotImplemented or mixin_cls is None:
            return parent_opview_cls

        # Have a mixin_cls. Create an appropriate subclass.
        try:

            class LocalOpView(mixin_cls, parent_opview_cls):
                pass

        except TypeError as e:
            raise TypeError(
                f"Could not mixin {mixin_cls} into {parent_opview_cls}"
            ) from e
        LocalOpView.__name__ = parent_opview_cls.__name__
        LocalOpView.__qualname__ = parent_opview_cls.__qualname__
        return LocalOpView

    return class_decorator


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
