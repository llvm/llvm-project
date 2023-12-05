#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._affine_ops_gen import *
from ._affine_ops_gen import _Dialect

try:
    from ..ir import *
    from ._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
        _cext as _ods_cext,
        ResultValueT as _ResultValueT,
        VariadicResultValueT as _VariadicResultValueT,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union


@_ods_cext.register_operation(_Dialect, replace=True)
class AffineForOp(AffineForOp):
    """Specialization for the Affine for op class."""

    def __init__(
        self,
        lower_bound: Union[int, _ResultValueT, AffineMap],
        upper_bound: Optional[Union[int, _ResultValueT, AffineMap]] = None,
        step: Optional[Union[int, _ResultValueT]] = None,
        iter_args: Optional[_ResultValueT] = None,
        *,
        lower_bound_operands: Optional[_VariadicResultValueT] = None,
        upper_bound_operands: Optional[_VariadicResultValueT] = None,
        loc=None,
        ip=None,
    ):
        """Creates an Affine `for` operation.

        - `lower_bound` is the affine map to use as lower bound of the loop.
        - `upper_bound` is the affine map to use as upper bound of the loop.
        - `step` is the value to use as loop step.
        - `iter_args` is a list of additional loop-carried arguments or an operation
          producing them as results.
        - `lower_bound_operands` is the list of arguments to substitute the dimensions,
          then symbols in the `lower_bound` affine map, in an increasing order
        - `upper_bound_operands` is the list of arguments to substitute the dimensions,
          then symbols in the `upper_bound` affine map, in an increasing order.
        """

        if lower_bound_operands is None:
            lower_bound_operands = []
        if upper_bound_operands is None:
            upper_bound_operands = []

        if step is None:
            step = 1
        if upper_bound is None:
            upper_bound, lower_bound = lower_bound, 0

        if isinstance(lower_bound, int):
            lower_bound = AffineMap.get_constant(lower_bound)
        elif isinstance(lower_bound, _ResultValueT):
            lower_bound_operands.append(lower_bound)
            lower_bound = AffineMap.get_constant(1)

        if not isinstance(lower_bound, AffineMap):
            raise ValueError(f"{lower_bound=} must be int | ResultValueT | AffineMap")

        if isinstance(upper_bound, int):
            upper_bound = AffineMap.get_constant(upper_bound)
        elif isinstance(upper_bound, _ResultValueT):
            upper_bound_operands.append(upper_bound)
            upper_bound = AffineMap.get_constant(1)

        if not isinstance(upper_bound, AffineMap):
            raise ValueError(f"{upper_bound=} must be int | ResultValueT | AffineMap")

        if iter_args is None:
            iter_args = []
        iter_args = _get_op_results_or_values(iter_args)

        results = [arg.type for arg in iter_args]
        super().__init__(
            results_=results,
            lowerBoundOperands=_get_op_results_or_values(lower_bound_operands),
            upperBoundOperands=_get_op_results_or_values(upper_bound_operands),
            inits=list(iter_args),
            lowerBoundMap=AffineMapAttr.get(lower_bound),
            upperBoundMap=AffineMapAttr.get(upper_bound),
            step=step,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(IndexType.get(), *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def induction_variable(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments[0]

    @property
    def inner_iter_args(self):
        """Returns the loop-carried arguments usable within the loop.

        To obtain the loop-carried operands, use `iter_args`.
        """
        return self.body.arguments[1:]


def for_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    for_op = AffineForOp(
        start,
        stop,
        step,
        iter_args=iter_args,
        loc=loc,
        ip=ip,
    )
    iv = for_op.induction_variable
    iter_args = tuple(for_op.inner_iter_args)
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args
        elif len(iter_args) == 1:
            yield iv, iter_args[0]
        else:
            yield iv
