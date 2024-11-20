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
        ResultValueTypeTuple as _ResultValueTypeTuple,
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
        upper_bound: Optional[Union[int, _ResultValueT, AffineMap]],
        step: Optional[Union[int, Attribute]] = None,
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
          then symbols in the `lower_bound` affine map, in an increasing order.
        - `upper_bound_operands` is the list of arguments to substitute the dimensions,
          then symbols in the `upper_bound` affine map, in an increasing order.
        """

        if lower_bound_operands is None:
            lower_bound_operands = []
        if upper_bound_operands is None:
            upper_bound_operands = []

        if step is None:
            step = 1

        bounds_operands = [lower_bound_operands, upper_bound_operands]
        bounds = [lower_bound, upper_bound]
        bounds_names = ["lower", "upper"]
        for i, name in enumerate(bounds_names):
            if isinstance(bounds[i], int):
                bounds[i] = AffineMap.get_constant(bounds[i])
            elif isinstance(bounds[i], _ResultValueTypeTuple):
                if len(bounds_operands[i]):
                    raise ValueError(
                        f"Either a concrete {name} bound or an AffineMap in combination "
                        f"with {name} bound operands, but not both, is supported."
                    )
                if (
                    isinstance(bounds[i], (OpView, Operation))
                    and len(bounds[i].results) > 1
                ):
                    raise ValueError(
                        f"Only a single concrete value is supported for {name} bound."
                    )

                bounds_operands[i].append(_get_op_result_or_value(bounds[i]))
                bounds[i] = AffineMap.get_identity(1)

            if not isinstance(bounds[i], AffineMap):
                raise ValueError(
                    f"{name} bound must be int | ResultValueT | AffineMap."
                )
            if len(bounds_operands[i]) != bounds[i].n_inputs:
                raise ValueError(
                    f"Wrong number of {name} bound operands passed to AffineForOp; "
                    + f"Expected {bounds[i].n_inputs}, got {len(bounds_operands[i])}."
                )

        lower_bound, upper_bound = bounds

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
    stop,
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


@_ods_cext.register_operation(_Dialect, replace=True)
class AffineIfOp(AffineIfOp):
    """Specialization for the Affine if op class."""

    def __init__(
        self,
        cond: IntegerSet,
        results_: Optional[Type] = None,
        *,
        cond_operands: Optional[_VariadicResultValueT] = None,
        has_else: bool = False,
        loc=None,
        ip=None,
    ):
        """Creates an Affine `if` operation.

        - `cond` is the integer set used to determine which regions of code
          will be executed.
        - `results` are the list of types to be yielded by the operand.
        - `cond_operands` is the list of arguments to substitute the
          dimensions, then symbols in the `cond` integer set expression to
          determine whether they are in the set.
        - `has_else` determines whether the affine if operation has the else
          branch.
        """
        if results_ is None:
            results_ = []
        if cond_operands is None:
            cond_operands = []

        if cond.n_inputs != len(cond_operands):
            raise ValueError(
                f"expected {cond.n_inputs} condition operands, got {len(cond_operands)}"
            )

        operands = []
        operands.extend(cond_operands)
        results = []
        results.extend(results_)

        super().__init__(results, cond_operands, cond)
        self.regions[0].blocks.append(*[])
        if has_else:
            self.regions[1].blocks.append(*[])

    @property
    def then_block(self) -> Block:
        """Returns the then block of the if operation."""
        return self.regions[0].blocks[0]

    @property
    def else_block(self) -> Optional[Block]:
        """Returns the else block of the if operation."""
        if len(self.regions[1].blocks) == 0:
            return None
        return self.regions[1].blocks[0]
