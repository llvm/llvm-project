#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from ._scf_ops_gen import *
from ._scf_ops_gen import _Dialect
from . import arith
from ..extras.meta import region_op, region_adder

try:
    from ..ir import *
    from ..util import is_index_type
    from ._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import List, Optional, Sequence, Tuple, Union


@_ods_cext.register_operation(_Dialect, replace=True)
class ForOp(ForOp):
    """Specialization for the SCF for op class."""

    def __init__(
        self,
        lower_bound,
        upper_bound,
        step,
        iter_args: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Creates an SCF `for` operation.

        - `lower_bound` is the value to use as lower bound of the loop.
        - `upper_bound` is the value to use as upper bound of the loop.
        - `step` is the value to use as loop step.
        - `iter_args` is a list of additional loop-carried arguments or an operation
          producing them as results.
        """
        if iter_args is None:
            iter_args = []
        iter_args = _get_op_results_or_values(iter_args)

        results = [arg.type for arg in iter_args]
        super().__init__(
            results, lower_bound, upper_bound, step, iter_args, loc=loc, ip=ip
        )
        self.regions[0].blocks.append(self.operands[0].type, *results)

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


def _dispatch_index_op_fold_results(
    ofrs: Sequence[Union[Operation, OpView, Value, int]],
) -> Tuple[List[Value], List[int]]:
    """`mlir::dispatchIndexOpFoldResults`"""
    dynamic_vals = []
    static_vals = []
    for ofr in ofrs:
        if isinstance(ofr, (Operation, OpView, Value)):
            val = _get_op_result_or_value(ofr)
            dynamic_vals.append(val)
            static_vals.append(ShapedType.get_dynamic_size())
        else:
            static_vals.append(ofr)
    return dynamic_vals, static_vals


@_ods_cext.register_operation(_Dialect, replace=True)
class ForallOp(ForallOp):
    """Specialization for the SCF forall op class."""

    def __init__(
        self,
        lower_bounds: Sequence[Union[Operation, OpView, Value, int]],
        upper_bounds: Sequence[Union[Operation, OpView, Value, int]],
        steps: Sequence[Union[Value, int]],
        shared_outs: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        mapping=None,
        loc=None,
        ip=None,
    ):
        """Creates an SCF `forall` operation.

        - `lower_bounds` are the values to use as lower bounds of the loop.
        - `upper_bounds` are the values to use as upper bounds of the loop.
        - `steps` are the values to use as loop steps.
        - `shared_outs` is a list of additional loop-carried arguments or an operation
          producing them as results.
        """
        assert (
            len(lower_bounds) == len(upper_bounds) == len(steps)
        ), "Mismatch in length of lower bounds, upper bounds, and steps"
        if shared_outs is None:
            shared_outs = []
        shared_outs = _get_op_results_or_values(shared_outs)

        dynamic_lbs, static_lbs = _dispatch_index_op_fold_results(lower_bounds)
        dynamic_ubs, static_ubs = _dispatch_index_op_fold_results(upper_bounds)
        dynamic_steps, static_steps = _dispatch_index_op_fold_results(steps)

        results = [arg.type for arg in shared_outs]
        super().__init__(
            results,
            dynamic_lbs,
            dynamic_ubs,
            dynamic_steps,
            static_lbs,
            static_ubs,
            static_steps,
            shared_outs,
            mapping=mapping,
            loc=loc,
            ip=ip,
        )
        rank = len(static_lbs)
        iv_types = [IndexType.get()] * rank
        self.regions[0].blocks.append(*iv_types, *results)

    @property
    def body(self) -> Block:
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def rank(self) -> int:
        """Returns the number of induction variables the loop has."""
        return len(self.staticLowerBound)

    @property
    def induction_variables(self) -> BlockArgumentList:
        """Returns the induction variables usable within the loop."""
        return self.body.arguments[: self.rank]

    @property
    def inner_iter_args(self) -> BlockArgumentList:
        """Returns the loop-carried arguments usable within the loop.

        To obtain the loop-carried operands, use `iter_args`.
        """
        return self.body.arguments[self.rank :]

    def terminator(self) -> InParallelOp:
        """
        Returns the loop terminator if it exists.
        Otherwise, creates a new one.
        """
        ops = self.body.operations
        with InsertionPoint(self.body):
            if not ops:
                return InParallelOp()
            last = ops[len(ops) - 1]
            return last if isinstance(last, InParallelOp) else InParallelOp()


@_ods_cext.register_operation(_Dialect, replace=True)
class InParallelOp(InParallelOp):
    """Specialization of the SCF forall.in_parallel op class."""

    def __init__(self, loc=None, ip=None):
        super().__init__(loc=loc, ip=ip)
        self.region.blocks.append()

    @property
    def block(self) -> Block:
        return self.region.blocks[0]


@_ods_cext.register_operation(_Dialect, replace=True)
class IfOp(IfOp):
    """Specialization for the SCF if op class."""

    def __init__(self, cond, results_=None, *, hasElse=False, loc=None, ip=None):
        """Creates an SCF `if` operation.

        - `cond` is a MLIR value of 'i1' type to determine which regions of code will be executed.
        - `hasElse` determines whether the if operation has the else branch.
        """
        if results_ is None:
            results_ = []
        operands = []
        operands.append(cond)
        results = []
        results.extend(results_)
        super().__init__(results, cond, loc=loc, ip=ip)
        self.regions[0].blocks.append(*[])
        if hasElse:
            self.regions[1].blocks.append(*[])

    @property
    def then_block(self):
        """Returns the then block of the if operation."""
        return self.regions[0].blocks[0]

    @property
    def else_block(self):
        """Returns the else block of the if operation."""
        return self.regions[1].blocks[0]


def for_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            p = arith.constant(IndexType.get(), p)
        elif isinstance(p, float):
            raise ValueError(f"{p=} must be int.")
        params[i] = p

    start, stop, step = params

    for_op = ForOp(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = for_op.induction_variable
    iter_args = tuple(for_op.inner_iter_args)
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args, for_op.results
        elif len(iter_args) == 1:
            yield iv, iter_args[0], for_op.results[0]
        else:
            yield iv


def _parfor(op_ctor):
    def _base(
        lower_bounds, upper_bounds=None, steps=None, *, loc=None, ip=None, **kwargs
    ):
        if upper_bounds is None:
            upper_bounds = lower_bounds
            lower_bounds = [0] * len(upper_bounds)
        if steps is None:
            steps = [1] * len(lower_bounds)

        params = [lower_bounds, upper_bounds, steps]
        for i, p in enumerate(params):
            for j, pp in enumerate(p):
                if isinstance(pp, int):
                    pp = arith.constant(IndexType.get(), pp)
                assert isinstance(pp, Value), f"expected ir.Value, got {type(pp)=}"
                if not is_index_type(pp.type):
                    pp = arith.index_cast(pp)
                p[j] = pp
            params[i] = p

        return op_ctor(*params, loc=loc, ip=ip, **kwargs)

    return _base


def _parfor_cm(op_ctor):
    def _base(*args, **kwargs):
        for_op = _parfor(op_ctor)(*args, **kwargs)
        block = for_op.regions[0].blocks[0]
        block_args = tuple(block.arguments)
        with InsertionPoint(block):
            yield block_args

    return _base


forall = _parfor_cm(ForallOp)


class ParallelOp(ParallelOp):
    def __init__(
        self,
        lower_bounds: Sequence[Union[Operation, OpView, Value, int]],
        upper_bounds: Sequence[Union[Operation, OpView, Value, int]],
        steps: Sequence[Union[Value, int]],
        inits: Optional[Sequence[Union[Operation, OpView, Sequence[Value]]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        assert len(lower_bounds) == len(upper_bounds) == len(steps)
        if inits is None:
            inits = []
        results = [i.type for i in inits]
        iv_types = [IndexType.get()] * len(lower_bounds)
        super().__init__(
            results,
            lower_bounds,
            upper_bounds,
            steps,
            inits,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(*iv_types)

    @property
    def body(self):
        return self.regions[0].blocks[0]

    @property
    def induction_variables(self):
        return self.body.arguments


parallel = _parfor_cm(ParallelOp)


class ReduceOp(ReduceOp):
    def __init__(
        self,
        operands: Sequence[Union[Operation, OpView, Sequence[Value]]],
        num_reductions: int,
        *,
        loc=None,
        ip=None,
    ):
        operands = _get_op_results_or_values(operands)
        super().__init__(operands, num_reductions, loc=loc, ip=ip)
        for i in range(num_reductions):
            self.regions[i].blocks.append(operands[i].type, operands[i].type)


def reduce_(*operands, num_reductions=1, loc=None, ip=None):
    return ReduceOp(operands, num_reductions, loc=loc, ip=ip)


reduce = region_op(reduce_, terminator=lambda xs: reduce_return(*xs))


@region_adder(terminator=lambda xs: reduce_return(*xs))
def another_reduce(reduce_op):
    for r in reduce_op.regions:
        if len(r.blocks[0].operations) == 0:
            return r


@region_op
def in_parallel():
    return InParallelOp()


def parallel_insert_slice(
    source: Union[Operation, OpView, Value],
    dest: Union[Operation, OpView, Value],
    offsets: Optional[Sequence[Union[Operation, OpView, Value, int]]] = None,
    sizes: Optional[Sequence[Union[Operation, OpView, Value, int]]] = None,
    strides: Optional[Sequence[Union[Operation, OpView, Value, int]]] = None,
):
    from . import tensor

    @in_parallel
    def foo():
        tensor.parallel_insert_slice(
            source,
            dest,
            offsets,
            sizes,
            strides,
        )
