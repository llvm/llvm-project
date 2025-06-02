#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._smt_ops_gen import *

from .._mlir_libs._mlirDialectsSMT import *
from ..extras.meta import region_op


def bool_t():
    return BoolType.get()


def bv_t(width):
    return BitVectorType.get(width)


def _solver(
    inputs=None,
    results=None,
    loc=None,
    ip=None,
):
    if inputs is None:
        inputs = []
    if results is None:
        results = []

    return SolverOp(results, inputs, loc=loc, ip=ip)


solver = region_op(_solver, terminator=YieldOp)
