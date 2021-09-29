# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import scf
from mlir.dialects import builtin


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testSimpleLoop
@run
def testSimpleLoop():
  with Context(), Location.unknown():
    module = Module.create()
    index_type = IndexType.get()
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(index_type, index_type, index_type)
      def simple_loop(lb, ub, step):
        loop = scf.ForOp(lb, ub, step, [lb, lb])
        with InsertionPoint(loop.body):
          scf.YieldOp(loop.inner_iter_args)
        return

  # CHECK: func @simple_loop(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
  # CHECK: scf.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
  # CHECK: iter_args(%[[I1:.*]] = %[[ARG0]], %[[I2:.*]] = %[[ARG0]])
  # CHECK: scf.yield %[[I1]], %[[I2]]
  print(module)


# CHECK-LABEL: TEST: testInductionVar
@run
def testInductionVar():
  with Context(), Location.unknown():
    module = Module.create()
    index_type = IndexType.get()
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(index_type, index_type, index_type)
      def induction_var(lb, ub, step):
        loop = scf.ForOp(lb, ub, step, [lb])
        with InsertionPoint(loop.body):
          scf.YieldOp([loop.induction_variable])
        return

  # CHECK: func @induction_var(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
  # CHECK: scf.for %[[IV:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
  # CHECK: scf.yield %[[IV]]
  print(module)
