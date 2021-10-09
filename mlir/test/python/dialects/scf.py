# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import scf
from mlir.dialects import std
from mlir.dialects import builtin


def constructAndPrintInModule(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      f()
    print(module)
  return f


# CHECK-LABEL: TEST: testSimpleLoop
@constructAndPrintInModule
def testSimpleLoop():
  index_type = IndexType.get()

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


# CHECK-LABEL: TEST: testInductionVar
@constructAndPrintInModule
def testInductionVar():
  index_type = IndexType.get()

  @builtin.FuncOp.from_py_func(index_type, index_type, index_type)
  def induction_var(lb, ub, step):
    loop = scf.ForOp(lb, ub, step, [lb])
    with InsertionPoint(loop.body):
      scf.YieldOp([loop.induction_variable])
    return


# CHECK: func @induction_var(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
# CHECK: scf.for %[[IV:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
# CHECK: scf.yield %[[IV]]


@constructAndPrintInModule
def testOpsAsArguments():
  index_type = IndexType.get()
  callee = builtin.FuncOp(
      "callee", ([], [index_type, index_type]), visibility="private")
  func = builtin.FuncOp("ops_as_arguments", ([], []))
  with InsertionPoint(func.add_entry_block()):
    lb = std.ConstantOp.create_index(0)
    ub = std.ConstantOp.create_index(42)
    step = std.ConstantOp.create_index(2)
    iter_args = std.CallOp(callee, [])
    loop = scf.ForOp(lb, ub, step, iter_args)
    with InsertionPoint(loop.body):
      scf.YieldOp(loop.inner_iter_args)
    std.ReturnOp([])


# CHECK-LABEL: TEST: testOpsAsArguments
# CHECK: func private @callee() -> (index, index)
# CHECK: func @ops_as_arguments() {
# CHECK:   %[[LB:.*]] = constant 0
# CHECK:   %[[UB:.*]] = constant 42
# CHECK:   %[[STEP:.*]] = constant 2
# CHECK:   %[[ARGS:.*]]:2 = call @callee()
# CHECK:   scf.for %arg0 = %c0 to %c42 step %c2
# CHECK:   iter_args(%{{.*}} = %[[ARGS]]#0, %{{.*}} = %[[ARGS]]#1)
# CHECK:     scf.yield %{{.*}}, %{{.*}}
# CHECK:   return
