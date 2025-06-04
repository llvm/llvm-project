# RUN: %PYTHON %s | FileCheck %s

from mlir.dialects import smt, arith
from mlir.ir import Context, Location, Module, InsertionPoint, F32Type


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f(module)
        print(module)
        assert module.operation.verify()


# CHECK-LABEL: TEST: test_smoke
@run
def test_smoke(_module):
    true = smt.constant(True)
    false = smt.constant(False)
    # CHECK: smt.constant true
    # CHECK: smt.constant false


# CHECK-LABEL: TEST: test_types
@run
def test_types(_module):
    bool_t = smt.bool_t()
    bitvector_t = smt.bv_t(5)
    # CHECK: !smt.bool
    print(bool_t)
    # CHECK: !smt.bv<5>
    print(bitvector_t)


# CHECK-LABEL: TEST: test_solver_op
@run
def test_solver_op(_module):
    @smt.solver
    def foo1():
        true = smt.constant(True)
        false = smt.constant(False)

    # CHECK: smt.solver() : () -> () {
    # CHECK:   %true = smt.constant true
    # CHECK:   %false = smt.constant false
    # CHECK: }

    f32 = F32Type.get()

    @smt.solver(results=[f32])
    def foo2():
        return arith.ConstantOp(f32, 1.0)

    # CHECK: %{{.*}} = smt.solver() : () -> f32 {
    # CHECK:   %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:   smt.yield %[[CST1]] : f32
    # CHECK: }

    two = arith.ConstantOp(f32, 2.0)
    # CHECK: %[[CST2:.*]] = arith.constant 2.000000e+00 : f32
    print(two)

    @smt.solver(inputs=[two], results=[f32])
    def foo3(z: f32):
        return z

    # CHECK: %{{.*}} = smt.solver(%[[CST2]]) : (f32) -> f32 {
    # CHECK: ^bb0(%[[ARG0:.*]]: f32):
    # CHECK:   smt.yield %[[ARG0]] : f32
    # CHECK: }


# CHECK-LABEL: TEST: test_export_smtlib
@run
def test_export_smtlib(module):
    @smt.solver
    def foo1():
        true = smt.constant(True)
        smt.assert_(true)

    query = smt.export_smtlib(module.operation)
    # CHECK: ; solver scope 0
    # CHECK: (assert true)
    # CHECK: (reset)
    print(query)
