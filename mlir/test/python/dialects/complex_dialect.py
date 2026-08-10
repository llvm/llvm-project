# RUN: %PYTHON %s | FileCheck %s

# Naming this file with a `_dialect` suffix to avoid a naming conflict with
# python package's math module (coming in from random.py).

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.complex as mlir_complex


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testComplexOps
@run
def testComplexOps():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(ComplexType.get(F32Type.get()))
            def emit_add(arg):
                return mlir_complex.AddOp(arg, arg)

        # CHECK-LABEL: func @emit_add(
        # CHECK-SAME:                  %[[ARG:.*]]: complex<f32>) -> complex<f32> {
        # CHECK:         %[[RES:.*]] = complex.add %[[ARG]], %[[ARG]] : complex<f32>
        # CHECK:         return %[[RES]] : complex<f32>
        # CHECK:       }
        print(module)
