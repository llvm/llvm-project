# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.builtin as builtin
import mlir.dialects.vector as vector

def run(f):
  print("\nTEST:", f.__name__)
  f()

# CHECK-LABEL: TEST: testPrintOp
@run
def testPrintOp():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      @builtin.FuncOp.from_py_func(VectorType.get((12, 5), F32Type.get()))
      def print_vector(arg):
        return vector.PrintOp(arg)

    # CHECK-LABEL: func @print_vector(
    # CHECK-SAME:                     %[[ARG:.*]]: vector<12x5xf32>) {
    #       CHECK:   vector.print %[[ARG]] : vector<12x5xf32>
    #       CHECK:   return
    #       CHECK: }
    print(module)
