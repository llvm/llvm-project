# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.arith as arith

def run(f):
  print("\nTEST:", f.__name__)
  f()

# CHECK-LABEL: TEST: testConstantOp
@run
def testConstantOps():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      arith.ConstantOp(value=42.42, result=F32Type.get())
    # CHECK:         %cst = arith.constant 4.242000e+01 : f32
    print(module)
