# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import std


def constructAndPrintInModule(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      f()
    print(module)
  return f

# CHECK-LABEL: TEST: testConstantOp

@constructAndPrintInModule
def testConstantOp():
  c1 = std.ConstantOp(IntegerType.get_signless(32), 42)
  c2 = std.ConstantOp(IntegerType.get_signless(64), 100)
  c3 = std.ConstantOp(F32Type.get(), 3.14)
  c4 = std.ConstantOp(F64Type.get(), 1.23)
  # CHECK: 42
  print(c1.literal_value)

  # CHECK: 100
  print(c2.literal_value)

  # CHECK: 3.140000104904175
  print(c3.literal_value)

  # CHECK: 1.23
  print(c4.literal_value)

# CHECK: = constant 42 : i32
# CHECK: = constant 100 : i64
# CHECK: = constant 3.140000e+00 : f32
# CHECK: = constant 1.230000e+00 : f64

# CHECK-LABEL: TEST: testVectorConstantOp
@constructAndPrintInModule
def testVectorConstantOp():
  int_type = IntegerType.get_signless(32)
  vec_type = VectorType.get([2, 2], int_type)
  c1 = std.ConstantOp(vec_type,
                      DenseElementsAttr.get_splat(vec_type, IntegerAttr.get(int_type, 42)))
  try:
    print(c1.literal_value)
  except ValueError as e:
    assert "only integer and float constants have literal values" in str(e)
  else:
    assert False

# CHECK: = constant dense<42> : vector<2x2xi32>

# CHECK-LABEL: TEST: testConstantIndexOp
@constructAndPrintInModule
def testConstantIndexOp():
  c1 = std.ConstantOp.create_index(10)
  # CHECK: 10
  print(c1.literal_value)

# CHECK: = constant 10 : index
