# RUN: %PYTHON %s | FileCheck %s
# Note that this is separate from ir_attributes.py since it depends on numpy,
# and we may want to disable if not available.

import gc
import mlir
import numpy as np

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert mlir.ir.Context._get_live_count() == 0

################################################################################
# Tests of the array/buffer .get() factory method on unsupported dtype.
################################################################################

def testGetDenseElementsUnsupported():
  ctx = mlir.ir.Context()
  array = np.array([["hello", "goodbye"]])
  try:
    attr = mlir.ir.DenseElementsAttr.get(ctx, array)
  except ValueError as e:
    # CHECK: unimplemented array format conversion from format:
    print(e)

run(testGetDenseElementsUnsupported)

################################################################################
# Splats.
################################################################################

# CHECK-LABEL: TEST: testGetDenseElementsSplatInt
def testGetDenseElementsSplatInt():
  ctx = mlir.ir.Context()
  loc = ctx.get_unknown_location()
  t = mlir.ir.IntegerType.get_signless(ctx, 32)
  element = mlir.ir.IntegerAttr.get(t, 555)
  shaped_type = mlir.ir.RankedTensorType.get((2, 3, 4), t, loc)
  attr = mlir.ir.DenseElementsAttr.get_splat(shaped_type, element)
  # CHECK: dense<555> : tensor<2x3x4xi32>
  print(attr)
  # CHECK: is_splat: True
  print("is_splat:", attr.is_splat)

run(testGetDenseElementsSplatInt)


# CHECK-LABEL: TEST: testGetDenseElementsSplatFloat
def testGetDenseElementsSplatFloat():
  ctx = mlir.ir.Context()
  loc = ctx.get_unknown_location()
  t = mlir.ir.F32Type.get(ctx)
  element = mlir.ir.FloatAttr.get(t, 1.2, loc)
  shaped_type = mlir.ir.RankedTensorType.get((2, 3, 4), t, loc)
  attr = mlir.ir.DenseElementsAttr.get_splat(shaped_type, element)
  # CHECK: dense<1.200000e+00> : tensor<2x3x4xf32>
  print(attr)

run(testGetDenseElementsSplatFloat)


# CHECK-LABEL: TEST: testGetDenseElementsSplatErrors
def testGetDenseElementsSplatErrors():
  ctx = mlir.ir.Context()
  loc = ctx.get_unknown_location()
  t = mlir.ir.F32Type.get(ctx)
  other_t = mlir.ir.F64Type.get(ctx)
  element = mlir.ir.FloatAttr.get(t, 1.2, loc)
  other_element = mlir.ir.FloatAttr.get(other_t, 1.2, loc)
  shaped_type = mlir.ir.RankedTensorType.get((2, 3, 4), t, loc)
  dynamic_shaped_type = mlir.ir.UnrankedTensorType.get(t, loc)
  non_shaped_type = t

  try:
    attr = mlir.ir.DenseElementsAttr.get_splat(non_shaped_type, element)
  except ValueError as e:
    # CHECK: Expected a static ShapedType for the shaped_type parameter: Type(f32)
    print(e)

  try:
    attr = mlir.ir.DenseElementsAttr.get_splat(dynamic_shaped_type, element)
  except ValueError as e:
    # CHECK: Expected a static ShapedType for the shaped_type parameter: Type(tensor<*xf32>)
    print(e)

  try:
    attr = mlir.ir.DenseElementsAttr.get_splat(shaped_type, other_element)
  except ValueError as e:
    # CHECK: Shaped element type and attribute type must be equal: shaped=Type(tensor<2x3x4xf32>), element=Attribute(1.200000e+00 : f64)
    print(e)

run(testGetDenseElementsSplatErrors)


################################################################################
# Tests of the array/buffer .get() factory method, in all of its permutations.
################################################################################

### float and double arrays.

# CHECK-LABEL: TEST: testGetDenseElementsF32
def testGetDenseElementsF32():
  ctx = mlir.ir.Context()
  array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array)
  # CHECK: dense<{{\[}}[1.100000e+00, 2.200000e+00, 3.300000e+00], [4.400000e+00, 5.500000e+00, 6.600000e+00]]> : tensor<2x3xf32>
  print(attr)
  # CHECK: is_splat: False
  print("is_splat:", attr.is_splat)

run(testGetDenseElementsF32)


# CHECK-LABEL: TEST: testGetDenseElementsF64
def testGetDenseElementsF64():
  ctx = mlir.ir.Context()
  array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array)
  # CHECK: dense<{{\[}}[1.100000e+00, 2.200000e+00, 3.300000e+00], [4.400000e+00, 5.500000e+00, 6.600000e+00]]> : tensor<2x3xf64>
  print(attr)

run(testGetDenseElementsF64)


### 32 bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI32Signless
def testGetDenseElementsI32Signless():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  print(attr)

run(testGetDenseElementsI32Signless)


# CHECK-LABEL: TEST: testGetDenseElementsUI32Signless
def testGetDenseElementsUI32Signless():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  print(attr)

run(testGetDenseElementsUI32Signless)

# CHECK-LABEL: TEST: testGetDenseElementsI32
def testGetDenseElementsI32():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array, signless=False)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi32>
  print(attr)

run(testGetDenseElementsI32)


# CHECK-LABEL: TEST: testGetDenseElementsUI32
def testGetDenseElementsUI32():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array, signless=False)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui32>
  print(attr)

run(testGetDenseElementsUI32)


## 64bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI64Signless
def testGetDenseElementsI64Signless():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
  print(attr)

run(testGetDenseElementsI64Signless)


# CHECK-LABEL: TEST: testGetDenseElementsUI64Signless
def testGetDenseElementsUI64Signless():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
  print(attr)

run(testGetDenseElementsUI64Signless)

# CHECK-LABEL: TEST: testGetDenseElementsI64
def testGetDenseElementsI64():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array, signless=False)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi64>
  print(attr)

run(testGetDenseElementsI64)


# CHECK-LABEL: TEST: testGetDenseElementsUI64
def testGetDenseElementsUI64():
  ctx = mlir.ir.Context()
  array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
  attr = mlir.ir.DenseElementsAttr.get(ctx, array, signless=False)
  # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui64>
  print(attr)

run(testGetDenseElementsUI64)

