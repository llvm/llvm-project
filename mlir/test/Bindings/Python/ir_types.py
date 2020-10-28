# RUN: %PYTHON %s | FileCheck %s

import gc
import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert mlir.ir.Context._get_live_count() == 0


# CHECK-LABEL: TEST: testParsePrint
def testParsePrint():
  ctx = mlir.ir.Context()
  t = ctx.parse_type("i32")
  ctx = None
  gc.collect()
  # CHECK: i32
  print(str(t))
  # CHECK: Type(i32)
  print(repr(t))

run(testParsePrint)


# CHECK-LABEL: TEST: testParseError
# TODO: Hook the diagnostic manager to capture a more meaningful error
# message.
def testParseError():
  ctx = mlir.ir.Context()
  try:
    t = ctx.parse_type("BAD_TYPE_DOES_NOT_EXIST")
  except ValueError as e:
    # CHECK: Unable to parse type: 'BAD_TYPE_DOES_NOT_EXIST'
    print("testParseError:", e)
  else:
    print("Exception not produced")

run(testParseError)


# CHECK-LABEL: TEST: testTypeEq
def testTypeEq():
  ctx = mlir.ir.Context()
  t1 = ctx.parse_type("i32")
  t2 = ctx.parse_type("f32")
  t3 = ctx.parse_type("i32")
  # CHECK: t1 == t1: True
  print("t1 == t1:", t1 == t1)
  # CHECK: t1 == t2: False
  print("t1 == t2:", t1 == t2)
  # CHECK: t1 == t3: True
  print("t1 == t3:", t1 == t3)
  # CHECK: t1 == None: False
  print("t1 == None:", t1 == None)

run(testTypeEq)


# CHECK-LABEL: TEST: testTypeEqDoesNotRaise
def testTypeEqDoesNotRaise():
  ctx = mlir.ir.Context()
  t1 = ctx.parse_type("i32")
  not_a_type = "foo"
  # CHECK: False
  print(t1 == not_a_type)
  # CHECK: False
  print(t1 == None)
  # CHECK: True
  print(t1 != None)

run(testTypeEqDoesNotRaise)


# CHECK-LABEL: TEST: testStandardTypeCasts
def testStandardTypeCasts():
  ctx = mlir.ir.Context()
  t1 = ctx.parse_type("i32")
  tint = mlir.ir.IntegerType(t1)
  tself = mlir.ir.IntegerType(tint)
  # CHECK: Type(i32)
  print(repr(tint))
  try:
    tillegal = mlir.ir.IntegerType(ctx.parse_type("f32"))
  except ValueError as e:
    # CHECK: ValueError: Cannot cast type to IntegerType (from Type(f32))
    print("ValueError:", e)
  else:
    print("Exception not produced")

run(testStandardTypeCasts)


# CHECK-LABEL: TEST: testIntegerType
def testIntegerType():
  ctx = mlir.ir.Context()
  i32 = mlir.ir.IntegerType(ctx.parse_type("i32"))
  # CHECK: i32 width: 32
  print("i32 width:", i32.width)
  # CHECK: i32 signless: True
  print("i32 signless:", i32.is_signless)
  # CHECK: i32 signed: False
  print("i32 signed:", i32.is_signed)
  # CHECK: i32 unsigned: False
  print("i32 unsigned:", i32.is_unsigned)

  s32 = mlir.ir.IntegerType(ctx.parse_type("si32"))
  # CHECK: s32 signless: False
  print("s32 signless:", s32.is_signless)
  # CHECK: s32 signed: True
  print("s32 signed:", s32.is_signed)
  # CHECK: s32 unsigned: False
  print("s32 unsigned:", s32.is_unsigned)

  u32 = mlir.ir.IntegerType(ctx.parse_type("ui32"))
  # CHECK: u32 signless: False
  print("u32 signless:", u32.is_signless)
  # CHECK: u32 signed: False
  print("u32 signed:", u32.is_signed)
  # CHECK: u32 unsigned: True
  print("u32 unsigned:", u32.is_unsigned)

  # CHECK: signless: i16
  print("signless:", mlir.ir.IntegerType.get_signless(ctx, 16))
  # CHECK: signed: si8
  print("signed:", mlir.ir.IntegerType.get_signed(ctx, 8))
  # CHECK: unsigned: ui64
  print("unsigned:", mlir.ir.IntegerType.get_unsigned(ctx, 64))

run(testIntegerType)

# CHECK-LABEL: TEST: testIndexType
def testIndexType():
  ctx = mlir.ir.Context()
  # CHECK: index type: index
  print("index type:", mlir.ir.IndexType(ctx))

run(testIndexType)

# CHECK-LABEL: TEST: testFloatType
def testFloatType():
  ctx = mlir.ir.Context()
  # CHECK: float: bf16
  print("float:", mlir.ir.BF16Type(ctx))
  # CHECK: float: f16
  print("float:", mlir.ir.F16Type(ctx))
  # CHECK: float: f32
  print("float:", mlir.ir.F32Type(ctx))
  # CHECK: float: f64
  print("float:", mlir.ir.F64Type(ctx))

run(testFloatType)

# CHECK-LABEL: TEST: testNoneType
def testNoneType():
  ctx = mlir.ir.Context()
  # CHECK: none type: none
  print("none type:", mlir.ir.NoneType(ctx))

run(testNoneType)

# CHECK-LABEL: TEST: testComplexType
def testComplexType():
  ctx = mlir.ir.Context()
  complex_i32 = mlir.ir.ComplexType(ctx.parse_type("complex<i32>"))
  # CHECK: complex type element: i32
  print("complex type element:", complex_i32.element_type)

  f32 = mlir.ir.F32Type(ctx)
  # CHECK: complex type: complex<f32>
  print("complex type:", mlir.ir.ComplexType.get_complex(f32))

  index = mlir.ir.IndexType(ctx)
  try:
    complex_invalid = mlir.ir.ComplexType.get_complex(index)
  except ValueError as e:
    # CHECK: invalid 'Type(index)' and expected floating point or integer type.
    print(e)
  else:
    print("Exception not produced")

run(testComplexType)

# CHECK-LABEL: TEST: testConcreteShapedType
# Shaped type is not a kind of standard types, it is the base class for
# vectors, memrefs and tensors, so this test case uses an instance of vector
# to test the shaped type. The class hierarchy is preserved on the python side.
def testConcreteShapedType():
  ctx = mlir.ir.Context()
  vector = mlir.ir.VectorType(ctx.parse_type("vector<2x3xf32>"))
  # CHECK: element type: f32
  print("element type:", vector.element_type)
  # CHECK: whether the given shaped type is ranked: True
  print("whether the given shaped type is ranked:", vector.has_rank)
  # CHECK: rank: 2
  print("rank:", vector.rank)
  # CHECK: whether the shaped type has a static shape: True
  print("whether the shaped type has a static shape:", vector.has_static_shape)
  # CHECK: whether the dim-th dimension is dynamic: False
  print("whether the dim-th dimension is dynamic:", vector.is_dynamic_dim(0))
  # CHECK: dim size: 3
  print("dim size:", vector.get_dim_size(1))
  # CHECK: is_dynamic_size: False
  print("is_dynamic_size:", vector.is_dynamic_size(3))
  # CHECK: is_dynamic_stride_or_offset: False
  print("is_dynamic_stride_or_offset:", vector.is_dynamic_stride_or_offset(1))
  # CHECK: isinstance(ShapedType): True
  print("isinstance(ShapedType):", isinstance(vector, mlir.ir.ShapedType))

run(testConcreteShapedType)

# CHECK-LABEL: TEST: testAbstractShapedType
# Tests that ShapedType operates as an abstract base class of a concrete
# shaped type (using vector as an example).
def testAbstractShapedType():
  ctx = mlir.ir.Context()
  vector = mlir.ir.ShapedType(ctx.parse_type("vector<2x3xf32>"))
  # CHECK: element type: f32
  print("element type:", vector.element_type)

run(testAbstractShapedType)

# CHECK-LABEL: TEST: testVectorType
def testVectorType():
  ctx = mlir.ir.Context()
  f32 = mlir.ir.F32Type(ctx)
  shape = [2, 3]
  loc = ctx.get_unknown_location()
  # CHECK: vector type: vector<2x3xf32>
  print("vector type:", mlir.ir.VectorType.get_vector(shape, f32, loc))

  none = mlir.ir.NoneType(ctx)
  try:
    vector_invalid = mlir.ir.VectorType.get_vector(shape, none, loc)
  except ValueError as e:
    # CHECK: invalid 'Type(none)' and expected floating point or integer type.
    print(e)
  else:
    print("Exception not produced")

run(testVectorType)

# CHECK-LABEL: TEST: testRankedTensorType
def testRankedTensorType():
  ctx = mlir.ir.Context()
  f32 = mlir.ir.F32Type(ctx)
  shape = [2, 3]
  loc = ctx.get_unknown_location()
  # CHECK: ranked tensor type: tensor<2x3xf32>
  print("ranked tensor type:",
        mlir.ir.RankedTensorType.get_ranked_tensor(shape, f32, loc))

  none = mlir.ir.NoneType(ctx)
  try:
    tensor_invalid = mlir.ir.RankedTensorType.get_ranked_tensor(shape, none,
                                                                loc)
  except ValueError as e:
    # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
    # CHECK: or complex type.
    print(e)
  else:
    print("Exception not produced")

run(testRankedTensorType)

# CHECK-LABEL: TEST: testUnrankedTensorType
def testUnrankedTensorType():
  ctx = mlir.ir.Context()
  f32 = mlir.ir.F32Type(ctx)
  loc = ctx.get_unknown_location()
  unranked_tensor = mlir.ir.UnrankedTensorType.get_unranked_tensor(f32, loc)
  # CHECK: unranked tensor type: tensor<*xf32>
  print("unranked tensor type:", unranked_tensor)
  try:
    invalid_rank = unranked_tensor.rank
  except ValueError as e:
    # CHECK: calling this method requires that the type has a rank.
    print(e)
  else:
    print("Exception not produced")
  try:
    invalid_is_dynamic_dim = unranked_tensor.is_dynamic_dim(0)
  except ValueError as e:
    # CHECK: calling this method requires that the type has a rank.
    print(e)
  else:
    print("Exception not produced")
  try:
    invalid_get_dim_size = unranked_tensor.get_dim_size(1)
  except ValueError as e:
    # CHECK: calling this method requires that the type has a rank.
    print(e)
  else:
    print("Exception not produced")

  none = mlir.ir.NoneType(ctx)
  try:
    tensor_invalid = mlir.ir.UnrankedTensorType.get_unranked_tensor(none, loc)
  except ValueError as e:
    # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
    # CHECK: or complex type.
    print(e)
  else:
    print("Exception not produced")

run(testUnrankedTensorType)

# CHECK-LABEL: TEST: testMemRefType
def testMemRefType():
  ctx = mlir.ir.Context()
  f32 = mlir.ir.F32Type(ctx)
  shape = [2, 3]
  loc = ctx.get_unknown_location()
  memref = mlir.ir.MemRefType.get_contiguous_memref(f32, shape, 2, loc)
  # CHECK: memref type: memref<2x3xf32, 2>
  print("memref type:", memref)
  # CHECK: number of affine layout maps: 0
  print("number of affine layout maps:", memref.num_affine_maps)
  # CHECK: memory space: 2
  print("memory space:", memref.memory_space)

  none = mlir.ir.NoneType(ctx)
  try:
    memref_invalid = mlir.ir.MemRefType.get_contiguous_memref(none, shape, 2,
                                                              loc)
  except ValueError as e:
    # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
    # CHECK: or complex type.
    print(e)
  else:
    print("Exception not produced")

run(testMemRefType)

# CHECK-LABEL: TEST: testUnrankedMemRefType
def testUnrankedMemRefType():
  ctx = mlir.ir.Context()
  f32 = mlir.ir.F32Type(ctx)
  loc = ctx.get_unknown_location()
  unranked_memref = mlir.ir.UnrankedMemRefType.get_unranked_memref(f32, 2, loc)
  # CHECK: unranked memref type: memref<*xf32, 2>
  print("unranked memref type:", unranked_memref)
  try:
    invalid_rank = unranked_memref.rank
  except ValueError as e:
    # CHECK: calling this method requires that the type has a rank.
    print(e)
  else:
    print("Exception not produced")
  try:
    invalid_is_dynamic_dim = unranked_memref.is_dynamic_dim(0)
  except ValueError as e:
    # CHECK: calling this method requires that the type has a rank.
    print(e)
  else:
    print("Exception not produced")
  try:
    invalid_get_dim_size = unranked_memref.get_dim_size(1)
  except ValueError as e:
    # CHECK: calling this method requires that the type has a rank.
    print(e)
  else:
    print("Exception not produced")

  none = mlir.ir.NoneType(ctx)
  try:
    memref_invalid = mlir.ir.UnrankedMemRefType.get_unranked_memref(none, 2,
                                                                    loc)
  except ValueError as e:
    # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
    # CHECK: or complex type.
    print(e)
  else:
    print("Exception not produced")

run(testUnrankedMemRefType)

# CHECK-LABEL: TEST: testTupleType
def testTupleType():
  ctx = mlir.ir.Context()
  i32 = mlir.ir.IntegerType(ctx.parse_type("i32"))
  f32 = mlir.ir.F32Type(ctx)
  vector = mlir.ir.VectorType(ctx.parse_type("vector<2x3xf32>"))
  l = [i32, f32, vector]
  tuple_type = mlir.ir.TupleType.get_tuple(ctx, l)
  # CHECK: tuple type: tuple<i32, f32, vector<2x3xf32>>
  print("tuple type:", tuple_type)
  # CHECK: number of types: 3
  print("number of types:", tuple_type.num_types)
  # CHECK: pos-th type in the tuple type: f32
  print("pos-th type in the tuple type:", tuple_type.get_type(1))

run(testTupleType)


# CHECK-LABEL: TEST: testFunctionType
def testFunctionType():
  ctx = mlir.ir.Context()
  input_types = [mlir.ir.IntegerType.get_signless(ctx, 32),
                 mlir.ir.IntegerType.get_signless(ctx, 16)]
  result_types = [mlir.ir.IndexType(ctx)]
  func = mlir.ir.FunctionType.get(ctx, input_types, result_types)
  # CHECK: INPUTS: [Type(i32), Type(i16)]
  print("INPUTS:", func.inputs)
  # CHECK: RESULTS: [Type(index)]
  print("RESULTS:", func.results)


run(testFunctionType)
