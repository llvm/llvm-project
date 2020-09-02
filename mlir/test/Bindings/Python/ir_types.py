# RUN: %PYTHON %s | FileCheck %s

import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()


# CHECK-LABEL: TEST: testParsePrint
def testParsePrint():
  ctx = mlir.ir.Context()
  t = ctx.parse_type("i32")
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

# CHECK-LABEL: TEST: testVectorType
def testVectorType():
  ctx = mlir.ir.Context()
  f32 = mlir.ir.F32Type(ctx)
  shape = [2, 3]
  # CHECK: vector type: vector<2x3xf32>
  print("vector type:", mlir.ir.VectorType.get_vector(shape, f32))

  index = mlir.ir.IndexType(ctx)
  try:
    vector_invalid = mlir.ir.VectorType.get_vector(shape, index)
  except ValueError as e:
    # CHECK: invalid 'Type(index)' and expected floating point or integer type.
    print(e)
  else:
    print("Exception not produced")

run(testVectorType)

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
