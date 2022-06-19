# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0
  return f


# CHECK-LABEL: TEST: testParsePrint
@run
def testParsePrint():
  ctx = Context()
  t = Type.parse("i32", ctx)
  assert t.context is ctx
  ctx = None
  gc.collect()
  # CHECK: i32
  print(str(t))
  # CHECK: Type(i32)
  print(repr(t))


# CHECK-LABEL: TEST: testParseError
# TODO: Hook the diagnostic manager to capture a more meaningful error
# message.
@run
def testParseError():
  ctx = Context()
  try:
    t = Type.parse("BAD_TYPE_DOES_NOT_EXIST", ctx)
  except ValueError as e:
    # CHECK: Unable to parse type: 'BAD_TYPE_DOES_NOT_EXIST'
    print("testParseError:", e)
  else:
    print("Exception not produced")


# CHECK-LABEL: TEST: testTypeEq
@run
def testTypeEq():
  ctx = Context()
  t1 = Type.parse("i32", ctx)
  t2 = Type.parse("f32", ctx)
  t3 = Type.parse("i32", ctx)
  # CHECK: t1 == t1: True
  print("t1 == t1:", t1 == t1)
  # CHECK: t1 == t2: False
  print("t1 == t2:", t1 == t2)
  # CHECK: t1 == t3: True
  print("t1 == t3:", t1 == t3)
  # CHECK: t1 == None: False
  print("t1 == None:", t1 == None)


# CHECK-LABEL: TEST: testTypeHash
@run
def testTypeHash():
  ctx = Context()
  t1 = Type.parse("i32", ctx)
  t2 = Type.parse("f32", ctx)
  t3 = Type.parse("i32", ctx)

  # CHECK: hash(t1) == hash(t3): True
  print("hash(t1) == hash(t3):", t1.__hash__() == t3.__hash__())

  s = set()
  s.add(t1)
  s.add(t2)
  s.add(t3)
  # CHECK: len(s): 2
  print("len(s): ", len(s))

# CHECK-LABEL: TEST: testTypeCast
@run
def testTypeCast():
  ctx = Context()
  t1 = Type.parse("i32", ctx)
  t2 = Type(t1)
  # CHECK: t1 == t2: True
  print("t1 == t2:", t1 == t2)


# CHECK-LABEL: TEST: testTypeIsInstance
@run
def testTypeIsInstance():
  ctx = Context()
  t1 = Type.parse("i32", ctx)
  t2 = Type.parse("f32", ctx)
  # CHECK: True
  print(IntegerType.isinstance(t1))
  # CHECK: False
  print(F32Type.isinstance(t1))
  # CHECK: True
  print(F32Type.isinstance(t2))


# CHECK-LABEL: TEST: testTypeEqDoesNotRaise
@run
def testTypeEqDoesNotRaise():
  ctx = Context()
  t1 = Type.parse("i32", ctx)
  not_a_type = "foo"
  # CHECK: False
  print(t1 == not_a_type)
  # CHECK: False
  print(t1 == None)
  # CHECK: True
  print(t1 != None)


# CHECK-LABEL: TEST: testTypeCapsule
@run
def testTypeCapsule():
  with Context() as ctx:
    t1 = Type.parse("i32", ctx)
  # CHECK: mlir.ir.Type._CAPIPtr
  type_capsule = t1._CAPIPtr
  print(type_capsule)
  t2 = Type._CAPICreate(type_capsule)
  assert t2 == t1
  assert t2.context is ctx


# CHECK-LABEL: TEST: testStandardTypeCasts
@run
def testStandardTypeCasts():
  ctx = Context()
  t1 = Type.parse("i32", ctx)
  tint = IntegerType(t1)
  tself = IntegerType(tint)
  # CHECK: Type(i32)
  print(repr(tint))
  try:
    tillegal = IntegerType(Type.parse("f32", ctx))
  except ValueError as e:
    # CHECK: ValueError: Cannot cast type to IntegerType (from Type(f32))
    print("ValueError:", e)
  else:
    print("Exception not produced")


# CHECK-LABEL: TEST: testIntegerType
@run
def testIntegerType():
  with Context() as ctx:
    i32 = IntegerType(Type.parse("i32"))
    # CHECK: i32 width: 32
    print("i32 width:", i32.width)
    # CHECK: i32 signless: True
    print("i32 signless:", i32.is_signless)
    # CHECK: i32 signed: False
    print("i32 signed:", i32.is_signed)
    # CHECK: i32 unsigned: False
    print("i32 unsigned:", i32.is_unsigned)

    s32 = IntegerType(Type.parse("si32"))
    # CHECK: s32 signless: False
    print("s32 signless:", s32.is_signless)
    # CHECK: s32 signed: True
    print("s32 signed:", s32.is_signed)
    # CHECK: s32 unsigned: False
    print("s32 unsigned:", s32.is_unsigned)

    u32 = IntegerType(Type.parse("ui32"))
    # CHECK: u32 signless: False
    print("u32 signless:", u32.is_signless)
    # CHECK: u32 signed: False
    print("u32 signed:", u32.is_signed)
    # CHECK: u32 unsigned: True
    print("u32 unsigned:", u32.is_unsigned)

    # CHECK: signless: i16
    print("signless:", IntegerType.get_signless(16))
    # CHECK: signed: si8
    print("signed:", IntegerType.get_signed(8))
    # CHECK: unsigned: ui64
    print("unsigned:", IntegerType.get_unsigned(64))

# CHECK-LABEL: TEST: testIndexType
@run
def testIndexType():
  with Context() as ctx:
    # CHECK: index type: index
    print("index type:", IndexType.get())


# CHECK-LABEL: TEST: testFloatType
@run
def testFloatType():
  with Context():
    # CHECK: float: bf16
    print("float:", BF16Type.get())
    # CHECK: float: f16
    print("float:", F16Type.get())
    # CHECK: float: f32
    print("float:", F32Type.get())
    # CHECK: float: f64
    print("float:", F64Type.get())


# CHECK-LABEL: TEST: testNoneType
@run
def testNoneType():
  with Context():
    # CHECK: none type: none
    print("none type:", NoneType.get())


# CHECK-LABEL: TEST: testComplexType
@run
def testComplexType():
  with Context() as ctx:
    complex_i32 = ComplexType(Type.parse("complex<i32>"))
    # CHECK: complex type element: i32
    print("complex type element:", complex_i32.element_type)

    f32 = F32Type.get()
    # CHECK: complex type: complex<f32>
    print("complex type:", ComplexType.get(f32))

    index = IndexType.get()
    try:
      complex_invalid = ComplexType.get(index)
    except ValueError as e:
      # CHECK: invalid 'Type(index)' and expected floating point or integer type.
      print(e)
    else:
      print("Exception not produced")


# CHECK-LABEL: TEST: testConcreteShapedType
# Shaped type is not a kind of builtin types, it is the base class for vectors,
# memrefs and tensors, so this test case uses an instance of vector to test the
# shaped type. The class hierarchy is preserved on the python side.
@run
def testConcreteShapedType():
  with Context() as ctx:
    vector = VectorType(Type.parse("vector<2x3xf32>"))
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
    print("isinstance(ShapedType):", isinstance(vector, ShapedType))


# CHECK-LABEL: TEST: testAbstractShapedType
# Tests that ShapedType operates as an abstract base class of a concrete
# shaped type (using vector as an example).
@run
def testAbstractShapedType():
  ctx = Context()
  vector = ShapedType(Type.parse("vector<2x3xf32>", ctx))
  # CHECK: element type: f32
  print("element type:", vector.element_type)


# CHECK-LABEL: TEST: testVectorType
@run
def testVectorType():
  with Context(), Location.unknown():
    f32 = F32Type.get()
    shape = [2, 3]
    # CHECK: vector type: vector<2x3xf32>
    print("vector type:", VectorType.get(shape, f32))

    none = NoneType.get()
    try:
      vector_invalid = VectorType.get(shape, none)
    except ValueError as e:
      # CHECK: invalid 'Type(none)' and expected floating point or integer type.
      print(e)
    else:
      print("Exception not produced")


# CHECK-LABEL: TEST: testRankedTensorType
@run
def testRankedTensorType():
  with Context(), Location.unknown():
    f32 = F32Type.get()
    shape = [2, 3]
    loc = Location.unknown()
    # CHECK: ranked tensor type: tensor<2x3xf32>
    print("ranked tensor type:",
          RankedTensorType.get(shape, f32))

    none = NoneType.get()
    try:
      tensor_invalid = RankedTensorType.get(shape, none)
    except ValueError as e:
      # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
      # CHECK: or complex type.
      print(e)
    else:
      print("Exception not produced")

    # Encoding should be None.
    assert RankedTensorType.get(shape, f32).encoding is None

    tensor = RankedTensorType.get(shape, f32)
    assert tensor.shape == shape


# CHECK-LABEL: TEST: testUnrankedTensorType
@run
def testUnrankedTensorType():
  with Context(), Location.unknown():
    f32 = F32Type.get()
    loc = Location.unknown()
    unranked_tensor = UnrankedTensorType.get(f32)
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

    none = NoneType.get()
    try:
      tensor_invalid = UnrankedTensorType.get(none)
    except ValueError as e:
      # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
      # CHECK: or complex type.
      print(e)
    else:
      print("Exception not produced")


# CHECK-LABEL: TEST: testMemRefType
@run
def testMemRefType():
  with Context(), Location.unknown():
    f32 = F32Type.get()
    shape = [2, 3]
    loc = Location.unknown()
    memref = MemRefType.get(shape, f32, memory_space=Attribute.parse("2"))
    # CHECK: memref type: memref<2x3xf32, 2>
    print("memref type:", memref)
    # CHECK: memref layout: affine_map<(d0, d1) -> (d0, d1)>
    print("memref layout:", memref.layout)
    # CHECK: memref affine map: (d0, d1) -> (d0, d1)
    print("memref affine map:", memref.affine_map)
    # CHECK: memory space: 2
    print("memory space:", memref.memory_space)

    layout = AffineMapAttr.get(AffineMap.get_permutation([1, 0]))
    memref_layout = MemRefType.get(shape, f32, layout=layout)
    # CHECK: memref type: memref<2x3xf32, affine_map<(d0, d1) -> (d1, d0)>>
    print("memref type:", memref_layout)
    # CHECK: memref layout: affine_map<(d0, d1) -> (d1, d0)>
    print("memref layout:", memref_layout.layout)
    # CHECK: memref affine map: (d0, d1) -> (d1, d0)
    print("memref affine map:", memref_layout.affine_map)
    # CHECK: memory space: <<NULL ATTRIBUTE>>
    print("memory space:", memref_layout.memory_space)

    none = NoneType.get()
    try:
      memref_invalid = MemRefType.get(shape, none)
    except ValueError as e:
      # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
      # CHECK: or complex type.
      print(e)
    else:
      print("Exception not produced")

    assert memref.shape == shape


# CHECK-LABEL: TEST: testUnrankedMemRefType
@run
def testUnrankedMemRefType():
  with Context(), Location.unknown():
    f32 = F32Type.get()
    loc = Location.unknown()
    unranked_memref = UnrankedMemRefType.get(f32, Attribute.parse("2"))
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

    none = NoneType.get()
    try:
      memref_invalid = UnrankedMemRefType.get(none, Attribute.parse("2"))
    except ValueError as e:
      # CHECK: invalid 'Type(none)' and expected floating point, integer, vector
      # CHECK: or complex type.
      print(e)
    else:
      print("Exception not produced")


# CHECK-LABEL: TEST: testTupleType
@run
def testTupleType():
  with Context() as ctx:
    i32 = IntegerType(Type.parse("i32"))
    f32 = F32Type.get()
    vector = VectorType(Type.parse("vector<2x3xf32>"))
    l = [i32, f32, vector]
    tuple_type = TupleType.get_tuple(l)
    # CHECK: tuple type: tuple<i32, f32, vector<2x3xf32>>
    print("tuple type:", tuple_type)
    # CHECK: number of types: 3
    print("number of types:", tuple_type.num_types)
    # CHECK: pos-th type in the tuple type: f32
    print("pos-th type in the tuple type:", tuple_type.get_type(1))


# CHECK-LABEL: TEST: testFunctionType
@run
def testFunctionType():
  with Context() as ctx:
    input_types = [IntegerType.get_signless(32),
                  IntegerType.get_signless(16)]
    result_types = [IndexType.get()]
    func = FunctionType.get(input_types, result_types)
    # CHECK: INPUTS: [Type(i32), Type(i16)]
    print("INPUTS:", func.inputs)
    # CHECK: RESULTS: [Type(index)]
    print("RESULTS:", func.results)


# CHECK-LABEL: TEST: testOpaqueType
@run
def testOpaqueType():
  with Context() as ctx:
    ctx.allow_unregistered_dialects = True
    opaque = OpaqueType.get("dialect", "type")
    # CHECK: opaque type: !dialect.type
    print("opaque type:", opaque)
    # CHECK: dialect namespace: dialect
    print("dialect namespace:", opaque.dialect_namespace)
    # CHECK: data: type
    print("data:", opaque.data)
