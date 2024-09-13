# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *
from mlir.dialects import arith, tensor, func, memref
import mlir.extras.types as T


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
@run
def testParseError():
    ctx = Context()
    try:
        t = Type.parse("BAD_TYPE_DOES_NOT_EXIST", ctx)
    except MLIRError as e:
        # CHECK: testParseError: <
        # CHECK:   Unable to parse type:
        # CHECK:   error: "BAD_TYPE_DOES_NOT_EXIST":1:1: expected non-function type
        # CHECK: >
        print(f"testParseError: <{e}>")
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
    # CHECK: t1 is None: False
    print("t1 is None:", t1 is None)


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
    # CHECK: False
    print(FloatType.isinstance(t1))
    # CHECK: True
    print(F32Type.isinstance(t2))
    # CHECK: True
    print(FloatType.isinstance(t2))


# CHECK-LABEL: TEST: testFloatTypeSubclasses
@run
def testFloatTypeSubclasses():
    ctx = Context()
    # CHECK: True
    print(isinstance(Type.parse("f6E3M2FN", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f8E3M4", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f8E4M3", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f8E4M3FN", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f8E5M2", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f8E4M3FNUZ", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f8E4M3B11FNUZ", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f8E5M2FNUZ", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f16", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("bf16", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f32", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("tf32", ctx), FloatType))
    # CHECK: True
    print(isinstance(Type.parse("f64", ctx), FloatType))


# CHECK-LABEL: TEST: testTypeEqDoesNotRaise
@run
def testTypeEqDoesNotRaise():
    ctx = Context()
    t1 = Type.parse("i32", ctx)
    not_a_type = "foo"
    # CHECK: False
    print(t1 == not_a_type)
    # CHECK: False
    print(t1 is None)
    # CHECK: True
    print(t1 is not None)


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
        # CHECK: float: f6E3M2FN
        print("float:", Float6E3M2FNType.get())
        # CHECK: float: f8E3M4
        print("float:", Float8E3M4Type.get())
        # CHECK: float: f8E4M3
        print("float:", Float8E4M3Type.get())
        # CHECK: float: f8E4M3FN
        print("float:", Float8E4M3FNType.get())
        # CHECK: float: f8E5M2
        print("float:", Float8E5M2Type.get())
        # CHECK: float: f8E5M2FNUZ
        print("float:", Float8E5M2FNUZType.get())
        # CHECK: float: f8E4M3FNUZ
        print("float:", Float8E4M3FNUZType.get())
        # CHECK: float: f8E4M3B11FNUZ
        print("float:", Float8E4M3B11FNUZType.get())
        # CHECK: float: bf16
        print("float:", BF16Type.get())
        # CHECK: float: f16
        print("float:", F16Type.get())
        # CHECK: float: tf32
        print("float:", FloatTF32Type.get())
        # CHECK: float: f32
        print("float:", F32Type.get())
        # CHECK: float: f64
        f64 = F64Type.get()
        print("float:", f64)
        # CHECK: f64 width: 64
        print("f64 width:", f64.width)


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
            VectorType.get(shape, none)
        except MLIRError as e:
            # CHECK: Invalid type:
            # CHECK: error: unknown: failed to verify 'elementType': integer or index or floating-point
            print(e)
        else:
            print("Exception not produced")

        scalable_1 = VectorType.get(shape, f32, scalable=[False, True])
        scalable_2 = VectorType.get([2, 3, 4], f32, scalable=[True, False, True])
        assert scalable_1.scalable
        assert scalable_2.scalable
        assert scalable_1.scalable_dims == [False, True]
        assert scalable_2.scalable_dims == [True, False, True]
        # CHECK: scalable 1: vector<2x[3]xf32>
        print("scalable 1: ", scalable_1)
        # CHECK: scalable 2: vector<[2]x3x[4]xf32>
        print("scalable 2: ", scalable_2)

        scalable_3 = VectorType.get(shape, f32, scalable_dims=[1])
        scalable_4 = VectorType.get([2, 3, 4], f32, scalable_dims=[0, 2])
        assert scalable_3 == scalable_1
        assert scalable_4 == scalable_2

        try:
            VectorType.get(shape, f32, scalable=[False, True, True])
        except ValueError as e:
            # CHECK: Expected len(scalable) == len(shape).
            print(e)
        else:
            print("Exception not produced")

        try:
            VectorType.get(shape, f32, scalable=[False, True], scalable_dims=[1])
        except ValueError as e:
            # CHECK: kwargs are mutually exclusive.
            print(e)
        else:
            print("Exception not produced")

        try:
            VectorType.get(shape, f32, scalable_dims=[42])
        except ValueError as e:
            # CHECK: Scalable dimension index out of bounds.
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
        print("ranked tensor type:", RankedTensorType.get(shape, f32))

        none = NoneType.get()
        try:
            tensor_invalid = RankedTensorType.get(shape, none)
        except MLIRError as e:
            # CHECK: Invalid type:
            # CHECK: error: unknown: invalid tensor element type: 'none'
            print(e)
        else:
            print("Exception not produced")

        tensor = RankedTensorType.get(shape, f32, StringAttr.get("encoding"))
        assert tensor.shape == shape
        assert tensor.encoding.value == "encoding"

        # Encoding should be None.
        assert RankedTensorType.get(shape, f32).encoding is None


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
        except MLIRError as e:
            # CHECK: Invalid type:
            # CHECK: error: unknown: invalid tensor element type: 'none'
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
        memref_f32 = MemRefType.get(shape, f32, memory_space=Attribute.parse("2"))
        # CHECK: memref type: memref<2x3xf32, 2>
        print("memref type:", memref_f32)
        # CHECK: memref layout: AffineMapAttr(affine_map<(d0, d1) -> (d0, d1)>)
        print("memref layout:", repr(memref_f32.layout))
        # CHECK: memref affine map: (d0, d1) -> (d0, d1)
        print("memref affine map:", memref_f32.affine_map)
        # CHECK: memory space: IntegerAttr(2 : i64)
        print("memory space:", repr(memref_f32.memory_space))

        layout = AffineMapAttr.get(AffineMap.get_permutation([1, 0]))
        memref_layout = MemRefType.get(shape, f32, layout=layout)
        # CHECK: memref type: memref<2x3xf32, affine_map<(d0, d1) -> (d1, d0)>>
        print("memref type:", memref_layout)
        # CHECK: memref layout: affine_map<(d0, d1) -> (d1, d0)>
        print("memref layout:", memref_layout.layout)
        # CHECK: memref affine map: (d0, d1) -> (d1, d0)
        print("memref affine map:", memref_layout.affine_map)
        # CHECK: memory space: None
        print("memory space:", memref_layout.memory_space)

        none = NoneType.get()
        try:
            memref_invalid = MemRefType.get(shape, none)
        except MLIRError as e:
            # CHECK: Invalid type:
            # CHECK: error: unknown: invalid memref element type
            print(e)
        else:
            print("Exception not produced")

    assert memref_f32.shape == shape


# CHECK-LABEL: TEST: testUnrankedMemRefType
@run
def testUnrankedMemRefType():
    with Context(), Location.unknown():
        f32 = F32Type.get()
        loc = Location.unknown()
        unranked_memref = UnrankedMemRefType.get(f32, Attribute.parse("2"))
        # CHECK: unranked memref type: memref<*xf32, 2>
        print("unranked memref type:", unranked_memref)
        # CHECK: memory space: IntegerAttr(2 : i64)
        print("memory space:", repr(unranked_memref.memory_space))
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
        except MLIRError as e:
            # CHECK: Invalid type:
            # CHECK: error: unknown: invalid memref element type
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
        input_types = [IntegerType.get_signless(32), IntegerType.get_signless(16)]
        result_types = [IndexType.get()]
        func = FunctionType.get(input_types, result_types)
        # CHECK: INPUTS: [IntegerType(i32), IntegerType(i16)]
        print("INPUTS:", func.inputs)
        # CHECK: RESULTS: [IndexType(index)]
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


# CHECK-LABEL: TEST: testShapedTypeConstants
# Tests that ShapedType exposes magic value constants.
@run
def testShapedTypeConstants():
    # CHECK: <class 'int'>
    print(type(ShapedType.get_dynamic_size()))
    # CHECK: <class 'int'>
    print(type(ShapedType.get_dynamic_stride_or_offset()))


# CHECK-LABEL: TEST: testTypeIDs
@run
def testTypeIDs():
    with Context(), Location.unknown():
        f32 = F32Type.get()

        types = [
            (IntegerType, IntegerType.get_signless(16)),
            (IndexType, IndexType.get()),
            (Float6E3M2FNType, Float6E3M2FNType.get()),
            (Float8E3M4Type, Float8E3M4Type.get()),
            (Float8E4M3Type, Float8E4M3Type.get()),
            (Float8E4M3FNType, Float8E4M3FNType.get()),
            (Float8E5M2Type, Float8E5M2Type.get()),
            (Float8E4M3FNUZType, Float8E4M3FNUZType.get()),
            (Float8E4M3B11FNUZType, Float8E4M3B11FNUZType.get()),
            (Float8E5M2FNUZType, Float8E5M2FNUZType.get()),
            (BF16Type, BF16Type.get()),
            (F16Type, F16Type.get()),
            (F32Type, F32Type.get()),
            (F64Type, F64Type.get()),
            (NoneType, NoneType.get()),
            (ComplexType, ComplexType.get(f32)),
            (VectorType, VectorType.get([2, 3], f32)),
            (RankedTensorType, RankedTensorType.get([2, 3], f32)),
            (UnrankedTensorType, UnrankedTensorType.get(f32)),
            (MemRefType, MemRefType.get([2, 3], f32)),
            (UnrankedMemRefType, UnrankedMemRefType.get(f32, Attribute.parse("2"))),
            (TupleType, TupleType.get_tuple([f32])),
            (FunctionType, FunctionType.get([], [])),
            (OpaqueType, OpaqueType.get("tensor", "bob")),
        ]

        # CHECK: IntegerType(i16)
        # CHECK: IndexType(index)
        # CHECK: Float6E3M2FNType(f6E3M2FN)
        # CHECK: Float8E3M4Type(f8E3M4)
        # CHECK: Float8E4M3Type(f8E4M3)
        # CHECK: Float8E4M3FNType(f8E4M3FN)
        # CHECK: Float8E5M2Type(f8E5M2)
        # CHECK: Float8E4M3FNUZType(f8E4M3FNUZ)
        # CHECK: Float8E4M3B11FNUZType(f8E4M3B11FNUZ)
        # CHECK: Float8E5M2FNUZType(f8E5M2FNUZ)
        # CHECK: BF16Type(bf16)
        # CHECK: F16Type(f16)
        # CHECK: F32Type(f32)
        # CHECK: F64Type(f64)
        # CHECK: NoneType(none)
        # CHECK: ComplexType(complex<f32>)
        # CHECK: VectorType(vector<2x3xf32>)
        # CHECK: RankedTensorType(tensor<2x3xf32>)
        # CHECK: UnrankedTensorType(tensor<*xf32>)
        # CHECK: MemRefType(memref<2x3xf32>)
        # CHECK: UnrankedMemRefType(memref<*xf32, 2>)
        # CHECK: TupleType(tuple<f32>)
        # CHECK: FunctionType(() -> ())
        # CHECK: OpaqueType(!tensor.bob)
        for _, t in types:
            print(repr(t))

        # Test getTypeIdFunction agrees with
        # mlirTypeGetTypeID(self) for an instance.
        # CHECK: all equal
        for t1, t2 in types:
            tid1, tid2 = t1.static_typeid, Type(t2).typeid
            assert tid1 == tid2 and hash(tid1) == hash(
                tid2
            ), f"expected hash and value equality {t1} {t2}"
        else:
            print("all equal")

        # Test that storing PyTypeID in python dicts
        # works as expected.
        typeid_dict = dict(types)
        assert len(typeid_dict)

        # CHECK: all equal
        for t1, t2 in typeid_dict.items():
            assert t1.static_typeid == t2.typeid and hash(t1.static_typeid) == hash(
                t2.typeid
            ), f"expected hash and value equality {t1} {t2}"
        else:
            print("all equal")

        # CHECK: ShapedType has no typeid.
        try:
            print(ShapedType.static_typeid)
        except AttributeError as e:
            print(e)

        vector_type = Type.parse("vector<2x3xf32>")
        # CHECK: True
        print(ShapedType(vector_type).typeid == vector_type.typeid)


# CHECK-LABEL: TEST: testConcreteTypesRoundTrip
@run
def testConcreteTypesRoundTrip():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True

        def print_downcasted(typ):
            downcasted = Type(typ).maybe_downcast()
            print(type(downcasted).__name__)
            print(repr(downcasted))

        # CHECK: F16Type
        # CHECK: F16Type(f16)
        print_downcasted(F16Type.get())
        # CHECK: F32Type
        # CHECK: F32Type(f32)
        print_downcasted(F32Type.get())
        # CHECK: F64Type
        # CHECK: F64Type(f64)
        print_downcasted(F64Type.get())
        # CHECK: Float6E3M2FNType
        # CHECK: Float6E3M2FNType(f6E3M2FN)
        print_downcasted(Float6E3M2FNType.get())
        # CHECK: Float8E3M4Type
        # CHECK: Float8E3M4Type(f8E3M4)
        print_downcasted(Float8E3M4Type.get())
        # CHECK: Float8E4M3B11FNUZType
        # CHECK: Float8E4M3B11FNUZType(f8E4M3B11FNUZ)
        print_downcasted(Float8E4M3B11FNUZType.get())
        # CHECK: Float8E4M3Type
        # CHECK: Float8E4M3Type(f8E4M3)
        print_downcasted(Float8E4M3Type.get())
        # CHECK: Float8E4M3FNType
        # CHECK: Float8E4M3FNType(f8E4M3FN)
        print_downcasted(Float8E4M3FNType.get())
        # CHECK: Float8E4M3FNUZType
        # CHECK: Float8E4M3FNUZType(f8E4M3FNUZ)
        print_downcasted(Float8E4M3FNUZType.get())
        # CHECK: Float8E5M2Type
        # CHECK: Float8E5M2Type(f8E5M2)
        print_downcasted(Float8E5M2Type.get())
        # CHECK: Float8E5M2FNUZType
        # CHECK: Float8E5M2FNUZType(f8E5M2FNUZ)
        print_downcasted(Float8E5M2FNUZType.get())
        # CHECK: BF16Type
        # CHECK: BF16Type(bf16)
        print_downcasted(BF16Type.get())
        # CHECK: IndexType
        # CHECK: IndexType(index)
        print_downcasted(IndexType.get())
        # CHECK: IntegerType
        # CHECK: IntegerType(i32)
        print_downcasted(IntegerType.get_signless(32))

        f32 = F32Type.get()
        ranked_tensor = tensor.EmptyOp([10, 10], f32).result
        # CHECK: RankedTensorType
        print(type(ranked_tensor.type).__name__)
        # CHECK: RankedTensorType(tensor<10x10xf32>)
        print(repr(ranked_tensor.type))

        cf32 = ComplexType.get(f32)
        # CHECK: ComplexType
        print(type(cf32).__name__)
        # CHECK: ComplexType(complex<f32>)
        print(repr(cf32))

        ranked_tensor = tensor.EmptyOp([10, 10], f32).result
        # CHECK: RankedTensorType
        print(type(ranked_tensor.type).__name__)
        # CHECK: RankedTensorType(tensor<10x10xf32>)
        print(repr(ranked_tensor.type))

        vector = VectorType.get([10, 10], f32)
        tuple_type = TupleType.get_tuple([f32, vector])
        # CHECK: TupleType
        print(type(tuple_type).__name__)
        # CHECK: TupleType(tuple<f32, vector<10x10xf32>>)
        print(repr(tuple_type))
        # CHECK: F32Type(f32)
        print(repr(tuple_type.get_type(0)))
        # CHECK: VectorType(vector<10x10xf32>)
        print(repr(tuple_type.get_type(1)))

        index_type = IndexType.get()

        @func.FuncOp.from_py_func()
        def default_builder():
            c0 = arith.ConstantOp(f32, 0.0)
            unranked_tensor_type = UnrankedTensorType.get(f32)
            unranked_tensor = tensor.FromElementsOp(unranked_tensor_type, [c0]).result
            # CHECK: UnrankedTensorType
            print(type(unranked_tensor.type).__name__)
            # CHECK: UnrankedTensorType(tensor<*xf32>)
            print(repr(unranked_tensor.type))

            c10 = arith.ConstantOp(index_type, 10)
            memref_f32_t = MemRefType.get([10, 10], f32)
            memref_f32 = memref.AllocOp(memref_f32_t, [c10, c10], []).result
            # CHECK: MemRefType
            print(type(memref_f32.type).__name__)
            # CHECK: MemRefType(memref<10x10xf32>)
            print(repr(memref_f32.type))

            unranked_memref_t = UnrankedMemRefType.get(f32, Attribute.parse("2"))
            memref_f32 = memref.AllocOp(unranked_memref_t, [c10, c10], []).result
            # CHECK: UnrankedMemRefType
            print(type(memref_f32.type).__name__)
            # CHECK: UnrankedMemRefType(memref<*xf32, 2>)
            print(repr(memref_f32.type))

            tuple_type = Operation.parse(
                f'"test.make_tuple"() : () -> tuple<i32, f32>'
            ).result
            # CHECK: TupleType
            print(type(tuple_type.type).__name__)
            # CHECK: TupleType(tuple<i32, f32>)
            print(repr(tuple_type.type))

            return c0, c10


# CHECK-LABEL: TEST: testCustomTypeTypeCaster
# This tests being able to materialize a type from a dialect *and* have
# the implemented type caster called without explicitly importing the dialect.
# I.e., we get a transform.OperationType without explicitly importing the transform dialect.
@run
def testCustomTypeTypeCaster():
    with Context() as ctx, Location.unknown():
        t = Type.parse('!transform.op<"foo.bar">', Context())
        # CHECK: !transform.op<"foo.bar">
        print(t)
        # CHECK: OperationType(!transform.op<"foo.bar">)
        print(repr(t))


# CHECK-LABEL: TEST: testTypeWrappers
@run
def testTypeWrappers():
    def stride(strides, offset=0):
        return StridedLayoutAttr.get(offset, strides)

    with Context(), Location.unknown():
        ia = T.i(5)
        sia = T.si(6)
        uia = T.ui(7)
        assert repr(ia) == "IntegerType(i5)"
        assert repr(sia) == "IntegerType(si6)"
        assert repr(uia) == "IntegerType(ui7)"

        assert T.i(16) == T.i16()
        assert T.si(16) == T.si16()
        assert T.ui(16) == T.ui16()

        c1 = T.complex(T.f16())
        c2 = T.complex(T.i32())
        assert repr(c1) == "ComplexType(complex<f16>)"
        assert repr(c2) == "ComplexType(complex<i32>)"

        vec_1 = T.vector(2, 3, T.f32())
        vec_2 = T.vector(2, 3, 4, T.f32())
        assert repr(vec_1) == "VectorType(vector<2x3xf32>)"
        assert repr(vec_2) == "VectorType(vector<2x3x4xf32>)"

        m1 = T.memref(2, 3, 4, T.f64())
        assert repr(m1) == "MemRefType(memref<2x3x4xf64>)"

        m2 = T.memref(2, 3, 4, T.f64(), memory_space=1)
        assert repr(m2) == "MemRefType(memref<2x3x4xf64, 1>)"

        m3 = T.memref(2, 3, 4, T.f64(), memory_space=1, layout=stride([5, 7, 13]))
        assert repr(m3) == "MemRefType(memref<2x3x4xf64, strided<[5, 7, 13]>, 1>)"

        m4 = T.memref(2, 3, 4, T.f64(), memory_space=1, layout=stride([5, 7, 13], 42))
        assert (
            repr(m4)
            == "MemRefType(memref<2x3x4xf64, strided<[5, 7, 13], offset: 42>, 1>)"
        )

        S = ShapedType.get_dynamic_size()

        t1 = T.tensor(S, 3, S, T.f64())
        assert repr(t1) == "RankedTensorType(tensor<?x3x?xf64>)"
        ut1 = T.tensor(T.f64())
        assert repr(ut1) == "UnrankedTensorType(tensor<*xf64>)"
        t2 = T.tensor(S, 3, S, element_type=T.f64())
        assert repr(t2) == "RankedTensorType(tensor<?x3x?xf64>)"
        ut2 = T.tensor(element_type=T.f64())
        assert repr(ut2) == "UnrankedTensorType(tensor<*xf64>)"

        t3 = T.tensor(S, 3, S, T.f64(), encoding="encoding")
        assert repr(t3) == 'RankedTensorType(tensor<?x3x?xf64, "encoding">)'

        v = T.vector(3, 3, 3, T.f64())
        assert repr(v) == "VectorType(vector<3x3x3xf64>)"

        m5 = T.memref(S, 3, S, T.f64())
        assert repr(m5) == "MemRefType(memref<?x3x?xf64>)"
        um1 = T.memref(T.f64())
        assert repr(um1) == "UnrankedMemRefType(memref<*xf64>)"
        m6 = T.memref(S, 3, S, element_type=T.f64())
        assert repr(m6) == "MemRefType(memref<?x3x?xf64>)"
        um2 = T.memref(element_type=T.f64())
        assert repr(um2) == "UnrankedMemRefType(memref<*xf64>)"

        m7 = T.memref(S, 3, S, T.f64())
        assert repr(m7) == "MemRefType(memref<?x3x?xf64>)"
        um3 = T.memref(T.f64())
        assert repr(um3) == "UnrankedMemRefType(memref<*xf64>)"

        scalable_1 = T.vector(2, 3, T.f32(), scalable=[False, True])
        scalable_2 = T.vector(2, 3, 4, T.f32(), scalable=[True, False, True])
        assert repr(scalable_1) == "VectorType(vector<2x[3]xf32>)"
        assert repr(scalable_2) == "VectorType(vector<[2]x3x[4]xf32>)"

        scalable_3 = T.vector(2, 3, T.f32(), scalable_dims=[1])
        scalable_4 = T.vector(2, 3, 4, T.f32(), scalable_dims=[0, 2])
        assert scalable_3 == scalable_1
        assert scalable_4 == scalable_2

        opaq = T.opaque("scf", "placeholder")
        assert repr(opaq) == "OpaqueType(!scf.placeholder)"

        tup1 = T.tuple(T.i16(), T.i32(), T.i64())
        tup2 = T.tuple(T.f16(), T.f32(), T.f64())
        assert repr(tup1) == "TupleType(tuple<i16, i32, i64>)"
        assert repr(tup2) == "TupleType(tuple<f16, f32, f64>)"

        func = T.function(
            inputs=(T.i16(), T.i32(), T.i64()), results=(T.f16(), T.f32(), T.f64())
        )
        assert repr(func) == "FunctionType((i16, i32, i64) -> (f16, f32, f64))"
