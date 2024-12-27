# RUN: %PYTHON %s | FileCheck %s
# Note that this is separate from ir_attributes.py since it depends on numpy,
# and we may want to disable if not available.

import gc
from mlir.ir import *
import numpy as np
import weakref


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


################################################################################
# Tests of the array/buffer .get() factory method on unsupported dtype.
################################################################################


@run
def testGetDenseElementsUnsupported():
    with Context():
        array = np.array([["hello", "goodbye"]])
        try:
            attr = DenseElementsAttr.get(array)
        except ValueError as e:
            # CHECK: unimplemented array format conversion from format:
            print(e)

# CHECK-LABEL: TEST: testGetDenseElementsUnSupportedTypeOkIfExplicitTypeProvided
@run
def testGetDenseElementsUnSupportedTypeOkIfExplicitTypeProvided():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        # datetime64 specifically isn't important: it's just a 64-bit type that
        # doesn't have a format under the Python buffer protocol. A more
        # realistic example would be a NumPy extension type like the bfloat16
        # type from the ml_dtypes package, which isn't a dependency of this
        # test.
        attr = DenseElementsAttr.get(array.view(np.datetime64),
                                     type=IntegerType.get_signless(64))
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


################################################################################
# Tests of the list of attributes .get() factory method
################################################################################


# CHECK-LABEL: TEST: testGetDenseElementsFromList
@run
def testGetDenseElementsFromList():
    with Context(), Location.unknown():
        attrs = [FloatAttr.get(F64Type.get(), 1.0), FloatAttr.get(F64Type.get(), 2.0)]
        attr = DenseElementsAttr.get(attrs)

        # CHECK: dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
        print(attr)


# CHECK-LABEL: TEST: testGetDenseElementsFromListWithExplicitType
@run
def testGetDenseElementsFromListWithExplicitType():
    with Context(), Location.unknown():
        attrs = [FloatAttr.get(F64Type.get(), 1.0), FloatAttr.get(F64Type.get(), 2.0)]
        shaped_type = ShapedType(Type.parse("tensor<2xf64>"))
        attr = DenseElementsAttr.get(attrs, shaped_type)

        # CHECK: dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
        print(attr)


# CHECK-LABEL: TEST: testGetDenseElementsFromListEmptyList
@run
def testGetDenseElementsFromListEmptyList():
    with Context(), Location.unknown():
        attrs = []

        try:
            attr = DenseElementsAttr.get(attrs)
        except ValueError as e:
            # CHECK: Attributes list must be non-empty
            print(e)


# CHECK-LABEL: TEST: testGetDenseElementsFromListNonAttributeType
@run
def testGetDenseElementsFromListNonAttributeType():
    with Context(), Location.unknown():
        attrs = [1.0]

        try:
            attr = DenseElementsAttr.get(attrs)
        except RuntimeError as e:
            # CHECK: Invalid attribute when attempting to create an ArrayAttribute
            print(e)


# CHECK-LABEL: TEST: testGetDenseElementsFromListMismatchedType
@run
def testGetDenseElementsFromListMismatchedType():
    with Context(), Location.unknown():
        attrs = [FloatAttr.get(F64Type.get(), 1.0), FloatAttr.get(F64Type.get(), 2.0)]
        shaped_type = ShapedType(Type.parse("tensor<2xf32>"))

        try:
            attr = DenseElementsAttr.get(attrs, shaped_type)
        except ValueError as e:
            # CHECK: All attributes must be of the same type and match the type parameter
            print(e)


# CHECK-LABEL: TEST: testGetDenseElementsFromListMixedTypes
@run
def testGetDenseElementsFromListMixedTypes():
    with Context(), Location.unknown():
        attrs = [FloatAttr.get(F64Type.get(), 1.0), FloatAttr.get(F32Type.get(), 2.0)]

        try:
            attr = DenseElementsAttr.get(attrs)
        except ValueError as e:
            # CHECK: All attributes must be of the same type and match the type parameter
            print(e)


################################################################################
# Splats.
################################################################################

# CHECK-LABEL: TEST: testGetDenseElementsSplatInt
@run
def testGetDenseElementsSplatInt():
    with Context(), Location.unknown():
        t = IntegerType.get_signless(32)
        element = IntegerAttr.get(t, 555)
        shaped_type = RankedTensorType.get((2, 3, 4), t)
        attr = DenseElementsAttr.get_splat(shaped_type, element)
        # CHECK: dense<555> : tensor<2x3x4xi32>
        print(attr)
        # CHECK: is_splat: True
        print("is_splat:", attr.is_splat)

        # CHECK: splat_value: IntegerAttr(555 : i32)
        splat_value = attr.get_splat_value()
        print("splat_value:", repr(splat_value))
        assert splat_value == element


# CHECK-LABEL: TEST: testGetDenseElementsSplatFloat
@run
def testGetDenseElementsSplatFloat():
    with Context(), Location.unknown():
        t = F32Type.get()
        element = FloatAttr.get(t, 1.2)
        shaped_type = RankedTensorType.get((2, 3, 4), t)
        attr = DenseElementsAttr.get_splat(shaped_type, element)
        # CHECK: dense<1.200000e+00> : tensor<2x3x4xf32>
        print(attr)
        assert attr.get_splat_value() == element


# CHECK-LABEL: TEST: testGetDenseElementsSplatErrors
@run
def testGetDenseElementsSplatErrors():
    with Context(), Location.unknown():
        t = F32Type.get()
        other_t = F64Type.get()
        element = FloatAttr.get(t, 1.2)
        other_element = FloatAttr.get(other_t, 1.2)
        shaped_type = RankedTensorType.get((2, 3, 4), t)
        dynamic_shaped_type = UnrankedTensorType.get(t)
        non_shaped_type = t

        try:
            attr = DenseElementsAttr.get_splat(non_shaped_type, element)
        except ValueError as e:
            # CHECK: Expected a static ShapedType for the shaped_type parameter: Type(f32)
            print(e)

        try:
            attr = DenseElementsAttr.get_splat(dynamic_shaped_type, element)
        except ValueError as e:
            # CHECK: Expected a static ShapedType for the shaped_type parameter: Type(tensor<*xf32>)
            print(e)

        try:
            attr = DenseElementsAttr.get_splat(shaped_type, other_element)
        except ValueError as e:
            # CHECK: Shaped element type and attribute type must be equal: shaped=Type(tensor<2x3x4xf32>), element=Attribute(1.200000e+00 : f64)
            print(e)


# CHECK-LABEL: TEST: testRepeatedValuesSplat
@run
def testRepeatedValuesSplat():
    with Context():
        array = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<1.000000e+00> : tensor<2x3xf32>
        print(attr)
        # CHECK: is_splat: True
        print("is_splat:", attr.is_splat)
        # CHECK{LITERAL}: [[1. 1. 1.]
        # CHECK{LITERAL}:  [1. 1. 1.]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testNonSplat
@run
def testNonSplat():
    with Context():
        array = np.array([2.0, 1.0, 1.0], dtype=np.float32)
        attr = DenseElementsAttr.get(array)
        # CHECK: is_splat: False
        print("is_splat:", attr.is_splat)


################################################################################
# Tests of the array/buffer .get() factory method, in all of its permutations.
################################################################################

### explicitly provided types


@run
def testGetDenseElementsBF16():
    with Context():
        array = np.array([[2, 4, 8], [16, 32, 64]], dtype=np.uint16)
        attr = DenseElementsAttr.get(array, type=BF16Type.get())
        # Note: These values don't mean much since just bit-casting. But they
        # shouldn't change.
        # CHECK: dense<{{\[}}[1.836710e-40, 3.673420e-40, 7.346840e-40], [1.469370e-39, 2.938740e-39, 5.877470e-39]]> : tensor<2x3xbf16>
        print(attr)


@run
def testGetDenseElementsInteger4():
    with Context():
        array = np.array([[2, 4, 7], [-2, -4, -8]], dtype=np.int8)
        attr = DenseElementsAttr.get(array, type=IntegerType.get_signless(4))
        # Note: These values don't mean much since just bit-casting. But they
        # shouldn't change.
        # CHECK: dense<{{\[}}[2, 4, 7], [-2, -4, -8]]> : tensor<2x3xi4>
        print(attr)


@run
def testGetDenseElementsBool():
    with Context():
        bool_array = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool_)
        array = np.packbits(bool_array, axis=None, bitorder="little")
        attr = DenseElementsAttr.get(
            array, type=IntegerType.get_signless(1), shape=bool_array.shape
        )
        # CHECK: dense<{{\[}}[true, false, true], [false, true, false]]> : tensor<2x3xi1>
        print(attr)


@run
def testGetDenseElementsBoolSplat():
    with Context():
        zero = np.array(0, dtype=np.uint8)
        one = np.array(255, dtype=np.uint8)
        print(one)
        # CHECK: dense<false> : tensor<4x2x5xi1>
        print(
            DenseElementsAttr.get(
                zero, type=IntegerType.get_signless(1), shape=(4, 2, 5)
            )
        )
        # CHECK: dense<true> : tensor<4x2x5xi1>
        print(
            DenseElementsAttr.get(
                one, type=IntegerType.get_signless(1), shape=(4, 2, 5)
            )
        )


### float and double arrays.


# CHECK-LABEL: TEST: testGetDenseElementsF16
@run
def testGetDenseElementsF16():
    with Context():
        array = np.array([[2.0, 4.0, 8.0], [16.0, 32.0, 64.0]], dtype=np.float16)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[2.000000e+00, 4.000000e+00, 8.000000e+00], [1.600000e+01, 3.200000e+01, 6.400000e+01]]> : tensor<2x3xf16>
        print(attr)
        # CHECK: {{\[}}[ 2. 4. 8.]
        # CHECK: {{\[}}16. 32. 64.]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsF32
@run
def testGetDenseElementsF32():
    with Context():
        array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1.100000e+00, 2.200000e+00, 3.300000e+00], [4.400000e+00, 5.500000e+00, 6.600000e+00]]> : tensor<2x3xf32>
        print(attr)
        # CHECK: {{\[}}[1.1 2.2 3.3]
        # CHECK: {{\[}}4.4 5.5 6.6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsF64
@run
def testGetDenseElementsF64():
    with Context():
        array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1.100000e+00, 2.200000e+00, 3.300000e+00], [4.400000e+00, 5.500000e+00, 6.600000e+00]]> : tensor<2x3xf64>
        print(attr)
        # CHECK: {{\[}}[1.1 2.2 3.3]
        # CHECK: {{\[}}4.4 5.5 6.6]]
        print(np.array(attr))


### 1 bit/boolean integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI1Signless
@run
def testGetDenseElementsI1Signless():
    with Context():
        array = np.array([True], dtype=np.bool_)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<true> : tensor<1xi1>
        print(attr)
        # CHECK{LITERAL}: [ True]
        print(np.array(attr))

        array = np.array([[True, False, True], [True, True, False]], dtype=np.bool_)
        attr = DenseElementsAttr.get(array)
        # CHECK{LITERAL}: dense<[[true, false, true], [true, true, false]]> : tensor<2x3xi1>
        print(attr)
        # CHECK{LITERAL}: [[ True False True]
        # CHECK{LITERAL}:  [ True True False]]
        print(np.array(attr))

        array = np.array(
            [[True, True, False, False], [True, False, True, False]], dtype=np.bool_
        )
        attr = DenseElementsAttr.get(array)
        # CHECK{LITERAL}: dense<[[true, true, false, false], [true, false, true, false]]> : tensor<2x4xi1>
        print(attr)
        # CHECK{LITERAL}: [[ True True False False]
        # CHECK{LITERAL}:  [ True False True False]]
        print(np.array(attr))

        array = np.array(
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, False, False, False],
                [True, True, True, True],
                [True, False, False, True],
            ],
            dtype=np.bool_,
        )
        attr = DenseElementsAttr.get(array)
        # CHECK{LITERAL}: dense<[[true, true, false, false], [true, false, true, false], [false, false, false, false], [true, true, true, true], [true, false, false, true]]> : tensor<5x4xi1>
        print(attr)
        # CHECK{LITERAL}: [[ True True False False]
        # CHECK{LITERAL}:  [ True False True False]
        # CHECK{LITERAL}:  [False False False False]
        # CHECK{LITERAL}:  [ True True True True]
        # CHECK{LITERAL}:  [ True False False True]]
        print(np.array(attr))

        array = np.array(
            [
                [True, True, False, False, True, True, False, False, False],
                [False, False, False, True, False, True, True, False, True],
            ],
            dtype=np.bool_,
        )
        attr = DenseElementsAttr.get(array)
        # CHECK{LITERAL}: dense<[[true, true, false, false, true, true, false, false, false], [false, false, false, true, false, true, true, false, true]]> : tensor<2x9xi1>
        print(attr)
        # CHECK{LITERAL}: [[ True True False False True True False False False]
        # CHECK{LITERAL}:  [False False False True False True True False True]]
        print(np.array(attr))

        array = np.array([], dtype=np.bool_)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<> : tensor<0xi1>
        print(attr)
        # CHECK{LITERAL}: []
        print(np.array(attr))


### 16 bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI16Signless
@run
def testGetDenseElementsI16Signless():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI16Signless
@run
def testGetDenseElementsUI16Signless():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsI16
@run
def testGetDenseElementsI16():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        attr = DenseElementsAttr.get(array, signless=False)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi16>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI16
@run
def testGetDenseElementsUI16():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        attr = DenseElementsAttr.get(array, signless=False)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui16>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


### 32 bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI32Signless
@run
def testGetDenseElementsI32Signless():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI32Signless
@run
def testGetDenseElementsUI32Signless():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsI32
@run
def testGetDenseElementsI32():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        attr = DenseElementsAttr.get(array, signless=False)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi32>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI32
@run
def testGetDenseElementsUI32():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
        attr = DenseElementsAttr.get(array, signless=False)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui32>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


## 64bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI64Signless
@run
def testGetDenseElementsI64Signless():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI64Signless
@run
def testGetDenseElementsUI64Signless():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
        attr = DenseElementsAttr.get(array)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsI64
@run
def testGetDenseElementsI64():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        attr = DenseElementsAttr.get(array, signless=False)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi64>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI64
@run
def testGetDenseElementsUI64():
    with Context():
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
        attr = DenseElementsAttr.get(array, signless=False)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui64>
        print(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsIndex
@run
def testGetDenseElementsIndex():
    with Context(), Location.unknown():
        idx_type = IndexType.get()
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        attr = DenseElementsAttr.get(array, type=idx_type)
        # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xindex>
        print(attr)
        arr = np.array(attr)
        # CHECK: {{\[}}[1 2 3]
        # CHECK: {{\[}}4 5 6]]
        print(arr)
        # CHECK: True
        print(arr.dtype == np.int64)


# CHECK-LABEL: TEST: testGetDenseResourceElementsAttr
@run
def testGetDenseResourceElementsAttr():
    def on_delete(_):
        print("BACKING MEMORY DELETED")

    context = Context()
    mview = memoryview(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ref = weakref.ref(mview, on_delete)

    def test_attribute(context, mview):
        with context, Location.unknown():
            element_type = IntegerType.get_signless(32)
            tensor_type = RankedTensorType.get((2, 3), element_type)
            resource = DenseResourceElementsAttr.get_from_buffer(
                mview, "from_py", tensor_type
            )
            module = Module.parse("module {}")
            module.operation.attributes["test.resource"] = resource
            # CHECK: test.resource = dense_resource<from_py> : tensor<2x3xi32>
            # CHECK: from_py: "0x04000000010000000200000003000000040000000500000006000000"
            print(module)

            # Verifies type casting.
            # CHECK: dense_resource<from_py> : tensor<2x3xi32>
            print(
                DenseResourceElementsAttr(module.operation.attributes["test.resource"])
            )

    test_attribute(context, mview)
    mview = None
    gc.collect()
    # CHECK: FREEING CONTEXT
    print("FREEING CONTEXT")
    context = None
    gc.collect()
    # CHECK: BACKING MEMORY DELETED
    # CHECK: EXIT FUNCTION
    print("EXIT FUNCTION")
