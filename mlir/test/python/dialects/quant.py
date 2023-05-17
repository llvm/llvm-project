# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import quant


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: test_type_hierarchy
@run
def test_type_hierarchy():
    with Context():
        i8 = IntegerType.get_signless(8)
        any = Type.parse("!quant.any<i8<-8:7>:f32>")
        uniform = Type.parse("!quant.uniform<i8<-8:7>:f32, 0.99872:127>")
        per_axis = Type.parse("!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>")
        calibrated = Type.parse("!quant.calibrated<f32<-0.998:1.2321>>")

        assert not quant.QuantizedType.isinstance(i8)
        assert quant.QuantizedType.isinstance(any)
        assert quant.QuantizedType.isinstance(uniform)
        assert quant.QuantizedType.isinstance(per_axis)
        assert quant.QuantizedType.isinstance(calibrated)

        assert quant.AnyQuantizedType.isinstance(any)
        assert quant.UniformQuantizedType.isinstance(uniform)
        assert quant.UniformQuantizedPerAxisType.isinstance(per_axis)
        assert quant.CalibratedQuantizedType.isinstance(calibrated)

        assert not quant.AnyQuantizedType.isinstance(uniform)
        assert not quant.UniformQuantizedType.isinstance(per_axis)


# CHECK-LABEL: TEST: test_any_quantized_type
@run
def test_any_quantized_type():
    with Context():
        i8 = IntegerType.get_signless(8)
        f32 = F32Type.get()
        any = quant.AnyQuantizedType.get(
            quant.QuantizedType.FLAG_SIGNED, i8, f32, -8, 7
        )

        # CHECK: flags: 1
        print(f"flags: {any.flags}")
        # CHECK: signed: True
        print(f"signed: {any.is_signed}")
        # CHECK: storage type: i8
        print(f"storage type: {any.storage_type}")
        # CHECK: expressed type: f32
        print(f"expressed type: {any.expressed_type}")
        # CHECK: storage min: -8
        print(f"storage min: {any.storage_type_min}")
        # CHECK: storage max: 7
        print(f"storage max: {any.storage_type_max}")
        # CHECK: storage width: 8
        print(f"storage width: {any.storage_type_integral_width}")
        # CHECK: quantized element type: !quant.any<i8<-8:7>:f32>
        print(f"quantized element type: {any.quantized_element_type}")
        # CHECK: !quant.any<i8<-8:7>:f32>
        print(any)
        assert any == Type.parse("!quant.any<i8<-8:7>:f32>")


# CHECK-LABEL: TEST: test_uniform_type
@run
def test_uniform_type():
    with Context():
        i8 = IntegerType.get_signless(8)
        f32 = F32Type.get()
        uniform = quant.UniformQuantizedType.get(
            quant.UniformQuantizedType.FLAG_SIGNED, i8, f32, 0.99872, 127, -8, 7
        )

        # CHECK: scale: 0.99872
        print(f"scale: {uniform.scale}")
        # CHECK: zero point: 127
        print(f"zero point: {uniform.zero_point}")
        # CHECK: fixed point: False
        print(f"fixed point: {uniform.is_fixed_point}")
        # CHECK: !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
        print(uniform)
        assert uniform == Type.parse("!quant.uniform<i8<-8:7>:f32, 0.99872:127>")


# CHECK-LABEL: TEST: test_uniform_per_axis_type
@run
def test_uniform_per_axis_type():
    with Context():
        i8 = IntegerType.get_signless(8)
        f32 = F32Type.get()
        per_axis = quant.UniformQuantizedPerAxisType.get(
            quant.QuantizedType.FLAG_SIGNED,
            i8,
            f32,
            [200, 0.99872],
            [0, 120],
            quantized_dimension=1,
            storage_type_min=quant.QuantizedType.default_minimum_for_integer(
                is_signed=True, integral_width=8
            ),
            storage_type_max=quant.QuantizedType.default_maximum_for_integer(
                is_signed=True, integral_width=8
            ),
        )

        # CHECK: scales: None
        print(f"scales: {per_axis.scales}")
        # CHECK: zero_points: None
        print(f"zero_points: {per_axis.zero_points}")
        # CHECK: quantized dim: 1
        print(f"quantized dim: {per_axis.quantized_dimension}")
        # CHECK: fixed point: False
        print(f"fixed point: {per_axis.is_fixed_point}")
        # CHECK: !quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01:120}>
        print(per_axis)
        assert per_axis == Type.parse("!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>")


# CHECK-LABEL: TEST: test_calibrated_type
@run
def test_calibrated_type():
    with Context():
        f32 = F32Type.get()
        calibrated = quant.CalibratedQuantizedType.get(f32, -0.998, 1.2321)

        # CHECK: min: -0.998
        print(f"min: {calibrated.min}")
        # CHECK: max: 1.2321
        print(f"max: {calibrated.max}")
        # CHECK: !quant.calibrated<f32<-0.998:1.232100e+00>>
        print(calibrated)
        assert calibrated == Type.parse("!quant.calibrated<f32<-0.998:1.2321>>")
