# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_standalone.ir import *

if sys.argv[1] == "pybind11":
    from mlir_standalone.dialects import standalone_pybind11 as standalone_d
elif sys.argv[1] == "nanobind":
    from mlir_standalone.dialects import standalone_nanobind as standalone_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")

from mlir_standalone.dialects import arith, quant


with Context(), Location.unknown():
    standalone_d.register_dialects()
    f32 = F32Type.get()
    i8 = IntegerType.get_signless(8)
    i32 = IntegerType.get_signless(32)
    uniform = quant.UniformQuantizedType.get(
        quant.UniformQuantizedType.FLAG_SIGNED, i8, f32, 0.99872, 127, -8, 7
    )

    module = Module.create()
    with InsertionPoint(module.body):
        two_i32 = arith.constant(i32, 2)
        standalone_d.foo(two_i32)
        two_f32 = arith.constant(f32, 2.0)
        quant.qcast(uniform, two_f32)
    # CHECK: %[[TWOI32:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[TWOI32]] : i32
    # CHECK: %[[TWOF32:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK: quant.qcast %[[TWOF32]] : f32 to !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
    print(str(module))
