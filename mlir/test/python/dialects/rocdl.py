# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.
from array import array

from mlir.ir import *
from mlir.dialects import rocdl, arith
from mlir.extras import types as T


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: testSmoke
@constructAndPrintInModule
def testSmoke():
    v_len = 16
    f32 = F32Type.get()
    # Note: this isn't actually the right type for the intrinsic (should be f16)
    # but array doesn't support f16.
    v16f32 = T.vector(v_len, f32)
    f32_array = array("f", [0.0] * v_len)
    a_frag = arith.constant(v16f32, f32_array)
    b_frag = arith.constant(v16f32, f32_array)
    c_frag = arith.constant(v16f32, f32_array)
    false = arith.constant(T.bool(), False)

    c_frag = rocdl.wmma_f16_16x16x16_f16(v16f32, [a_frag, b_frag, c_frag, false])
    # CHECK: %{{.*}} = rocdl.wmma.f16.16x16x16.f16
    print(c_frag)
    assert isinstance(c_frag, OpView)
    # CHECK: Value(%{{.*}} = rocdl.wmma.f16.16x16x16.f16
    c_frag = rocdl.wmma_f16_16x16x16_f16_(v16f32, [a_frag, b_frag, c_frag, false])
    print(c_frag)
    assert isinstance(c_frag, Value)
