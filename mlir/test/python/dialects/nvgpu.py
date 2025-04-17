# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import nvgpu, arith, memref


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: testTypes
@constructAndPrintInModule
def testTypes():
    tensorMemrefType = MemRefType.get(
        (128, 64), F16Type.get(), memory_space=Attribute.parse("3")
    )
    # CHECK: !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = l2promo_256b, oob = nan, interleave = none>
    tma_desc = nvgpu.TensorMapDescriptorType.get(
        tensorMemrefType,
        nvgpu.TensorMapSwizzleKind.SWIZZLE_128B,
        nvgpu.TensorMapL2PromoKind.L2PROMO_256B,
        nvgpu.TensorMapOOBKind.OOB_NAN,
        nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
    )
    print(tma_desc)


# CHECK-LABEL: testSmoke
@constructAndPrintInModule
def testSmoke():
    cst = arith.ConstantOp(value=42, result=IndexType.get())
    mem_t = MemRefType.get((10, 10), F32Type.get(), memory_space=Attribute.parse("3"))
    vec_t = VectorType.get((4, 1), F32Type.get())
    mem = memref.AllocOp(mem_t, [], [])
    # CHECK: %0 = nvgpu.ldmatrix %alloc[%c42, %c42] {numTiles = 4 : i32, transpose = false} : memref<10x10xf32, 3> -> vector<4x1xf32>
    nvgpu.LdMatrixOp(vec_t, mem, [cst, cst], False, 4)
