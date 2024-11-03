# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import nvvm
from mlir.dialects import llvm
from mlir.dialects import func


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
    i64 = IntegerType.get_signless(64)
    mat64f32_t = Type.parse(
        "!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>"
    )
    shape_attr = Attribute.parse("#nvvm.shape<m = 64, n = 32, k = 16>")
    # CHECK-LABEL: func @wgmma_f32_f16_f16(%arg0: i64, %arg1: i64)
    @func.FuncOp.from_py_func(i64, i64)
    def wgmma_f32_f16_f16(desc_a, desc_b):
        # CHECK: nvvm.cp.async.wait.group 5
        nvvm.CpAsyncWaitGroupOp(5)
        # CHECK: %0 = llvm.mlir.undef : [[MAT_T:.*]]
        result = llvm.UndefOp(mat64f32_t)
        # CHECK: %1 = nvvm.wgmma.mma_async %arg0, %arg1, <m = 64, n = 32, k = 16>, D[%0, <zero>], A[<f16>, <neg>, <col>], B[<f16>, <neg>, <col>] : [[MAT_T]] -> [[MAT_T]]
        result1 = nvvm.WgmmaMmaAsyncOp(
            results_=mat64f32_t,
            inouts=result,
            descriptorA=desc_a,
            descriptorB=desc_b,
            shape=shape_attr,
            typeA=nvvm.WGMMATypes.f16,
            typeB=nvvm.WGMMATypes.f16,
            scaleD=nvvm.WGMMAScaleOut.zero,
            scaleA=nvvm.WGMMAScaleIn.neg,
            scaleB=nvvm.WGMMAScaleIn.neg,
            layoutA=nvvm.MMALayout.col,
            layoutB=nvvm.MMALayout.col,
        )
