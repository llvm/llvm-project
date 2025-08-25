# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import nvvm
from mlir.dialects import llvm
from mlir.dialects import func
import mlir.extras.types as T
from mlir.dialects import arith


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
        # CHECK: %1 = nvvm.wgmma.mma_async %arg0, %arg1, %0, <m = 64, n = 32, k = 16>, D[<f32>, <zero>], A[<f16>, <neg>, <col>], B[<f16>, <neg>, <col>] : [[MAT_T]] -> [[MAT_T]]
        result1 = nvvm.WgmmaMmaAsyncOp(
            results_=mat64f32_t,
            inouts=result,
            descriptorA=desc_a,
            descriptorB=desc_b,
            shape=shape_attr,
            typeA=nvvm.WGMMATypes.f16,
            typeB=nvvm.WGMMATypes.f16,
            typeD=nvvm.WGMMATypes.f32,
            scaleD=nvvm.WGMMAScaleOut.zero,
            scaleA=nvvm.WGMMAScaleIn.neg,
            scaleB=nvvm.WGMMAScaleIn.neg,
            layoutA=nvvm.MMALayout.col,
            layoutB=nvvm.MMALayout.col,
        )


# CHECK-LABEL: TEST: test_inline_ptx
# CHECK-LABEL: func.func @my_inline_ptx(
# CHECK-SAME: %[[arg0:[a-zA-Z0-9_]+]]: f32, %[[arg1:[a-zA-Z0-9_]+]]: f32, %[[arg2:[a-zA-Z0-9_]+]]: i32, %[[arg3:[a-zA-Z0-9_]+]]: i32)
# CHECK: %[[S0:.+]]:2 = nvvm.inline_ptx
# CHECK-SAME: ro(%[[arg0]], %[[arg1]] : f32, f32) rw(%[[arg2]], %[[arg3]] : i32, i32) -> f32, f32
# CHECK: %[[S1:.+]] = arith.addf %[[arg0]], %[[arg1]] : f32
# CHECK: %[[S2:.+]] = arith.addi %[[arg2]], %[[arg3]] : i32
# CHECK: %[[S3:.+]] = arith.addf %[[S0]]#0, %[[S0]]#1 : f32


@constructAndPrintInModule
def test_inline_ptx():
    i32 = T.i32()
    f32 = T.f32()

    @func.FuncOp.from_py_func(f32, f32, i32, i32)
    def my_inline_ptx(a, b, c, d):
        ptx = r"""
            {
                .reg .pred p;
                setp.ge.s32   p,      {$r0}, {$r1};
                selp.s32      {$r0},  {$r0}, {$r1}, p;
                selp.s32      {$r1},  {$r0}, {$r1}, p;
                selp.s32      {$rw0}, {$r0}, {$r1}, p;
                selp.s32      {$rw1}, {$r0}, {$r1}, p;
            }
            """
        wo0, wo1 = nvvm.inline_ptx(
            read_only_args=[a, b],
            read_write_args=[c, d],
            write_only_args=[f32, f32],
            ptx_code=ptx,
        )
        arith.addf(a, b)
        arith.addi(c, d)
        arith.addf(wo0, wo1)
