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
        module.operation.verify()
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


@constructAndPrintInModule
def test_barriers():
    i32 = T.i32()
    f32 = T.f32()

    @func.FuncOp.from_py_func(i32, i32, f32)
    def barriers(mask, vi32, vf32):
        c0 = arith.constant(T.i32(), 0)
        cffff = arith.constant(T.i32(), 0xFFFF)
        res = nvvm.barrier(
            res=i32,
            barrier_id=c0,
            number_of_threads=cffff,
        )

        for reduction in (
            nvvm.BarrierReduction.AND,
            nvvm.BarrierReduction.OR,
            nvvm.BarrierReduction.POPC,
        ):
            res = nvvm.barrier(
                res=i32,
                reduction_op=reduction,
                reduction_predicate=res,
            )

        nvvm.barrier0()
        nvvm.bar_warp_sync(mask)
        nvvm.cluster_arrive()
        nvvm.cluster_arrive(aligned=True)
        nvvm.cluster_arrive_relaxed()
        nvvm.cluster_arrive_relaxed(aligned=True)
        nvvm.cluster_wait()
        nvvm.cluster_wait(aligned=True)
        nvvm.fence_mbarrier_init()
        nvvm.bar_warp_sync(mask)
        return res


# CHECK-LABEL:   func.func @barriers(
# CHECK:           %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: f32) -> i32 {
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 65535 : i32
# CHECK:           %[[BARRIER_0:.*]] = nvvm.barrier id = %[[CONSTANT_0]] number_of_threads = %[[CONSTANT_1]] -> i32
# CHECK:           %[[BARRIER_1:.*]] = nvvm.barrier #nvvm.reduction<and> %[[BARRIER_0]] -> i32
# CHECK:           %[[BARRIER_2:.*]] = nvvm.barrier #nvvm.reduction<or> %[[BARRIER_1]] -> i32
# CHECK:           %[[BARRIER_3:.*]] = nvvm.barrier #nvvm.reduction<popc> %[[BARRIER_2]] -> i32
# CHECK:           nvvm.barrier0
# CHECK:           nvvm.bar.warp.sync %[[ARG0]] : i32
# CHECK:           nvvm.cluster.arrive
# CHECK:           nvvm.cluster.arrive {aligned}
# CHECK:           nvvm.cluster.arrive.relaxed
# CHECK:           nvvm.cluster.arrive.relaxed {aligned}
# CHECK:           nvvm.cluster.wait
# CHECK:           nvvm.cluster.wait {aligned}
# CHECK:           nvvm.fence.mbarrier.init
# CHECK:           nvvm.bar.warp.sync %[[ARG0]] : i32
# CHECK:           return %[[BARRIER_3]] : i32
# CHECK:         }


@constructAndPrintInModule
def test_reductions():
    i32 = T.i32()
    f32 = T.f32()

    @func.FuncOp.from_py_func(i32, i32, f32)
    def reductions(mask, vi32, vf32):
        for abs in (True, False):
            for nan in (True, False):
                for kind in (
                    nvvm.ReduxKind.AND,
                    nvvm.ReduxKind.MAX,
                    nvvm.ReduxKind.MIN,
                    nvvm.ReduxKind.OR,
                    nvvm.ReduxKind.UMAX,
                    nvvm.ReduxKind.UMIN,
                    nvvm.ReduxKind.XOR,
                ):
                    nvvm.redux_sync(i32, vi32, kind, vi32)

                for kind in (
                    nvvm.ReduxKind.FMIN,
                    nvvm.ReduxKind.FMAX,
                ):
                    nvvm.redux_sync(f32, vf32, kind, vi32, abs=abs, nan=nan)


# CHECK-LABEL:   func.func @reductions(
# CHECK:           %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: f32) {
# CHECK:           %[[REDUX_0:.*]] = nvvm.redux.sync  and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_1:.*]] = nvvm.redux.sync  max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_2:.*]] = nvvm.redux.sync  min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_3:.*]] = nvvm.redux.sync  or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_4:.*]] = nvvm.redux.sync  umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_5:.*]] = nvvm.redux.sync  umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_6:.*]] = nvvm.redux.sync  xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_7:.*]] = nvvm.redux.sync  fmin %[[ARG2]], %[[ARG1]] {abs = true, nan = true} : f32 -> f32
# CHECK:           %[[REDUX_8:.*]] = nvvm.redux.sync  fmax %[[ARG2]], %[[ARG1]] {abs = true, nan = true} : f32 -> f32
# CHECK:           %[[REDUX_9:.*]] = nvvm.redux.sync  and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_10:.*]] = nvvm.redux.sync  max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_11:.*]] = nvvm.redux.sync  min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_12:.*]] = nvvm.redux.sync  or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_13:.*]] = nvvm.redux.sync  umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_14:.*]] = nvvm.redux.sync  umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_15:.*]] = nvvm.redux.sync  xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_16:.*]] = nvvm.redux.sync  fmin %[[ARG2]], %[[ARG1]] {abs = true} : f32 -> f32
# CHECK:           %[[REDUX_17:.*]] = nvvm.redux.sync  fmax %[[ARG2]], %[[ARG1]] {abs = true} : f32 -> f32
# CHECK:           %[[REDUX_18:.*]] = nvvm.redux.sync  and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_19:.*]] = nvvm.redux.sync  max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_20:.*]] = nvvm.redux.sync  min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_21:.*]] = nvvm.redux.sync  or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_22:.*]] = nvvm.redux.sync  umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_23:.*]] = nvvm.redux.sync  umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_24:.*]] = nvvm.redux.sync  xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_25:.*]] = nvvm.redux.sync  fmin %[[ARG2]], %[[ARG1]] {nan = true} : f32 -> f32
# CHECK:           %[[REDUX_26:.*]] = nvvm.redux.sync  fmax %[[ARG2]], %[[ARG1]] {nan = true} : f32 -> f32
# CHECK:           %[[REDUX_27:.*]] = nvvm.redux.sync  and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_28:.*]] = nvvm.redux.sync  max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_29:.*]] = nvvm.redux.sync  min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_30:.*]] = nvvm.redux.sync  or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_31:.*]] = nvvm.redux.sync  umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_32:.*]] = nvvm.redux.sync  umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_33:.*]] = nvvm.redux.sync  xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_34:.*]] = nvvm.redux.sync  fmin %[[ARG2]], %[[ARG1]] : f32 -> f32
# CHECK:           %[[REDUX_35:.*]] = nvvm.redux.sync  fmax %[[ARG2]], %[[ARG1]] : f32 -> f32
# CHECK:           return
# CHECK:         }
