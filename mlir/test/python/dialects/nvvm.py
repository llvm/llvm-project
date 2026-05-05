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
def test_mbarrier_arrive():
    ptr_shared = llvm.PointerType.get(3)
    ptr_cluster = llvm.PointerType.get(7)
    i32 = T.i32()

    @func.FuncOp.from_py_func(ptr_shared, ptr_cluster, i32)
    def mbarrier_arrive_ops(barrier_shared, barrier_cluster, txcount):
        token = nvvm.mbarrier_arrive(barrier_shared)
        nvvm.mbarrier_arrive(barrier_cluster)
        token2 = nvvm.mbarrier_arrive_drop(barrier_shared)
        nvvm.mbarrier_arrive_drop(barrier_cluster)
        token3 = nvvm.mbarrier_arrive_expect_tx(barrier_shared, txcount)
        nvvm.mbarrier_arrive_expect_tx(barrier_cluster, txcount)
        token4 = nvvm.mbarrier_arrive_drop_expect_tx(barrier_shared, txcount)
        nvvm.mbarrier_arrive_drop_expect_tx(barrier_cluster, txcount)


# CHECK-LABEL:   func.func @mbarrier_arrive_ops(
# CHECK-SAME:      %[[SHARED:.*]]: !llvm.ptr<3>, %[[CLUSTER:.*]]: !llvm.ptr<7>, %[[TXCOUNT:.*]]: i32)
# CHECK:           %{{.*}} = nvvm.mbarrier.arrive %[[SHARED]] : !llvm.ptr<3> -> i64
# CHECK-NEXT:      nvvm.mbarrier.arrive %[[CLUSTER]] : !llvm.ptr<7>{{$}}
# CHECK-NEXT:      %{{.*}} = nvvm.mbarrier.arrive_drop %[[SHARED]] : !llvm.ptr<3> -> i64
# CHECK-NEXT:      nvvm.mbarrier.arrive_drop %[[CLUSTER]] : !llvm.ptr<7>{{$}}
# CHECK-NEXT:      %{{.*}} = nvvm.mbarrier.arrive.expect_tx %[[SHARED]], %[[TXCOUNT]] : !llvm.ptr<3>, i32 -> i64
# CHECK-NEXT:      nvvm.mbarrier.arrive.expect_tx %[[CLUSTER]], %[[TXCOUNT]] : !llvm.ptr<7>, i32{{$}}
# CHECK-NEXT:      %{{.*}} = nvvm.mbarrier.arrive_drop.expect_tx %[[SHARED]], %[[TXCOUNT]] : !llvm.ptr<3>, i32 -> i64
# CHECK-NEXT:      nvvm.mbarrier.arrive_drop.expect_tx %[[CLUSTER]], %[[TXCOUNT]] : !llvm.ptr<7>, i32{{$}}


@constructAndPrintInModule
def test_barriers():
    i32 = T.i32()
    f32 = T.f32()

    @func.FuncOp.from_py_func(i32, i32, f32)
    def barriers(mask, vi32, vf32):
        c0 = arith.constant(T.i32(), 0)
        cffff = arith.constant(T.i32(), 0xFFFF)
        nvvm.barrier(
            barrier_id=c0,
            number_of_threads=cffff,
        )

        pred = arith.constant(T.i32(), 1)
        for reduction in (
            nvvm.BarrierReduction.AND,
            nvvm.BarrierReduction.OR,
            nvvm.BarrierReduction.POPC,
        ):
            pred = nvvm.barrier(
                reduction_op=reduction,
                reduction_predicate=pred,
            )

        nvvm.barrier()
        nvvm.bar_warp_sync(mask)
        nvvm.cluster_arrive()
        nvvm.cluster_arrive(aligned=True)
        nvvm.cluster_arrive_relaxed()
        nvvm.cluster_arrive_relaxed(aligned=True)
        nvvm.cluster_wait()
        nvvm.cluster_wait(aligned=True)
        nvvm.fence_mbarrier_init()
        nvvm.bar_warp_sync(mask)
        return pred


# CHECK-LABEL:   func.func @barriers(
# CHECK:           %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: f32) -> i32 {
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 65535 : i32
# CHECK:           nvvm.barrier id = %[[CONSTANT_0]] number_of_threads = %[[CONSTANT_1]]
# CHECK:           %[[PRED:.*]] = arith.constant 1 : i32
# CHECK:           %[[BARRIER_1:.*]] = nvvm.barrier #nvvm.reduction<and> %[[PRED]] -> i32
# CHECK:           %[[BARRIER_2:.*]] = nvvm.barrier #nvvm.reduction<or> %[[BARRIER_1]] -> i32
# CHECK:           %[[BARRIER_3:.*]] = nvvm.barrier #nvvm.reduction<popc> %[[BARRIER_2]] -> i32
# CHECK:           nvvm.barrier
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


@constructAndPrintInModule
def test_vote_sync_infer_type():
    i1 = IntegerType.get_signless(1)
    i32 = T.i32()

    @func.FuncOp.from_py_func(i32, i1)
    def vote_sync_ops(mask, pred):
        ballot_res = nvvm.vote_sync(mask, pred, nvvm.VoteSyncKind.ballot)
        any_res = nvvm.vote_sync(mask, pred, nvvm.VoteSyncKind.any)
        all_res = nvvm.vote_sync(mask, pred, nvvm.VoteSyncKind.all)
        uni_res = nvvm.vote_sync(mask, pred, nvvm.VoteSyncKind.uni)
        return ballot_res


# CHECK-LABEL:   func.func @vote_sync_ops(
# CHECK-SAME:      %[[MASK:.*]]: i32, %[[PRED:.*]]: i1) -> i32 {
# CHECK:           %[[BALLOT:.*]] = nvvm.vote.sync ballot %[[MASK]], %[[PRED]] -> i32
# CHECK:           %[[ANY:.*]] = nvvm.vote.sync any %[[MASK]], %[[PRED]] -> i1
# CHECK:           %[[ALL:.*]] = nvvm.vote.sync all %[[MASK]], %[[PRED]] -> i1
# CHECK:           %[[UNI:.*]] = nvvm.vote.sync uni %[[MASK]], %[[PRED]] -> i1
# CHECK:           return %[[BALLOT]] : i32


@constructAndPrintInModule
def test_clusterlaunchcontrol_query_cancel_infer_type():
    i128 = IntegerType.get_signless(128)

    @func.FuncOp.from_py_func(i128)
    def query_cancel_ops(response):
        is_canceled = nvvm.clusterlaunchcontrol_query_cancel(
            nvvm.ClusterLaunchControlQueryType.IS_CANCELED,
            response,
        )
        cta_x = nvvm.clusterlaunchcontrol_query_cancel(
            nvvm.ClusterLaunchControlQueryType.GET_FIRST_CTA_ID_X,
            response,
        )
        return cta_x


# CHECK-LABEL:   func.func @query_cancel_ops(
# CHECK-SAME:      %[[RESPONSE:.*]]: i128) -> i32 {
# CHECK:           %{{.*}} = nvvm.clusterlaunchcontrol.query.cancel query = is_canceled, %[[RESPONSE]] : i1
# CHECK:           %[[CTA_X:.*]] = nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_x, %[[RESPONSE]] : i32
# CHECK:           return %[[CTA_X]] : i32


@constructAndPrintInModule
def test_match_sync_infer_type():
    i32 = T.i32()
    i64 = IntegerType.get_signless(64)

    @func.FuncOp.from_py_func(i32, i32, i64)
    def match_sync_ops(mask, i32val, i64val):
        any_result = nvvm.match_sync(mask, i32val, nvvm.MatchSyncKind.any)
        all_result = nvvm.match_sync(mask, i32val, nvvm.MatchSyncKind.all)
        return any_result


# CHECK-LABEL:   func.func @match_sync_ops(
# CHECK-SAME:      %[[MASK:.*]]: i32, %[[I32VAL:.*]]: i32, %[[I64VAL:.*]]: i64) -> i32 {
# CHECK:           %[[ANY:.*]] = nvvm.match.sync any %[[MASK]], %[[I32VAL]] : i32 -> i32
# CHECK:           %[[ALL:.*]] = nvvm.match.sync all %[[MASK]], %[[I32VAL]] : i32 -> !llvm.struct<(i32, i1)>
# CHECK:           return %[[ANY]] : i32


@constructAndPrintInModule
def test_shfl_sync_infer_type():
    i32 = T.i32()
    f32 = T.f32()

    @func.FuncOp.from_py_func(i32, i32, f32, i32, i32)
    def shfl_sync_ops(mask, i32val, f32val, offset, clamp):
        i32_result = nvvm.shfl_sync(mask, i32val, offset, clamp, nvvm.ShflKind.bfly)
        f32_result = nvvm.shfl_sync(mask, f32val, offset, clamp, nvvm.ShflKind.bfly)
        struct_result = nvvm.shfl_sync(
            mask,
            i32val,
            offset,
            clamp,
            nvvm.ShflKind.bfly,
            return_value_and_is_valid=True,
        )
        return i32_result


# CHECK-LABEL:   func.func @shfl_sync_ops(
# CHECK-SAME:      %[[MASK:.*]]: i32, %[[I32VAL:.*]]: i32, %[[F32VAL:.*]]: f32, %[[OFF:.*]]: i32, %[[CLAMP:.*]]: i32) -> i32 {
# CHECK:           %[[I32R:.*]] = nvvm.shfl.sync bfly %[[MASK]], %[[I32VAL]], %[[OFF]], %[[CLAMP]] : i32 -> i32
# CHECK:           %[[F32R:.*]] = nvvm.shfl.sync bfly %[[MASK]], %[[F32VAL]], %[[OFF]], %[[CLAMP]] : f32 -> f32
# CHECK:           %[[STRUCT:.*]] = nvvm.shfl.sync bfly %[[MASK]], %[[I32VAL]], %[[OFF]], %[[CLAMP]] {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
# CHECK:           return %[[I32R]] : i32


@constructAndPrintInModule
def test_ldmatrix_infer_type():
    ptr_shared = llvm.PointerType.get(3)

    shape_8x8 = Attribute.parse("#nvvm.ld_st_matrix_shape<m = 8, n = 8>")
    elt_b16 = Attribute.parse("#nvvm.ld_st_matrix_elt_type<b16>")

    @func.FuncOp.from_py_func(ptr_shared)
    def ldmatrix_ops(ptr):
        r1 = nvvm.ldmatrix(
            ptr,
            num=1,
            layout=nvvm.MMALayout.row,
            shape=shape_8x8,
            elt_type=elt_b16,
        )
        r4 = nvvm.ldmatrix(
            ptr,
            num=4,
            layout=nvvm.MMALayout.row,
            shape=shape_8x8,
            elt_type=elt_b16,
        )
        return r1


# CHECK-LABEL:   func.func @ldmatrix_ops(
# CHECK-SAME:      %[[PTR:.*]]: !llvm.ptr<3>) -> i32 {
# CHECK:           %[[R1:.*]] = nvvm.ldmatrix %[[PTR]] {{.*}}num = 1{{.*}} : (!llvm.ptr<3>) -> i32
# CHECK:           %[[R4:.*]] = nvvm.ldmatrix %[[PTR]] {{.*}}num = 4{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
# CHECK:           return %[[R1]] : i32


@constructAndPrintInModule
def test_reductions():
    i32 = T.i32()
    f32 = T.f32()

    @func.FuncOp.from_py_func(i32, i32, f32)
    def reductions(mask, vi32, vf32):
        for abs in (True, False):
            for nan in (True, False):
                for kind in (
                    nvvm.ReductionKind.AND,
                    nvvm.ReductionKind.MAX,
                    nvvm.ReductionKind.MIN,
                    nvvm.ReductionKind.OR,
                    nvvm.ReductionKind.UMAX,
                    nvvm.ReductionKind.UMIN,
                    nvvm.ReductionKind.XOR,
                ):
                    nvvm.redux_sync(vi32, kind, vi32)

                for kind in (
                    nvvm.ReductionKind.FMIN,
                    nvvm.ReductionKind.FMAX,
                ):
                    nvvm.redux_sync(vf32, kind, vi32, abs=abs, nan=nan)


# CHECK-LABEL:   func.func @reductions(
# CHECK:           %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: f32) {
# CHECK:           %[[REDUX_0:.*]] = nvvm.redux.sync and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_1:.*]] = nvvm.redux.sync max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_2:.*]] = nvvm.redux.sync min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_3:.*]] = nvvm.redux.sync or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_4:.*]] = nvvm.redux.sync umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_5:.*]] = nvvm.redux.sync umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_6:.*]] = nvvm.redux.sync xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_7:.*]] = nvvm.redux.sync fmin %[[ARG2]], %[[ARG1]] {abs = true, nan = true} : f32 -> f32
# CHECK:           %[[REDUX_8:.*]] = nvvm.redux.sync fmax %[[ARG2]], %[[ARG1]] {abs = true, nan = true} : f32 -> f32
# CHECK:           %[[REDUX_9:.*]] = nvvm.redux.sync and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_10:.*]] = nvvm.redux.sync max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_11:.*]] = nvvm.redux.sync min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_12:.*]] = nvvm.redux.sync or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_13:.*]] = nvvm.redux.sync umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_14:.*]] = nvvm.redux.sync umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_15:.*]] = nvvm.redux.sync xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_16:.*]] = nvvm.redux.sync fmin %[[ARG2]], %[[ARG1]] {abs = true} : f32 -> f32
# CHECK:           %[[REDUX_17:.*]] = nvvm.redux.sync fmax %[[ARG2]], %[[ARG1]] {abs = true} : f32 -> f32
# CHECK:           %[[REDUX_18:.*]] = nvvm.redux.sync and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_19:.*]] = nvvm.redux.sync max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_20:.*]] = nvvm.redux.sync min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_21:.*]] = nvvm.redux.sync or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_22:.*]] = nvvm.redux.sync umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_23:.*]] = nvvm.redux.sync umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_24:.*]] = nvvm.redux.sync xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_25:.*]] = nvvm.redux.sync fmin %[[ARG2]], %[[ARG1]] {nan = true} : f32 -> f32
# CHECK:           %[[REDUX_26:.*]] = nvvm.redux.sync fmax %[[ARG2]], %[[ARG1]] {nan = true} : f32 -> f32
# CHECK:           %[[REDUX_27:.*]] = nvvm.redux.sync and %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_28:.*]] = nvvm.redux.sync max %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_29:.*]] = nvvm.redux.sync min %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_30:.*]] = nvvm.redux.sync or %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_31:.*]] = nvvm.redux.sync umax %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_32:.*]] = nvvm.redux.sync umin %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_33:.*]] = nvvm.redux.sync xor %[[ARG1]], %[[ARG1]] : i32 -> i32
# CHECK:           %[[REDUX_34:.*]] = nvvm.redux.sync fmin %[[ARG2]], %[[ARG1]] : f32 -> f32
# CHECK:           %[[REDUX_35:.*]] = nvvm.redux.sync fmax %[[ARG2]], %[[ARG1]] : f32 -> f32
# CHECK:           return
# CHECK:         }


# CHECK-LABEL: TEST: testSpecialRegisterInferredResults
@constructAndPrintInModule
def testSpecialRegisterInferredResults():
    # CHECK: %{{.*}} = nvvm.read.ptx.sreg.tid.x : i32
    nvvm.ThreadIdXOp()
    # CHECK: %{{.*}} = nvvm.read.ptx.sreg.clock : i32
    nvvm.ClockOp()
    # CHECK: %{{.*}} = nvvm.read.ptx.sreg.clock64 : i64
    nvvm.Clock64Op()
    # CHECK: %{{.*}} = nvvm.read.ptx.sreg.globaltimer : i64
    nvvm.GlobalTimerOp()
