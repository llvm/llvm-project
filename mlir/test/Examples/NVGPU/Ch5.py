# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN: sh -c 'if [[ "%mlir_run_cuda_sm90_tests" == "1" ]]; \
# RUN: then %PYTHON %s | FileCheck %s; \
# RUN: else export MLIR_NVDSL_PRINT_IR=1; \
# RUN: %PYTHON %s | FileCheck %s --check-prefix=DUMPIR; fi'


# ===----------------------------------------------------------------------===//
#  Chapter 5 : Warp Specialized GEMM with Tensor Core
# ===----------------------------------------------------------------------===//
#
# This program demonstrates a GEMM operation for `f32+=f16*f16`, utilizing the
# Warp Specialized method with a tile size of 128x128x64. The code completely
# parallelizes the two outermost loops into thread blocks. It launches two Warp
# Groups (256 threads in total): one for the producer and the other for the consumer.
# Each group takes a different control-flow. The producer thread group is responsible
# for loading data into shared memory, while the consumer group executes the Tensor
# Core GEMM operation and epilogue.
#
#  for ti in range(M//128):  # -> blockIdx.x
#   for tj in range(N//128): # -> blockIdx.y
#    with wg_producer:
#     for tk in range(K//64):
#        TMA_128x64_64x128...
#    with wg_consumer:
#     for tk in range(K//64):
#        MMA_128x128x64...
#     Epilogue..
#
# This chapter demonstrates:
#  2 WG (warpgroups)
#    Producer:
#       2.1.1 Wait MMA Barrier
#       2.1.1 Load TMA with TMA barrier
#       2.1.1 Arrive TMA barrier with txcount
#    Consumer:
#       Loop
#           Wait TMA barrier
#           Performs Tensor Core GEMM 64x128x64 by warpgroup
#           Arrive MMA Barrier
#       Epilogue
#           Store fragmented registers to shared memory
#           Store shared memory to global
#
# ===----------------------------------------------------------------------===//


from mlir import ir
from mlir.dialects import gpu, scf, nvgpu, nvvm
from mlir.extras import types as T
from tools.nvdsl import *
import numpy as np


def partition_shape():
    """
    Calculate the partition shape based on the block IDs.

    It parallelizes the two outermost loops into thread blocks.
    for ti in range(M//128):    # -> blockIdx.x
     for tj in range(N//128):   # -> blockIdx.y
      D = 0
      for tk in range(K//64):
       for i in range(128):
        for j in range(128):
         for k in range(64):
           FMA

    Returns:
        dimX (int): Dimension along the x-axis.
        dimY (int): Dimension along the y-axis.
    """
    bidx = gpu.block_id(gpu.Dimension.x)
    bidy = gpu.block_id(gpu.Dimension.y)
    dimX = bidx * TILE_M
    dimY = bidy * TILE_N
    return dimX, dimY


def tma_load(
    mbar_group: Mbarriers,
    a_tma: TMA,
    b_tma: TMA,
    slot,
    stage,
    num_stages,
    p=None,
):
    """
    TMA loads two input matrices from global memory to shared memory. It performs the following operations:

       - tma.load a_shared_memory[off_x]  at coordinate [x, z]      (Loads 128x64)
       - tma.load b_shared_memory[off_y1] at coordinate [y, x]      (Loads 64x64)
       - tma.load b_shared_memory[off_y2] at coordinate [y + 64, x] (Loads 64x64)

       mbarrier.arrive ta_count = 128x64x2x4
    """
    dimX, dimY = partition_shape()

    tidx = gpu.thread_id(gpu.Dimension.x)
    begin_b = num_stages * get_type_size(a_tma.tma_memref)
    size_tma_a = get_type_size(a_tma.tma_memref)
    size_tma_b = get_type_size(b_tma.tma_memref)
    ta_count = size_tma_a + (size_tma_b * 2)

    off_a = slot * size_tma_a
    off_b = (slot * size_tma_a) + begin_b
    off_b2 = off_b + size_tma_b
    a_elem_ty = a_tma.tma_memref.element_type
    b_elem_ty = b_tma.tma_memref.element_type
    a = get_dynamic_shared_memory(a_tma.tma_memref.shape, a_elem_ty, off_a)
    b1 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b)
    b2 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b2)

    mbar_group[slot].arrive(ta_count, predicate=p)
    p = (tidx % WARP_GROUP_SIZE) == 0
    c1 = stage * 64
    a_tma.load(a, mbar_group[slot], coords=[c1, dimX], predicate=p)
    b_tma.load(b1, mbar_group[slot], coords=[dimY, c1], predicate=p)
    b_tma.load(b2, mbar_group[slot], coords=[dimY + 64, c1], predicate=p)


def initialize(a_tma: TMA, b_tma: TMA, num_stages):
    """
    Initialize mbarriers and prefetch TMA descriptors.
    """
    tidx = gpu.thread_id(gpu.Dimension.x)
    mbar_group_tma = Mbarriers(number_of_barriers=num_stages)
    mbar_group_mma = Mbarriers(number_of_barriers=num_stages)
    isThread0 = tidx == const(0)
    with ir.InsertionPoint(scf.IfOp(isThread0).then_block):
        for i in scf.for_(0, num_stages, 1):
            mbar_group_tma[i].init(1)
            mbar_group_mma[i].init(1)
            scf.yield_([])
        a_tma.prefetch()
        b_tma.prefetch()
        scf.yield_([])

    return mbar_group_tma, mbar_group_mma


def switch_phase(stage, phase, num_stages):
    p = stage == (num_stages - 1)
    phase = arith.select(
        p,
        (phase ^ const(True, ty=T.bool())),
        phase,
    )
    return phase


def producer_loop(
    mbar_tma: Mbarriers,
    mbar_mma: Mbarriers,
    a_tma: TMA,
    b_tma: TMA,
    wg_me: Warpgroup,
    num_stages,
):
    phase = const(True, ty=T.bool())

    for iv, phase, _ in scf.for_(0, (K // TILE_K), 1, [phase]):
        stage = iv % num_stages
        # Wait MMA to be done
        mbar_mma[stage].try_wait(phase)
        # New phase for mbarrier
        phase = switch_phase(stage, phase, num_stages)
        # TMA Load
        tma_load(mbar_tma, a_tma, b_tma, stage, iv, num_stages, wg_me.is_wg_primary)
        scf.yield_([phase])


def consumer_loop(
    mbar_tma: Mbarriers,
    mbar_mma: Mbarriers,
    a_tma: TMA,
    b_tma: TMA,
    wg_me: Warpgroup,
    num_stages,
):
    begin_b = num_stages * get_type_size(a_tma.tma_memref)

    size_a = TILE_M * TILE_K * get_type_size(T.f16())

    phase = const(False, ty=T.bool())
    A = WGMMAMatrix(WGMMAType.Descriptor, [TILE_M, TILE_K], desc=a_tma)
    B = WGMMAMatrix(WGMMAType.Descriptor, [TILE_K, TILE_N], desc=b_tma)
    D = WGMMAMatrix(WGMMAType.Accumulator, shape=[TILE_M, TILE_N], ty=T.f32())

    for_op = scf.ForOp(const(0), const(K // TILE_K), const(1), [D.acc_op, phase])
    with ir.InsertionPoint(for_op.body):
        phase = for_op.inner_iter_args[1]
        iv = for_op.induction_variable
        stage = iv % num_stages

        # Wait TMA for current stage
        mbar_tma[stage].try_wait(phase)

        # Find shared memory slot
        offset_a = stage * size_a
        offset_b = offset_a + begin_b
        a_smem = get_dynamic_shared_memory([TILE_M, TILE_K], T.f16(), offset_a)
        b_smem = get_dynamic_shared_memory([TILE_K, TILE_N], T.f16(), offset_b)

        # Iterate input matrices, update accumulator
        A.update_smem(a_smem)
        B.update_smem(b_smem)
        D.update_accumulator(for_op.inner_iter_args[0])

        # Matrix Multiply
        D += A @ B

        # MMA Barrier Arrive
        p_arrive = (iv > 0) & wg_me.is_wg_primary
        with ir.InsertionPoint(scf.IfOp(p_arrive).then_block):
            barId = arith.select((stage == 0), const(num_stages - 1), (stage - 1))
            mbar_mma[barId].arrive()
            scf.yield_([])

        phase = switch_phase(stage, phase, num_stages)
        scf.yield_([D.acc_op, phase])

    nvvm.WgmmaWaitGroupSyncOp(0)
    D.update_accumulator(for_op.results[0])
    return D


def epilogue(D: WGMMAMatrix, d_dev):
    """
    Epilogue of the GEMM kernel. It stores the fragmented registers to global memory.

    MatrixAccumulator D               # Fragmented results
    store D -> Shared Memory          # Store Shared Memory
    Shared Memory -> Z[dimX][dimY]    # Store Shared Memory to Global Memory

    """
    tidx = gpu.thread_id(gpu.Dimension.x)
    dimX, dimY = partition_shape()
    # s = tidx - WARP_GROUP_SIZE
    # debug_print("[Epilogue] store to global memory @ s={}", s)

    d_smem = get_dynamic_shared_memory([TILE_M, TILE_N], T.f32())
    d_gmem = memref.subview(d_dev, [dimX, dimY], [TILE_M, TILE_N], [1, 1])

    # Store (registers -> shared memory)
    D.store_accumulator(d_smem)
    gpu.barrier()

    # Store (shared memory --> global memory)
    for i in scf.for_(0, TILE_M, 1):
        val = memref.load(d_smem, [i, tidx])
        memref.store(val, d_gmem, [i, tidx])
        scf.yield_([])


@NVDSL.mlir_func
def gemm_warp_specialized(a, b, d, num_stages):
    token_ty = gpu.AsyncTokenType.get()
    t1 = gpu.wait([])
    a_dev, t2 = gpu.alloc(a.type, token_ty, [t1], [], [])
    b_dev, t3 = gpu.alloc(b.type, token_ty, [t2], [], [])
    d_dev, t4 = gpu.alloc(d.type, token_ty, [t3], [], [])
    t5 = gpu.memcpy(token_ty, [t4], a_dev, a)
    t6 = gpu.memcpy(token_ty, [t5], b_dev, b)
    t7 = gpu.wait([t6])

    sw = nvgpu.TensorMapSwizzleKind.SWIZZLE_128B
    a_tma = TMA([128, 64], a.type, swizzle=sw)
    b_tma = TMA([64, 64], b.type, swizzle=sw)
    a_tma.create_descriptor(a_dev)
    b_tma.create_descriptor(b_dev)

    grid = [(M // TILE_M), (N // TILE_N), 1]
    block = [256, 1, 1]

    size_a = get_type_size(a.type.element_type) * TILE_M * TILE_K
    size_b = get_type_size(b.type.element_type) * TILE_N * TILE_K
    smem_size_in_bytes = (size_a + size_b) * num_stages

    @NVDSL.mlir_gpu_launch(grid=grid, block=block, smem=smem_size_in_bytes)
    def gemm_warp_specialized_kernel():
        # Init Warpgroups
        wg_producer = Warpgroup(primary_thread=128, register_size=40)
        wg_consumer = Warpgroup(primary_thread=0, register_size=232)

        # Initialize mbarriers and prefetch TMA descriptors
        mbar_mma, mbar_tma = initialize(a_tma, b_tma, num_stages)

        # Producer performs TMA
        with wg_producer:
            producer_loop(mbar_tma, mbar_mma, a_tma, b_tma, wg_producer, num_stages)

        # Consumer performs MMA/Tensor Core
        with wg_consumer:
            D = consumer_loop(mbar_tma, mbar_mma, a_tma, b_tma, wg_consumer, num_stages)
            epilogue(D, d_dev)

    gemm_warp_specialized_kernel()

    t8 = gpu.memcpy(token_ty, [t7], d, d_dev)
    gpu.wait([t8])


# Python pass arguments to MLIR
N = 256
M = 512
K = 1024
TILE_M = 128
TILE_N = 128
TILE_K = 64
a = np.random.randn(M, K).astype(np.float16)
b = np.random.randn(K, N).astype(np.float16)
d = np.zeros((M, N), np.float32)

gemm_warp_specialized(a, b, d, num_stages=7)

if os.getenv("MLIR_NVDSL_PRINT_IR") != "1":
    # Verify MLIR with reference computation
    ref_d = a.astype(np.float16) @ b.astype(np.float16)
    np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)

    print("PASS")
# CHECK-NOT: Mismatched elements
# CHECK: PASS

# DUMPIR:       %[[TID_X:.*]] = gpu.thread_id  x
# DUMPIR:       %[[C128:.*]] = arith.constant 128 : index
# DUMPIR:       %[[REM1:.*]] = arith.remui %[[TID_X]], %[[C128]] : index
# DUMPIR:       %[[C0:.*]] = arith.constant 0 : index
# DUMPIR:       %[[IS_PRIMARY:.*]] = arith.cmpi eq, %[[REM1]], %[[C0]] : index
# DUMPIR:       %[[C128_1:.*]] = arith.constant 128 : index
# DUMPIR:       %[[DIV1:.*]] = arith.divui %[[TID_X]], %[[C128_1]] : index
# DUMPIR:       %[[C1:.*]] = arith.constant 1 : index
# DUMPIR:       %[[IS_PRODUCER:.*]] = arith.cmpi eq, %[[DIV1]], %[[C1]] : index
# DUMPIR:       %[[TID_X_2:.*]] = gpu.thread_id  x
# DUMPIR:       %[[C128_2:.*]] = arith.constant 128 : index
# DUMPIR:       %[[REM2:.*]] = arith.remui %[[TID_X_2]], %[[C128_2]] : index
# DUMPIR:       %[[C0_2:.*]] = arith.constant 0 : index
# DUMPIR:       %[[IS_PRIMARY_2:.*]] = arith.cmpi eq, %[[REM2]], %[[C0_2]] : index
# DUMPIR:       %[[C128_3:.*]] = arith.constant 128 : index
# DUMPIR:       %[[DIV2:.*]] = arith.divui %[[TID_X_2]], %[[C128_3]] : index
# DUMPIR:       %[[C0_3:.*]] = arith.constant 0 : index
# DUMPIR:       %[[IS_CONSUMER:.*]] = arith.cmpi eq, %[[DIV2]], %[[C0_3]] : index
# DUMPIR:       %[[TID_X_3:.*]] = gpu.thread_id  x
# DUMPIR:       %[[MBAR_MMA:.*]] = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:       %[[MBAR_TMA:.*]] = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:       %[[C0_4:.*]] = arith.constant 0 : index
# DUMPIR:       %[[IS_THREAD0:.*]] = arith.cmpi eq, %[[TID_X_3]], %[[C0_4]] : index
# DUMPIR:       scf.if %[[IS_THREAD0]] {
# DUMPIR:         %[[C0_INIT:.*]] = arith.constant 0 : index
# DUMPIR:         %[[C7:.*]] = arith.constant 7 : index
# DUMPIR:         %[[C1_INIT:.*]] = arith.constant 1 : index
# DUMPIR:         scf.for %arg15 = %[[C0_INIT]] to %[[C7]] step %[[C1_INIT]] {
# DUMPIR:           %[[C1_INIT_VAL:.*]] = arith.constant 1 : index
# DUMPIR:           nvgpu.mbarrier.init %[[MBAR_MMA]][%arg15], %[[C1_INIT_VAL]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:           %[[C1_INIT_VAL_2:.*]] = arith.constant 1 : index
# DUMPIR:           nvgpu.mbarrier.init %[[MBAR_TMA]][%arg15], %[[C1_INIT_VAL_2]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:         }
# DUMPIR:         nvgpu.tma.prefetch.descriptor %{{.*}} : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:         nvgpu.tma.prefetch.descriptor %{{.*}} : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:       }
# DUMPIR:       scf.if %[[IS_PRODUCER]] {
# DUMPIR:         nvvm.setmaxregister  decrease 40
# DUMPIR:         %[[TRUE:.*]] = arith.constant true
# DUMPIR:         %[[C0_PROD:.*]] = arith.constant 0 : index
# DUMPIR:         %[[C16:.*]] = arith.constant 16 : index
# DUMPIR:         %[[C1_PROD:.*]] = arith.constant 1 : index
# DUMPIR:         %[[PROD_LOOP:.*]] = scf.for %arg15 = %[[C0_PROD]] to %[[C16]] step %[[C1_PROD]] iter_args(%arg16 = %[[TRUE]]) -> (i1) {
# DUMPIR:           %[[C7_PROD:.*]] = arith.constant 7 : index
# DUMPIR:           %[[SLOT:.*]] = arith.remui %arg15, %[[C7_PROD]] : index
# DUMPIR:           %[[TIMEOUT:.*]] = arith.constant 10000000 : index
# DUMPIR:           nvgpu.mbarrier.try_wait.parity %[[MBAR_MMA]][%[[SLOT]]], %arg16, %[[TIMEOUT]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:           %[[C6:.*]] = arith.constant 6 : index
# DUMPIR:           %[[IS_LAST:.*]] = arith.cmpi eq, %[[SLOT]], %[[C6]] : index
# DUMPIR:           %[[TRUE_2:.*]] = arith.constant true
# DUMPIR:           %[[FLIP:.*]] = arith.xori %arg16, %[[TRUE_2]] : i1
# DUMPIR:           %[[PHASE:.*]] = arith.select %[[IS_LAST]], %[[FLIP]], %arg16 : i1
# DUMPIR:           %[[BID_X:.*]] = gpu.block_id  x
# DUMPIR:           %[[BID_Y:.*]] = gpu.block_id  y
# DUMPIR:           %[[C128_TILE:.*]] = arith.constant 128 : index
# DUMPIR:           %[[DIM_X:.*]] = arith.muli %[[BID_X]], %[[C128_TILE]] : index
# DUMPIR:           %[[C128_TILE_2:.*]] = arith.constant 128 : index
# DUMPIR:           %[[DIM_Y:.*]] = arith.muli %[[BID_Y]], %[[C128_TILE_2]] : index
# DUMPIR:           %[[TID_PROD:.*]] = gpu.thread_id  x
# DUMPIR:           %[[C16384:.*]] = arith.constant 16384 : index
# DUMPIR:           %[[OFF_A:.*]] = arith.muli %[[SLOT]], %[[C16384]] : index
# DUMPIR:           %[[C16384_2:.*]] = arith.constant 16384 : index
# DUMPIR:           %[[OFF_B_BASE:.*]] = arith.muli %[[SLOT]], %[[C16384_2]] : index
# DUMPIR:           %[[C114688:.*]] = arith.constant 114688 : index
# DUMPIR:           %[[OFF_B1:.*]] = arith.addi %[[OFF_B_BASE]], %[[C114688]] : index
# DUMPIR:           %[[C8192:.*]] = arith.constant 8192 : index
# DUMPIR:           %[[OFF_B2:.*]] = arith.addi %[[OFF_B1]], %[[C8192]] : index
# DUMPIR:           %[[SMEM:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:           %[[VIEW_A:.*]] = memref.view %[[SMEM]][%[[OFF_A]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:           %[[SMEM_2:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:           %[[VIEW_B1:.*]] = memref.view %[[SMEM_2]][%[[OFF_B1]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:           %[[SMEM_3:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:           %[[VIEW_B2:.*]] = memref.view %[[SMEM_3]][%[[OFF_B2]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:           %[[TX_COUNT:.*]] = arith.constant 32768 : index
# DUMPIR:           nvgpu.mbarrier.arrive.expect_tx %[[MBAR_TMA]][%[[SLOT]]], %[[TX_COUNT]], predicate = %[[IS_PRIMARY]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:           %[[C128_WG:.*]] = arith.constant 128 : index
# DUMPIR:           %[[TID_MOD:.*]] = arith.remui %[[TID_PROD]], %[[C128_WG]] : index
# DUMPIR:           %[[C0_TMA:.*]] = arith.constant 0 : index
# DUMPIR:           %[[IS_TMA_THREAD:.*]] = arith.cmpi eq, %[[TID_MOD]], %[[C0_TMA]] : index
# DUMPIR:           %[[C64:.*]] = arith.constant 64 : index
# DUMPIR:           %[[K_COORD:.*]] = arith.muli %arg15, %[[C64]] : index
# DUMPIR:           nvgpu.tma.async.load %{{.*}}[%[[K_COORD]], %[[DIM_X]]], %[[MBAR_TMA]][%[[SLOT]]] to %[[VIEW_A]], predicate = %[[IS_TMA_THREAD]] : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:           nvgpu.tma.async.load %{{.*}}[%[[DIM_Y]], %[[K_COORD]]], %[[MBAR_TMA]][%[[SLOT]]] to %[[VIEW_B1]], predicate = %[[IS_TMA_THREAD]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:           %[[C64_OFF:.*]] = arith.constant 64 : index
# DUMPIR:           %[[DIM_Y_OFF:.*]] = arith.addi %[[DIM_Y]], %[[C64_OFF]] : index
# DUMPIR:           nvgpu.tma.async.load %{{.*}}[%[[DIM_Y_OFF]], %[[K_COORD]]], %[[MBAR_TMA]][%[[SLOT]]] to %[[VIEW_B2]], predicate = %[[IS_TMA_THREAD]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:           scf.yield %[[PHASE]] : i1
# DUMPIR:         }
# DUMPIR:       }
# DUMPIR:       scf.if %[[IS_CONSUMER]] {
# DUMPIR:         nvvm.setmaxregister  increase 232
# DUMPIR:         %[[FALSE:.*]] = arith.constant false
# DUMPIR:         %[[ACC_INIT:.*]] = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
# DUMPIR:         %[[C0_CONS:.*]] = arith.constant 0 : index
# DUMPIR:         %[[C16_CONS:.*]] = arith.constant 16 : index
# DUMPIR:         %[[C1_CONS:.*]] = arith.constant 1 : index
# DUMPIR:         %[[CONS_LOOP:.*]]:2 = scf.for %arg15 = %[[C0_CONS]] to %[[C16_CONS]] step %[[C1_CONS]] iter_args(%arg16 = %[[ACC_INIT]], %arg17 = %[[FALSE]]) -> (!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1) {
# DUMPIR:           %[[C7_CONS:.*]] = arith.constant 7 : index
# DUMPIR:           %[[SLOT_CONS:.*]] = arith.remui %arg15, %[[C7_CONS]] : index
# DUMPIR:           %[[TIMEOUT_CONS:.*]] = arith.constant 10000000 : index
# DUMPIR:           nvgpu.mbarrier.try_wait.parity %[[MBAR_TMA]][%[[SLOT_CONS]]], %arg17, %[[TIMEOUT_CONS]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:           %[[C16384_CONS:.*]] = arith.constant 16384 : index
# DUMPIR:           %[[OFF_A_CONS:.*]] = arith.muli %[[SLOT_CONS]], %[[C16384_CONS]] : index
# DUMPIR:           %[[C114688_CONS:.*]] = arith.constant 114688 : index
# DUMPIR:           %[[OFF_B_CONS:.*]] = arith.addi %[[OFF_A_CONS]], %[[C114688_CONS]] : index
# DUMPIR:           %[[SMEM_CONS:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:           %[[VIEW_A_CONS:.*]] = memref.view %[[SMEM_CONS]][%[[OFF_A_CONS]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:           %[[SMEM_CONS_2:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:           %[[VIEW_B_CONS:.*]] = memref.view %[[SMEM_CONS_2]][%[[OFF_B_CONS]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
# DUMPIR:           %[[DESC_A:.*]] = nvgpu.warpgroup.generate.descriptor %[[VIEW_A_CONS]], %{{.*}} : memref<128x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>
# DUMPIR:           %[[DESC_B:.*]] = nvgpu.warpgroup.generate.descriptor %[[VIEW_B_CONS]], %{{.*}} : memref<64x128xf16, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>
# DUMPIR:           %[[ACC:.*]] = nvgpu.warpgroup.mma %[[DESC_A]], %[[DESC_B]], %arg16 {transposeB} : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
# DUMPIR:           %[[C0_CMP:.*]] = arith.constant 0 : index
# DUMPIR:           %[[IS_NOT_FIRST:.*]] = arith.cmpi ugt, %arg15, %[[C0_CMP]] : index
# DUMPIR:           %[[ARRIVE_PRED:.*]] = arith.andi %[[IS_NOT_FIRST]], %[[IS_PRIMARY_2]] : i1
# DUMPIR:           scf.if %[[ARRIVE_PRED]] {
# DUMPIR:             %[[C0_ARR:.*]] = arith.constant 0 : index
# DUMPIR:             %[[IS_ZERO:.*]] = arith.cmpi eq, %[[SLOT_CONS]], %[[C0_ARR]] : index
# DUMPIR:             %[[C6_WRAP:.*]] = arith.constant 6 : index
# DUMPIR:             %[[C1_SUB:.*]] = arith.constant 1 : index
# DUMPIR:             %[[PREV_SLOT:.*]] = arith.subi %[[SLOT_CONS]], %[[C1_SUB]] : index
# DUMPIR:             %[[BARR_ID:.*]] = arith.select %[[IS_ZERO]], %[[C6_WRAP]], %[[PREV_SLOT]] : index
# DUMPIR:             %{{.*}} = nvgpu.mbarrier.arrive %[[MBAR_MMA]][%[[BARR_ID]]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> !nvgpu.mbarrier.token
# DUMPIR:           }
# DUMPIR:           %[[C6_LAST:.*]] = arith.constant 6 : index
# DUMPIR:           %[[IS_LAST_CONS:.*]] = arith.cmpi eq, %[[SLOT_CONS]], %[[C6_LAST]] : index
# DUMPIR:           %[[TRUE_CONS:.*]] = arith.constant true
# DUMPIR:           %[[FLIP_CONS:.*]] = arith.xori %arg17, %[[TRUE_CONS]] : i1
# DUMPIR:           %[[PHASE_CONS:.*]] = arith.select %[[IS_LAST_CONS]], %[[FLIP_CONS]], %arg17 : i1
# DUMPIR:           scf.yield %[[ACC]], %[[PHASE_CONS]] : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1
# DUMPIR:         }
# DUMPIR:         nvvm.wgmma.wait.group.sync.aligned 0
# DUMPIR:         %[[TID_EPI:.*]] = gpu.thread_id  x
# DUMPIR:         %[[BID_X_EPI:.*]] = gpu.block_id  x
# DUMPIR:         %[[BID_Y_EPI:.*]] = gpu.block_id  y
# DUMPIR:         %[[C128_EPI:.*]] = arith.constant 128 : index
# DUMPIR:         %[[DIM_X_EPI:.*]] = arith.muli %[[BID_X_EPI]], %[[C128_EPI]] : index
# DUMPIR:         %[[C128_EPI_2:.*]] = arith.constant 128 : index
# DUMPIR:         %[[DIM_Y_EPI:.*]] = arith.muli %[[BID_Y_EPI]], %[[C128_EPI_2]] : index
# DUMPIR:         %[[SMEM_EPI:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[C0_EPI:.*]] = arith.constant 0 : index
# DUMPIR:         %[[VIEW_EPI:.*]] = memref.view %[[SMEM_EPI]][%[[C0_EPI]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x128xf32, #gpu.address_space<workgroup>>
# DUMPIR:         %[[SUBVIEW:.*]] = memref.subview %{{.*}}[%[[DIM_X_EPI]], %[[DIM_Y_EPI]]] [128, 128] [1, 1] : memref<512x256xf32> to memref<128x128xf32, strided<[256, 1], offset: ?>>
# DUMPIR:         nvgpu.warpgroup.mma.store %[[CONS_LOOP]]#0, %[[VIEW_EPI]] : <fragmented = vector<128x128xf32>> to memref<128x128xf32, #gpu.address_space<workgroup>>
# DUMPIR:         gpu.barrier
# DUMPIR:         %[[C0_STORE:.*]] = arith.constant 0 : index
# DUMPIR:         %[[C128_STORE:.*]] = arith.constant 128 : index
# DUMPIR:         %[[C1_STORE:.*]] = arith.constant 1 : index
# DUMPIR:         scf.for %arg15 = %[[C0_STORE]] to %[[C128_STORE]] step %[[C1_STORE]] {
# DUMPIR:           %{{.*}} = memref.load %[[VIEW_EPI]][%arg15, %[[TID_EPI]]] : memref<128x128xf32, #gpu.address_space<workgroup>>
# DUMPIR:           memref.store %{{.*}}, %[[SUBVIEW]][%arg15, %[[TID_EPI]]] : memref<128x128xf32, strided<[256, 1], offset: ?>>
# DUMPIR:         }
# DUMPIR:       }
# DUMPIR:       gpu.terminator
