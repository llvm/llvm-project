# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

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

    for iv, phase in scf.for_(0, (K // TILE_K), 1, [phase]):
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
    t1 = gpu.wait(token_ty, [])
    a_dev, t2 = gpu.alloc(a.type, token_ty, [t1], [], [])
    b_dev, t3 = gpu.alloc(b.type, token_ty, [t2], [], [])
    d_dev, t4 = gpu.alloc(d.type, token_ty, [t3], [], [])
    t5 = gpu.memcpy(token_ty, [t4], a_dev, a)
    t6 = gpu.memcpy(token_ty, [t5], b_dev, b)
    t7 = gpu.wait(token_ty, [t6])

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
    gpu.wait(None, [t8])


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


# Verify MLIR with reference computation
ref_d = a.astype(np.float16) @ b.astype(np.float16)
np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)


print("PASS")
# CHECK-NOT: Mismatched elements
