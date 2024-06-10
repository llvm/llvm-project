# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 4 : Multistage GEMM with Tensor Core
# ===----------------------------------------------------------------------===//
#
# This program exemplifies a GEMM operation for `f32+=f16*f16`, utilizing the
# Multistage method with a tile size of 128x128x64. The code completely
# parallelizes the two outermost loops into thread blocks. It launches one Warp
# Groups (128 threads in total) and allocates multiple slots/stage in the
# shared memory. The program consists of three main parts: prologue, mainloop,
# and epilogue. In the prologue, thread0 requests for TMA to load data into
# shared memory slots. The mainloop executes MMA while simultaneously loading
# TMA for the utilized slots. This overlap of TMA and MMA operations enhances
# performance by maximizing computational throughput.
#
# Loops illustration:
#
#  for s in range(num_stages):
#    TMA_128x64_64x128...
#  for ti in range(M//128):  # -> blockIdx.x
#   for tj in range(N//128): # -> blockIdx.y
#    for tk in range(K//64):
#      MMA_128x128x64...
#      TMA_128x64_64x128...
#  Epilogue...
#
# This chapter introduces demonstrates:
#  1. Partition shape based on block IDs
#  2. Prologue
#    2.1 Execute TMA Load for two input matrices for each stage
#  3. Main loop
#    3.1 Wait for completion of TMA load with mbarrier
#    3.2 Performs Tensor Core GEMM 64x128x64 by warpgroup
#    3.3 Load next stage if needed
#  4. Epilogue
#    4.1 Store fragmented registers to shared memory
#    4.2 Store shared memory to global
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

    It partitions the shape like below:
    for(.. i < M ...)   --> blockIdx.x
     for(.. j < N ...)  --> blockIdx.y
      for(.. k < K ...)

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
    tidx = gpu.thread_id(gpu.Dimension.x)

    p = tidx == 0 if p is None else p

    off_a = slot * size_tma_a
    off_b = (slot * size_tma_a) + begin_b
    off_b2 = off_b + size_tma_b
    a_elem_ty = a_tma.tma_memref.element_type
    b_elem_ty = b_tma.tma_memref.element_type
    a = get_dynamic_shared_memory(a_tma.tma_memref.shape, a_elem_ty, off_a)
    b1 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b)
    b2 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b2)

    mbar_group[slot].arrive(ta_count, predicate=p)

    c1 = stage * 64
    a_tma.load(a, mbar_group[slot], coords=[c1, dimX], predicate=p)
    b_tma.load(b1, mbar_group[slot], coords=[dimY, c1], predicate=p)
    b_tma.load(b2, mbar_group[slot], coords=[dimY + 64, c1], predicate=p)


def initialize(a_tma: TMA, b_tma: TMA, num_stages):
    """
    Initialize mbarriers and prefetch TMA descriptors.
    """
    tidx = gpu.thread_id(gpu.Dimension.x)
    mbar_group = Mbarriers(number_of_barriers=num_stages)
    isThread0 = tidx == const(0)
    with ir.InsertionPoint(scf.IfOp(isThread0).then_block):
        for i in scf.for_(0, num_stages, 1):
            mbar_group[i].init(1)
            scf.yield_([])
        a_tma.prefetch()
        b_tma.prefetch()
        scf.yield_([])

    return mbar_group


def prologue(mbar_group: Mbarriers, a_tma: TMA, b_tma: TMA, num_stages):
    """
    Prologue of the GEMM kernel. It loads 2 input matrices for each stage in loop like below:

    for stage in range(NUM_STAGES):
        tma_load x, y, stage

    """
    ns = num_stages if num_stages == 1 else num_stages - 1
    for iv in scf.for_(0, ns, 1):
        tma_load(mbar_group, a_tma, b_tma, iv, iv, num_stages)
        scf.yield_([])


def mainloop(mbar_group: Mbarriers, a_tma: TMA, b_tma: TMA, num_stages):
    """
    Main loop of the Multistage GEMM kernel. It iterates through
    stages and performs matrix multiplication, loading data by TMA to shared memory. It like following

    MatrixAccumulator D
    for k in range(K // TILE_K):

        try_wait(stage, ...)    # Wait TMA load

        Matrix A(stage, ...)    # Find shared memory slot
        Matrix B(stage, ...)    # Find shared memory slot
        D += A @ B              # Multiply and accumulate

        if(needLoad)            # Load next stage if needed
            tma_load(x, y, nextSlot, nextStage)

    """
    ns = num_stages if num_stages == 1 else num_stages - 1

    tidx = gpu.thread_id(gpu.Dimension.x)
    begin_b = num_stages * get_type_size(a_tma.tma_memref)

    size_a = TILE_M * TILE_K * get_type_size(T.f16())

    # Initialize A and B (input matrices) and C (accumulator)
    A = WGMMAMatrix(WGMMAType.Descriptor, [TILE_M, TILE_K], desc=a_tma)
    B = WGMMAMatrix(WGMMAType.Descriptor, [TILE_K, TILE_N], desc=b_tma)
    D = WGMMAMatrix(WGMMAType.Accumulator, shape=[TILE_M, TILE_N], ty=T.f32())

    phase = const(False, ty=T.bool())

    # Main Loop
    for_op = scf.ForOp(const(0), const(K // TILE_K), const(1), [D.acc_op, phase])
    with ir.InsertionPoint(for_op.body):
        phase = for_op.inner_iter_args[1]
        iv = for_op.induction_variable
        stage = iv % num_stages

        # Wait for current stage
        mbar_group[stage].try_wait(phase=phase)

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

        # Wait Tensor Core for single stage
        if num_stages == 1:
            nvvm.WgmmaWaitGroupSyncOp(0)

        # Load next stage
        pred = ((iv + ns) < const(K // TILE_K)) & (tidx == 0)
        nextStage = iv + ns
        nextSlot = nextStage % num_stages
        tma_load(mbar_group, a_tma, b_tma, nextSlot, nextStage, num_stages, pred)

        # Switch phase parity for the mbarrier
        newPhase = arith.select(
            stage == (num_stages - 1),
            (phase ^ const(True, ty=T.bool())),
            phase,
        )
        scf.yield_([D.acc_op, newPhase])

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


# The decorator generates
#   a -> memref<MxKxf16>
#   b -> memref<NxKf16>
#   d -> memref<MxNxf32>
@NVDSL.mlir_func
def gemm_multistage(a, b, d, num_stages):
    token_ty = ir.Type.parse("!gpu.async.token")
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
    block = [128, 1, 1]

    size_a = get_type_size(a.type.element_type) * TILE_M * TILE_K
    size_b = get_type_size(b.type.element_type) * TILE_N * TILE_K
    smem_size_in_bytes = (size_a + size_b) * num_stages

    @NVDSL.mlir_gpu_launch(grid=grid, block=block, smem=smem_size_in_bytes)
    def gemm_multistage_kernel():
        # Initialize mbarriers and prefetch TMA descriptors
        mbar_group = initialize(a_tma, b_tma, num_stages)

        # Fill the pipeline stages
        prologue(mbar_group, a_tma, b_tma, num_stages)

        # Main loop
        D = mainloop(mbar_group, a_tma, b_tma, num_stages)

        # Store registers to global memory
        epilogue(D, d_dev)

    gemm_multistage_kernel()

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

gemm_multistage(a, b, d, num_stages=7)


# Verify MLIR with reference computation
ref_d = a.astype(np.float16) @ b.astype(np.float16)
np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)


print("PASS")
# CHECK-NOT: Mismatched elements
