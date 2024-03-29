# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 4 : Multistage GEMM with Tensor Core
# ===----------------------------------------------------------------------===//
#
# This program demonstrates a GEMM operation with 64x64x64 matrix multiplication
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
    x_tma: TMA,
    y_tma: TMA,
    slot,
    stage,
    p=None,
):
    """
    TMA loads two input matrices from global memory to shared memory. It performs the following operations:

       - tma.load x_shared_memory[offset] at coordinate [x, y] (Loads 128x64)
       - tma.load y_shared_memory[offset] at coordinate [x, y] (Loads 64x64)
       - tma.load y_shared_memory[offset] at coordinate [x, y] (Loads 64x64)

       mbarrier.arrive tx_count = 128x64x2x4
    """
    dimX, dimY = partition_shape()

    tidx = gpu.thread_id(gpu.Dimension.x)
    begin_y = NUM_STAGES * get_type_size(x_tma.tma_memref)
    size_tma_x = get_type_size(x_tma.tma_memref)
    size_tma_y = get_type_size(y_tma.tma_memref)
    tx_count = size_tma_x + (size_tma_y * 2)
    tidx = gpu.thread_id(gpu.Dimension.x)

    p = tidx == 0 if p is None else p

    off_x = slot * size_tma_x
    off_y = (slot * size_tma_x) + begin_y
    off_y2 = off_y + size_tma_y
    x = get_dynamic_shared_memory(
        x_tma.tma_memref.shape, x_tma.tma_memref.element_type, off_x
    )
    y1 = get_dynamic_shared_memory(
        y_tma.tma_memref.shape, y_tma.tma_memref.element_type, off_y
    )
    y2 = get_dynamic_shared_memory(
        y_tma.tma_memref.shape, y_tma.tma_memref.element_type, off_y2
    )

    mbar_group[slot].arrive(tx_count, predicate=p)

    c1 = stage * 64
    x_tma.load(x, mbar_group[slot], coords=[c1, dimX], predicate=p)
    y_tma.load(y1, mbar_group[slot], coords=[dimY, c1], predicate=p)
    y_tma.load(y2, mbar_group[slot], coords=[dimY + 64, c1], predicate=p)


def bootstrap(x_tma: TMA, y_tma: TMA):
    """
    Initialize mbarriers and prefetch TMA descriptors.
    """
    tidx = gpu.thread_id(gpu.Dimension.x)
    mbar_group = Mbarriers(number_of_barriers=NUM_STAGES)
    isThread0 = tidx == const(0)
    with ir.InsertionPoint(scf.IfOp(isThread0).then_block):
        for i in scf.for_(0, NUM_STAGES, 1):
            mbar_group[i].init(1)
            scf.yield_([])
        x_tma.prefetch()
        y_tma.prefetch()
        scf.yield_([])

    return mbar_group


def prologue(mbar_group: Mbarriers, x_tma: TMA, y_tma: TMA):
    """
    Prologue of the GEMM kernel. It loads 2 input matrices for each stage in loop like below:

    for stage in range(NUM_STAGES):
        tma_load x, y, stage

    """
    ns = NUM_STAGES if NUM_STAGES == 1 else NUM_STAGES - 1
    for iv in scf.for_(0, ns, 1):
        tma_load(mbar_group, x_tma, y_tma, iv, iv)
        scf.yield_([])


def mainloop(mbar_group: Mbarriers, x_tma: TMA, y_tma: TMA):
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
    ns = NUM_STAGES if NUM_STAGES == 1 else NUM_STAGES - 1

    tidx = gpu.thread_id(gpu.Dimension.x)
    begin_y = NUM_STAGES * get_type_size(x_tma.tma_memref)

    size_x = TILE_M * TILE_K * get_type_size(T.f16())

    C = MatrixAccumulator(TILE_M, TILE_N, T.f32()).op()
    pp = const(False, ty=T.bool())

    # Main Loop
    for_op = scf.ForOp(const(0), const(K // TILE_K), const(1), [C, pp])
    with ir.InsertionPoint(for_op.body):
        pp = for_op.inner_iter_args[1]
        iv = for_op.induction_variable
        stage = iv % NUM_STAGES

        # Wait for current stage
        mbar_group[stage].try_wait(phase=pp)

        # Find shared memory slot
        offset_x = stage * size_x
        offset_y = offset_x + begin_y
        x_smem = get_dynamic_shared_memory([TILE_M, TILE_K], T.f16(), offset_x)
        y_smem = get_dynamic_shared_memory([TILE_K, TILE_N], T.f16(), offset_y)

        # Matrix Multiply
        A = Matrix(x_smem, x_tma, TILE_M, TILE_K)
        B = Matrix(y_smem, y_tma, TILE_K, TILE_N)
        C = for_op.inner_iter_args[0]
        D = Matrix.matmul(A, B, C)
        if NUM_STAGES == 1:
            nvvm.WgmmaWaitGroupSyncOp(0)

        # Load next stage
        pred = ((iv + ns) < const(K // TILE_K)) & (tidx == 0)
        nextStage = iv + ns
        nextSlot = nextStage % NUM_STAGES
        tma_load(mbar_group, x_tma, y_tma, nextSlot, nextStage, pred)

        # Switch phase parity for the mbarrier
        switched = pp ^ const(True, ty=T.bool())
        newPP = arith.select(
            stage == (NUM_STAGES - 1),
            switched,
            pp,
        )
        scf.yield_([D, newPP])

    nvvm.WgmmaWaitGroupSyncOp(0)

    return for_op.results[0]


def epilogue(D, z_dev):
    """
    Epilogue of the GEMM kernel. It stores the fragmented registers to global memory.

    MatrixAccumulator D               # Fragmented results
    store D -> Shared Memory          # Store Shared Memory
    Shared Memory -> Z[dimX][dimY]    # Store Shared Memory to Global Memory

    """
    tidx = gpu.thread_id(gpu.Dimension.x)
    dimX, dimY = partition_shape()

    z_smem = get_dynamic_shared_memory([TILE_M, TILE_N], T.f32())
    z_gmem = memref.subview(z_dev, [dimX, dimY], [TILE_M, TILE_N], [1, 1])

    # Store (registers -> shared memory)
    nvgpu.WarpgroupMmaStoreOp(D, z_smem)
    gpu.barrier()

    # Store (shared memory --> global memory)
    for i in scf.for_(0, TILE_M, 1):
        val = memref.load(z_smem, [i, tidx])
        memref.store(val, z_gmem, [i, tidx])
        scf.yield_([])


@NVDSL.mlir_func
def gemm_multistage(x, y, z):
    token_ty = ir.Type.parse("!gpu.async.token")
    t1 = gpu.wait(token_ty, [])
    x_dev, t2 = gpu.alloc(x.type, token_ty, [t1], [], [])
    y_dev, t3 = gpu.alloc(y.type, token_ty, [t2], [], [])
    z_dev, t4 = gpu.alloc(z.type, token_ty, [t3], [], [])
    t5 = gpu.memcpy(token_ty, [t4], x_dev, x)
    t6 = gpu.memcpy(token_ty, [t5], y_dev, y)
    t7 = gpu.wait(token_ty, [t6])

    sw = nvgpu.TensorMapSwizzleKind.SWIZZLE_128B
    x_tma = TMA([128, 64], x.type, swizzle=sw)
    y_tma = TMA([64, 64], y.type, swizzle=sw)
    x_tma.create_descriptor(x_dev)
    y_tma.create_descriptor(y_dev)

    grid = [(M // TILE_M), (N // TILE_N), 1]
    block = [128, 1, 1]
    @NVDSL.mlir_gpu_launch(grid=grid, block=block, smem=229440)
    def gemm_multistage_kernel():
        # Initialize mbarriers and prefetch TMA descriptors
        mbar_group = bootstrap(x_tma, y_tma)

        # Fill the pipeline stages
        prologue(mbar_group, x_tma, y_tma)

        # Main loop
        D = mainloop(mbar_group, x_tma, y_tma)

        # Store registers to global memory
        epilogue(D, z_dev)

    gemm_multistage_kernel()

    t8 = gpu.memcpy(token_ty, [t7], z, z_dev)
    gpu.wait(None, [t8])


# Python pass arguments to MLIR
NUM_STAGES = 7
N = 256
M = 512
K = 1024
TILE_M = 128
TILE_N = 128
TILE_K = 64
x = np.random.randn(M, K).astype(np.float16)
y = np.random.randn(K, N).astype(np.float16)
z = np.zeros((M, N), np.float32)

gemm_multistage(x, y, z)


# Verify MLIR with reference computation
ref = x.astype(np.float16) @ y.astype(np.float16)
np.testing.assert_allclose(z, ref, rtol=5e-03, atol=1e-01)


print("PASS")
# CHECK-NOT: Mismatched elements
