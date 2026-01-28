# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN: sh -c 'if [[ "%mlir_run_cuda_sm90_tests" == "1" ]]; \
# RUN: then %PYTHON %s | FileCheck %s; \
# RUN: else export MLIR_NVDSL_PRINT_IR=1; \
# RUN: %PYTHON %s | FileCheck %s --check-prefix=DUMPIR; fi'


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

gemm_multistage(a, b, d, num_stages=7)

if os.getenv("MLIR_NVDSL_PRINT_IR") != "1":
    # Verify MLIR with reference computation
    ref_d = a.astype(np.float16) @ b.astype(np.float16)
    np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)

    print("PASS")
# CHECK-NOT: Mismatched elements
# CHECK: PASS

# DUMPIR:   func.func @gemm_multistage(%{{.*}}: memref<512x1024xf16>, %{{.*}}: memref<1024x256xf16>, %{{.*}}: memref<512x256xf32>) attributes {llvm.emit_c_interface} {
# DUMPIR:       scf.if %{{.*}} {
# DUMPIR:         %[[C0_INIT:.*]] = arith.constant 0 : index
# DUMPIR:         %[[C7:.*]] = arith.constant 7 : index
# DUMPIR:         %[[C1_INIT:.*]] = arith.constant 1 : index
# DUMPIR:         scf.for %arg15 = %[[C0_INIT]] to %[[C7]] step %[[C1_INIT]] {
# DUMPIR:           %[[C1_MBAR:.*]] = arith.constant 1 : index
# DUMPIR:           nvgpu.mbarrier.init %{{.*}}[%arg15], %[[C1_MBAR]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:         }
# DUMPIR:         nvgpu.tma.prefetch.descriptor %{{.*}} : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:         nvgpu.tma.prefetch.descriptor %{{.*}} : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:       }
# DUMPIR:       %[[C0_PROLOGUE:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C6:.*]] = arith.constant 6 : index
# DUMPIR:       %[[C1_PROLOGUE:.*]] = arith.constant 1 : index
# DUMPIR:       scf.for %arg15 = %[[C0_PROLOGUE]] to %[[C6]] step %[[C1_PROLOGUE]] {
# DUMPIR:         %[[BID_X_P:.*]] = gpu.block_id  x
# DUMPIR:         %[[BID_Y_P:.*]] = gpu.block_id  y
# DUMPIR:         %[[C128_P1:.*]] = arith.constant 128 : index
# DUMPIR:         %[[DIMX_P:.*]] = arith.muli %[[BID_X_P]], %[[C128_P1]] : index
# DUMPIR:         %[[C128_P2:.*]] = arith.constant 128 : index
# DUMPIR:         %[[DIMY_P:.*]] = arith.muli %[[BID_Y_P]], %[[C128_P2]] : index
# DUMPIR:         %{{.*}} = gpu.thread_id  x
# DUMPIR:         %[[TID_X_P:.*]] = gpu.thread_id  x
# DUMPIR:         %[[C0_P:.*]] = arith.constant 0 : index
# DUMPIR:         %[[PRED_P:.*]] = arith.cmpi eq, %[[TID_X_P]], %[[C0_P]] : index
# DUMPIR:         %[[C16384_P1:.*]] = arith.constant 16384 : index
# DUMPIR:         %[[OFF_A_P:.*]] = arith.muli %arg15, %[[C16384_P1]] : index
# DUMPIR:         %[[C16384_P2:.*]] = arith.constant 16384 : index
# DUMPIR:         %[[OFF_B_BASE_P:.*]] = arith.muli %arg15, %[[C16384_P2]] : index
# DUMPIR:         %[[C114688:.*]] = arith.constant 114688 : index
# DUMPIR:         %[[OFF_B1_P:.*]] = arith.addi %[[OFF_B_BASE_P]], %[[C114688]] : index
# DUMPIR:         %[[C8192:.*]] = arith.constant 8192 : index
# DUMPIR:         %[[OFF_B2_P:.*]] = arith.addi %[[OFF_B1_P]], %[[C8192]] : index
# DUMPIR:         %[[SMEM_A_P:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_A_P:.*]] = memref.view %[[SMEM_A_P]][%[[OFF_A_P]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[SMEM_B1_P:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_B1_P:.*]] = memref.view %[[SMEM_B1_P]][%[[OFF_B1_P]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[SMEM_B2_P:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_B2_P:.*]] = memref.view %[[SMEM_B2_P]][%[[OFF_B2_P]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[C32768:.*]] = arith.constant 32768 : index
# DUMPIR:         nvgpu.mbarrier.arrive.expect_tx %{{.*}}[%arg15], %[[C32768]], predicate = %[[PRED_P]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:         %[[C64_K_P:.*]] = arith.constant 64 : index
# DUMPIR:         %[[K_COORD_P:.*]] = arith.muli %arg15, %[[C64_K_P]] : index
# DUMPIR:         nvgpu.tma.async.load %{{.*}}[%[[K_COORD_P]], %[[DIMX_P]]], %{{.*}}[%arg15] to %[[VIEW_A_P]], predicate = %[[PRED_P]] : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         nvgpu.tma.async.load %{{.*}}[%[[DIMY_P]], %[[K_COORD_P]]], %{{.*}}[%arg15] to %[[VIEW_B1_P]], predicate = %[[PRED_P]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[C64_OFF:.*]] = arith.constant 64 : index
# DUMPIR:         %[[DIMY_P_OFF:.*]] = arith.addi %[[DIMY_P]], %[[C64_OFF]] : index
# DUMPIR:         nvgpu.tma.async.load %{{.*}}[%[[DIMY_P_OFF]], %[[K_COORD_P]]], %{{.*}}[%arg15] to %[[VIEW_B2_P]], predicate = %[[PRED_P]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       }
# DUMPIR:       %[[TID_X_LOOP:.*]] = gpu.thread_id  x
# DUMPIR:       %[[ACC_INIT:.*]] = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
# DUMPIR:       %[[FALSE_LOOP:.*]] = arith.constant false
# DUMPIR:       %[[C0_LOOP:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C16_LOOP:.*]] = arith.constant 16 : index
# DUMPIR:       %[[C1_LOOP:.*]] = arith.constant 1 : index
# DUMPIR:       %[[LOOP_RES:.*]]:2 = scf.for %arg15 = %[[C0_LOOP]] to %[[C16_LOOP]] step %[[C1_LOOP]] iter_args(%arg16 = %[[ACC_INIT]], %arg17 = %[[FALSE_LOOP]]) -> (!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1) {
# DUMPIR:         %[[C7_L:.*]] = arith.constant 7 : index
# DUMPIR:         %[[STAGE_L:.*]] = arith.remui %arg15, %[[C7_L]] : index
# DUMPIR:         %[[C10M:.*]] = arith.constant 10000000 : index
# DUMPIR:         nvgpu.mbarrier.try_wait.parity %{{.*}}[%[[STAGE_L]]], %arg17, %[[C10M]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:         %[[C16384_L:.*]] = arith.constant 16384 : index
# DUMPIR:         %[[OFF_A_L:.*]] = arith.muli %[[STAGE_L]], %[[C16384_L]] : index
# DUMPIR:         %[[C114688_L:.*]] = arith.constant 114688 : index
# DUMPIR:         %[[OFF_B_L:.*]] = arith.addi %[[OFF_A_L]], %[[C114688_L]] : index
# DUMPIR:         %[[SMEM_A_L:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_A_L:.*]] = memref.view %[[SMEM_A_L]][%[[OFF_A_L]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[SMEM_B_L:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_B_L:.*]] = memref.view %[[SMEM_B_L]][%[[OFF_B_L]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[DESC_A_L:.*]] = nvgpu.warpgroup.generate.descriptor %[[VIEW_A_L]], %{{.*}} : memref<128x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>
# DUMPIR:         %[[DESC_B_L:.*]] = nvgpu.warpgroup.generate.descriptor %[[VIEW_B_L]], %{{.*}} : memref<64x128xf16, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>
# DUMPIR:         %[[ACC_L:.*]] = nvgpu.warpgroup.mma %[[DESC_A_L]], %[[DESC_B_L]], %arg16 {transposeB} : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
# DUMPIR:         %[[C6_NEXT:.*]] = arith.constant 6 : index
# DUMPIR:         %[[ITER_NEXT:.*]] = arith.addi %arg15, %[[C6_NEXT]] : index
# DUMPIR:         %[[C16_CMP:.*]] = arith.constant 16 : index
# DUMPIR:         %[[IN_RANGE:.*]] = arith.cmpi ult, %[[ITER_NEXT]], %[[C16_CMP]] : index
# DUMPIR:         %[[C0_CMP:.*]] = arith.constant 0 : index
# DUMPIR:         %[[IS_THREAD0_L:.*]] = arith.cmpi eq, %[[TID_X_LOOP]], %[[C0_CMP]] : index
# DUMPIR:         %[[DO_LOAD:.*]] = arith.andi %[[IN_RANGE]], %[[IS_THREAD0_L]] : i1
# DUMPIR:         %[[C6_STAGE:.*]] = arith.constant 6 : index
# DUMPIR:         %[[STAGE_NEXT_L:.*]] = arith.addi %arg15, %[[C6_STAGE]] : index
# DUMPIR:         %[[C7_MOD:.*]] = arith.constant 7 : index
# DUMPIR:         %[[STAGE_LOAD:.*]] = arith.remui %[[STAGE_NEXT_L]], %[[C7_MOD]] : index
# DUMPIR:         %[[BID_X_L:.*]] = gpu.block_id  x
# DUMPIR:         %[[BID_Y_L:.*]] = gpu.block_id  y
# DUMPIR:         %[[C128_L1:.*]] = arith.constant 128 : index
# DUMPIR:         %[[DIMX_L:.*]] = arith.muli %[[BID_X_L]], %[[C128_L1]] : index
# DUMPIR:         %[[C128_L2:.*]] = arith.constant 128 : index
# DUMPIR:         %[[DIMY_L:.*]] = arith.muli %[[BID_Y_L]], %[[C128_L2]] : index
# DUMPIR:         %[[TID_X_L1:.*]] = gpu.thread_id  x
# DUMPIR:         %[[TID_X_L2:.*]] = gpu.thread_id  x
# DUMPIR:         %[[C16384_LA1:.*]] = arith.constant 16384 : index
# DUMPIR:         %[[OFF_A_LOAD:.*]] = arith.muli %[[STAGE_LOAD]], %[[C16384_LA1]] : index
# DUMPIR:         %[[C16384_LA2:.*]] = arith.constant 16384 : index
# DUMPIR:         %[[OFF_B_BASE_LOAD:.*]] = arith.muli %[[STAGE_LOAD]], %[[C16384_LA2]] : index
# DUMPIR:         %[[C114688_LOAD:.*]] = arith.constant 114688 : index
# DUMPIR:         %[[OFF_B1_LOAD:.*]] = arith.addi %[[OFF_B_BASE_LOAD]], %[[C114688_LOAD]] : index
# DUMPIR:         %[[C8192_LOAD:.*]] = arith.constant 8192 : index
# DUMPIR:         %[[OFF_B2_LOAD:.*]] = arith.addi %[[OFF_B1_LOAD]], %[[C8192_LOAD]] : index
# DUMPIR:         %[[SMEM_A_LOAD:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_A_LOAD:.*]] = memref.view %[[SMEM_A_LOAD]][%[[OFF_A_LOAD]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[SMEM_B1_LOAD:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_B1_LOAD:.*]] = memref.view %[[SMEM_B1_LOAD]][%[[OFF_B1_LOAD]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[SMEM_B2_LOAD:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:         %[[VIEW_B2_LOAD:.*]] = memref.view %[[SMEM_B2_LOAD]][%[[OFF_B2_LOAD]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[C32768_LOAD:.*]] = arith.constant 32768 : index
# DUMPIR:         nvgpu.mbarrier.arrive.expect_tx %{{.*}}[%[[STAGE_LOAD]]], %[[C32768_LOAD]], predicate = %[[DO_LOAD]] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
# DUMPIR:         %[[C64_K_LOAD:.*]] = arith.constant 64 : index
# DUMPIR:         %[[K_COORD_LOAD:.*]] = arith.muli %[[STAGE_NEXT_L]], %[[C64_K_LOAD]] : index
# DUMPIR:         nvgpu.tma.async.load %{{.*}}[%[[K_COORD_LOAD]], %[[DIMX_L]]], %{{.*}}[%[[STAGE_LOAD]]] to %[[VIEW_A_LOAD]], predicate = %[[DO_LOAD]] : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         nvgpu.tma.async.load %{{.*}}[%[[DIMY_L]], %[[K_COORD_LOAD]]], %{{.*}}[%[[STAGE_LOAD]]] to %[[VIEW_B1_LOAD]], predicate = %[[DO_LOAD]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[C64_OFF_LOAD:.*]] = arith.constant 64 : index
# DUMPIR:         %[[DIMY_L_OFF:.*]] = arith.addi %[[DIMY_L]], %[[C64_OFF_LOAD]] : index
# DUMPIR:         nvgpu.tma.async.load %{{.*}}[%[[DIMY_L_OFF]], %[[K_COORD_LOAD]]], %{{.*}}[%[[STAGE_LOAD]]] to %[[VIEW_B2_LOAD]], predicate = %[[DO_LOAD]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:         %[[C6_FLIP:.*]] = arith.constant 6 : index
# DUMPIR:         %[[IS_STAGE6:.*]] = arith.cmpi eq, %[[STAGE_L]], %[[C6_FLIP]] : index
# DUMPIR:         %[[TRUE:.*]] = arith.constant true
# DUMPIR:         %[[PARITY_FLIP:.*]] = arith.xori %arg17, %[[TRUE]] : i1
# DUMPIR:         %[[NEW_PARITY:.*]] = arith.select %[[IS_STAGE6]], %[[PARITY_FLIP]], %arg17 : i1
# DUMPIR:         scf.yield %[[ACC_L]], %[[NEW_PARITY]] : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1
# DUMPIR:       }
# DUMPIR:       nvvm.wgmma.wait.group.sync.aligned 0
# DUMPIR:       %[[TID_X_EPI:.*]] = gpu.thread_id  x
# DUMPIR:       %[[BID_X_EPI:.*]] = gpu.block_id  x
# DUMPIR:       %[[BID_Y_EPI:.*]] = gpu.block_id  y
# DUMPIR:       %[[C128_EPI1:.*]] = arith.constant 128 : index
# DUMPIR:       %[[DIMX_EPI:.*]] = arith.muli %[[BID_X_EPI]], %[[C128_EPI1]] : index
# DUMPIR:       %[[C128_EPI2:.*]] = arith.constant 128 : index
# DUMPIR:       %[[DIMY_EPI:.*]] = arith.muli %[[BID_Y_EPI]], %[[C128_EPI2]] : index
# DUMPIR:       %[[SMEM_EPI:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_VIEW:.*]] = arith.constant 0 : index
# DUMPIR:       %[[VIEW_EPI:.*]] = memref.view %[[SMEM_EPI]][%[[C0_VIEW]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x128xf32, #gpu.address_space<workgroup>>
# DUMPIR:       %[[SUBVIEW_EPI:.*]] = memref.subview %{{.*}}[%[[DIMX_EPI]], %[[DIMY_EPI]]] [128, 128] [1, 1] : memref<512x256xf32> to memref<128x128xf32, strided<[256, 1], offset: ?>>
# DUMPIR:       nvgpu.warpgroup.mma.store %[[LOOP_RES]]#0, %[[VIEW_EPI]] : <fragmented = vector<128x128xf32>> to memref<128x128xf32, #gpu.address_space<workgroup>>
# DUMPIR:       gpu.barrier
# DUMPIR:       %[[C0_STORE:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C128_STORE:.*]] = arith.constant 128 : index
# DUMPIR:       %[[C1_STORE:.*]] = arith.constant 1 : index
# DUMPIR:       scf.for %arg15 = %[[C0_STORE]] to %[[C128_STORE]] step %[[C1_STORE]] {
# DUMPIR:         %[[VAL_LOAD:.*]] = memref.load %[[VIEW_EPI]][%arg15, %[[TID_X_EPI]]] : memref<128x128xf32, #gpu.address_space<workgroup>>
# DUMPIR:         memref.store %[[VAL_LOAD]], %[[SUBVIEW_EPI]][%arg15, %[[TID_X_EPI]]] : memref<128x128xf32, strided<[256, 1], offset: ?>>
