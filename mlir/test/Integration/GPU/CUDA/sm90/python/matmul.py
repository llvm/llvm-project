# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s


# ===--- GEMM Hopper Tensor Core Integration Test ---===
#
# This test aims to validate the correctness of the supported GEMM kernels in
# NVGPU dialects, with current support for Multistage and Warp Specialization
# kernels.
# The test constructs and metaprograms IR using Python bindings, allowing
# generic IR building. This flexibility enables changes to the shape,
# tile size, or data type of the GEMM for testing purposes.
# The entry function is `matmul`, where one can specify GEMM shape, tile size,
# data type, GEMM algorithm (Multistage or Warp Specialization), and the maximum
# number of stages.
# Verification is done via numpy's matmul operation.
#
# Example:
# matmul(input_type=np.float16,                # input types
#        output_type=np.float32,               # output type
#        M=4096, N=4096, K=4096,               # Shape
#        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, # Tile Size
#        use_warp_specialization=True,         # Enable Warp Specialization
#        max_num_stages=3)                     # Number of stages in shared memory
#
# ===--- Parallelism Across CTAs  ---===
#
# GEMM includes three loops defining the shape of the GEMM, specified in the
# `matmul` function.
# The program builds IR using the following loop structure, tiling the loops
# with the given tile size and parallelizing the two outermost loops into the
# first and second dimensions of CTAs.
#
# for(bi = 0; i < M; i += BLOCK_M)          # parallelize across blockIdx.x
#     for(bj = 0; j < N; j += BLOCK_N)      # parallelize across blockIdx.y
#         for(bk = 0; k < K; K += BLOCK_K)
#             for(i = bi; i < (bi + BLOCK_M); ++i)
#                 for(j = bj; j < (bj + BLOCK_N); ++j)
#                     for(k = bk; k < (bk + BLOCK_K); ++k)
#
# ===--- Multistage Kernel ---===
#
# This kernel launches a single warp group (128 threads). The primary thread
# (pthread) requests load from TMA. Threads collectively wait for the data and
# perform mma operations. After completing the shape, threads together store
# first fragmented registers to shared memory, then from shared memory to global
# memory; this part is called the epilogue.
#
# Execution Timeline of Multistage Kernel with 3 stages:
# +-------+----------------+--------------------+--------------------+--------------------+-----+-----------------------+
# |       |Prologue ---->   |MainLoop ---->                                                                  |Epilogue  |
# +-------+----------------+--------------------+--------------------+--------------------+-----+-----------------------+
# |pthread|[tma-0,1,2]     |[wait-0][mma][tma-2]|[wait-1][mma][tma-0]|[wait-2][mma][tma-1]| ... | [mma-wait] |[epilogue]|
# |wgroup | ........       |[wait-0][mma]       |[wait-1][mma]       |[wait-2][mma]       | ... | [mma-wait] |[epilogue]|
# +-------+----------------+--------------------+--------------------+--------------------+-----+-----------------------+
#
# ===--- Warp Specialization Kernel  ---===
#
# This kernel launches 2 warp groups (2x128 threads) per CTA, specializing one
# as `producer warp group` and another as `consumer warp group`. The
# `producer warp group` is responsible for requesting TMA load, while the
# `consumer warp group` performs the mma operation. The epilogue section is
# handled by the `consumer warp group` as its threads own the fragmented registers.
#
# Execution Timeline of Warp Specialization Kernel with 2 stages:
# +--------+--------+---------+---------+---------+-----------------------+---+--------------+-----------------+
# |        |MainLoop ---->                                                    | 1st Epilogue | 2nd Epilogue    |
# +--------+--------+---------+---------+---------+-----------------------+---+--------------+-----------------+
# |pthread1|[tma-0] | [tma-1] | [tma-0] | [tma-1] | ..........................| ...........  | [shmem->global] |
# |wgroup1 | .......|         |         |         |                           |              | [shmem->global] |
# +--------+--------+---------+---------+---------+-----------------------+---+--------------+-----------------+
# |wgroup2 |[wait-0][mma], [wait-1][mma], [wait-0][mma], [wait-1][mma], ......| [reg->shmem] | [shmem->global]|
# +--------+--------+---------+---------+---------+-----------------------+---+--------------+-----------------+

import errno
import numpy as np
import subprocess
import ctypes
from tools import nvgpucompiler
from tools import matmulBuilder
import contextlib
import os
import sys
import pathlib
import ctypes
from mlir import runtime as rt


def generate_matmul(
    input_type=np.float16,
    output_type=np.float32,
    M=4096,
    N=4096,
    K=4096,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    use_warp_specialization=True,
    saveIR=False,
    max_num_stages=3,
    options=f"cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3",
):
    with matmulBuilder.ir.Context() as ctx, matmulBuilder.ir.Location.unknown():
        if use_warp_specialization:
            mlir_nvgpu_module = matmulBuilder.generate_matmul_ws(
                input_type,
                output_type,
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                max_num_stages,
            )
        else:
            mlir_nvgpu_module = matmulBuilder.generate_matmul_multistage(
                input_type,
                output_type,
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                max_num_stages,
            )

        mlir_nvgpu_module.operation.verify()

        # Save generated IR
        if saveIR:
            # print(mlir_nvgpu_module)
            original_stdout = sys.stdout
            with open("gemm.mlir", "w") as f:
                sys.stdout = f
                print(mlir_nvgpu_module)
                sys.stdout = original_stdout

        # Get compiler
        support_lib = os.getenv("SUPPORT_LIB")
        if not os.path.exists(support_lib):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), support_lib
            )
        compiler = nvgpucompiler.NvgpuCompiler(
            options, opt_level=3, shared_libs=[support_lib]
        )

        # Compile
        engine = compiler.compile_and_jit(mlir_nvgpu_module)
        return engine


def matmul(
    input_type=np.float16,
    output_type=np.float32,
    M=128,
    N=128,
    K=128,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    use_warp_specialization=True,
    saveIR=False,
    max_num_stages=3,
    print_results=False,
    no_verify=False,
):
    # Print the configuration
    required_stages = (M * K + K * N) // (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N)
    num_stages = min(required_stages, max_num_stages)
    ity = "f16" if input_type == np.float16 else "f32"
    oty = "f16" if output_type == np.float16 else "f32"
    gemmty = "Warp specialization" if use_warp_specialization else "Multistage"
    print(
        "===-- Running GEMM "
        + gemmty
        + " "
        + oty
        + " += "
        + ity
        + " * "
        + ity
        + ", Size "
        + str(M)
        + "x"
        + str(N)
        + "x"
        + str(K)
        + ", Tile "
        + str(BLOCK_M)
        + "x"
        + str(BLOCK_N)
        + "x"
        + str(BLOCK_K)
        + ", stages "
        + str(num_stages)
        + " --==="
    )

    # Build IR and compile
    engine = generate_matmul(
        input_type,
        output_type,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        use_warp_specialization,
        saveIR,
        num_stages,
    )

    # Allocate matrices and invoke the matmul
    c = np.zeros((M, N), output_type)
    a = np.random.randn(M, K).astype(input_type)
    b = np.random.randn(K, N).astype(input_type)
    mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
    mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))
    mem_c = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(c)))
    kernelName = matmulBuilder.make_kernel_name(
        input_type,
        output_type,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_stages,
        use_warp_specialization,
    )

    # Launch the MLIR generated kernel
    engine.invoke(kernelName, mem_a, mem_b, mem_c)

    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={"float_kind": float_formatter})

    if print_results:
        print(c)

    # Verify the results
    if not no_verify:
        ref = a.astype(input_type) @ b.astype(input_type)
        if print_results:
            print(ref)
        np.testing.assert_allclose(c, ref, rtol=5e-03, atol=1e-01)

    print("PASS ")


# Takes longer time to run
def test_long():
    for stages in range(1, 7):
        for M in [128, 512, 1024, 4096, 8192]:
            for N in [128, 512, 1024, 4096, 8192]:
                for K in [64, 128, 512, 1024, 4096, 8192]:
                    matmul(
                        np.float16,
                        np.float32,
                        M,
                        N,
                        K,
                        max_num_stages=stages,
                        use_warp_specialization=False,
                        no_verify=True,
                    )
                    matmul(
                        np.float16,
                        np.float32,
                        M,
                        N,
                        K,
                        max_num_stages=stages,
                        use_warp_specialization=True,
                    )


def test_short():
    for stages in [1, 3]:
        for M in [128, 512]:
            for N in [128]:
                for K in [64, 256]:
                    matmul(
                        np.float16,
                        np.float32,
                        M,
                        N,
                        K,
                        max_num_stages=stages,
                        use_warp_specialization=False,
                    )
                    matmul(
                        np.float16,
                        np.float32,
                        M,
                        N,
                        K,
                        max_num_stages=stages,
                        use_warp_specialization=True,
                    )


# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 128x128x64, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 128x128x64, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 128x128x256, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 128x128x256, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 512x128x64, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 512x128x64, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 512x128x256, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 512x128x256, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 128x128x64, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 128x128x64, Tile 128x128x64, stages 1 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 128x128x256, Tile 128x128x64, stages 3 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 128x128x256, Tile 128x128x64, stages 3 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 512x128x64, Tile 128x128x64, stages 2 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 512x128x64, Tile 128x128x64, stages 2 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Multistage f32 += f16 * f16, Size 512x128x256, Tile 128x128x64, stages 3 --===
# CHECK: PASS
# CHECK: ===-- Running GEMM Warp specialization f32 += f16 * f16, Size 512x128x256, Tile 128x128x64, stages 3 --===
# CHECK: PASS

test_short()
