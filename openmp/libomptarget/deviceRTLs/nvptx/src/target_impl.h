//===------------ target_impl.h - NVPTX OpenMP GPU options ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of target specific functions
//
//===----------------------------------------------------------------------===//
#ifndef _TARGET_IMPL_H_
#define _TARGET_IMPL_H_

#include <assert.h>
#include <cuda.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "nvptx_interface.h"

#define DEVICE __device__
#define INLINE __forceinline__ DEVICE
#define NOINLINE __noinline__ DEVICE
#define SHARED __shared__
#define ALIGN(N) __align__(N)

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The following def must match the absolute limit hardwired in the host RTL
// max number of threads per team
#define MAX_THREADS_PER_TEAM 1024

#define WARPSIZE 32

// Maximum number of preallocated arguments to an outlined parallel/simd function.
// Anything more requires dynamic memory allocation.
#define MAX_SHARED_ARGS 20

// Maximum number of omp state objects per SM allocated statically in global
// memory.
#if __CUDA_ARCH__ >= 600
#define OMP_STATE_COUNT 32
#else
#define OMP_STATE_COUNT 16
#endif

#if !defined(MAX_SM)
#if __CUDA_ARCH__ >= 900
#error unsupported compute capability, define MAX_SM via LIBOMPTARGET_NVPTX_MAX_SM cmake option
#elif __CUDA_ARCH__ >= 800
// GA100 design has a maxinum of 128 SMs but A100 product only has 108 SMs
// GA102 design has a maxinum of 84 SMs
#define MAX_SM 108
#elif __CUDA_ARCH__ >= 700
#define MAX_SM 84
#elif __CUDA_ARCH__ >= 600
#define MAX_SM 56
#else
#define MAX_SM 16
#endif
#endif

#define OMP_ACTIVE_PARALLEL_LEVEL 128

// Data sharing related quantities, need to match what is used in the compiler.
enum DATA_SHARING_SIZES {
  // The maximum number of workers in a kernel.
  DS_Max_Worker_Threads = 992,
  // The size reserved for data in a shared memory slot.
  DS_Slot_Size = 256,
  // The slot size that should be reserved for a working warp.
  DS_Worker_Warp_Slot_Size = WARPSIZE * DS_Slot_Size,
  // The maximum number of warps in use
  DS_Max_Warp_Number = 32,
  // The size of the preallocated shared memory buffer per team
  DS_Shared_Memory_Size = 128,
};

enum : __kmpc_impl_lanemask_t {
  __kmpc_impl_all_lanes = ~(__kmpc_impl_lanemask_t)0
};

DEVICE void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi);
DEVICE uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi);
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt();
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt();
DEVICE uint32_t __kmpc_impl_smid();
DEVICE double __kmpc_impl_get_wtick();
DEVICE double __kmpc_impl_get_wtime();

INLINE uint32_t __kmpc_impl_ffs(uint32_t x) { return __builtin_ffs(x); }
INLINE uint32_t __kmpc_impl_popc(uint32_t x) { return __builtin_popcount(x); }

#ifndef CUDA_VERSION
#error CUDA_VERSION macro is undefined, something wrong with cuda.
#endif

DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask();

DEVICE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t Mask, int32_t Var,
                                     int32_t SrcLane);

DEVICE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t Mask,
                                          int32_t Var, uint32_t Delta,
                                          int32_t Width);

DEVICE void __kmpc_impl_syncthreads();
DEVICE void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t Mask);

// NVPTX specific kernel initialization
DEVICE void __kmpc_impl_target_init();

// Barrier until num_threads arrive.
DEVICE void __kmpc_impl_named_sync(uint32_t num_threads);

DEVICE void __kmpc_impl_threadfence();
DEVICE void __kmpc_impl_threadfence_block();
DEVICE void __kmpc_impl_threadfence_system();

// Calls to the NVPTX layer (assuming 1D layout)
DEVICE int GetThreadIdInBlock();
DEVICE int GetBlockIdInKernel();
DEVICE int GetNumberOfBlocksInKernel();
DEVICE int GetNumberOfThreadsInBlock();
DEVICE unsigned GetWarpId();
DEVICE unsigned GetLaneId();

// Locks
DEVICE void __kmpc_impl_init_lock(omp_lock_t *lock);
DEVICE void __kmpc_impl_destroy_lock(omp_lock_t *lock);
DEVICE void __kmpc_impl_set_lock(omp_lock_t *lock);
DEVICE void __kmpc_impl_unset_lock(omp_lock_t *lock);
DEVICE int __kmpc_impl_test_lock(omp_lock_t *lock);

// Memory
DEVICE void *__kmpc_impl_malloc(size_t);
DEVICE void __kmpc_impl_free(void *);

#endif
