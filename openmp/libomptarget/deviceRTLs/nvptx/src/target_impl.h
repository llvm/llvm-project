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

#include <cuda.h>
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

// The named barrier for active parallel threads of a team in an L1 parallel
// region to synchronize with each other.
#define L1_BARRIER (1)

// Maximum number of preallocated arguments to an outlined parallel/simd function.
// Anything more requires dynamic memory allocation.
#define MAX_SHARED_ARGS 20

// Maximum number of omp state objects per SM allocated statically in global
// memory.
#if __CUDA_ARCH__ >= 700
#define OMP_STATE_COUNT 32
#define MAX_SM 84
#elif __CUDA_ARCH__ >= 600
#define OMP_STATE_COUNT 32
#define MAX_SM 56
#else
#define OMP_STATE_COUNT 16
#define MAX_SM 16
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

INLINE void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi) {
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));
}

INLINE uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi) {
  uint64_t val;
  asm volatile("mov.b64 %0, {%1,%2};" : "=l"(val) : "r"(lo), "r"(hi));
  return val;
}

static const __kmpc_impl_lanemask_t __kmpc_impl_all_lanes =
    UINT32_C(0xffffffff);

INLINE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(res));
  return res;
}

INLINE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(res));
  return res;
}

INLINE uint32_t __kmpc_impl_smid() {
  uint32_t id;
  asm("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

INLINE double __target_impl_get_wtick() {
  // Timer precision is 1ns
  return ((double)1E-9);
}

INLINE double __target_impl_get_wtime() {
  unsigned long long nsecs;
  asm("mov.u64  %0, %%globaltimer;" : "=l"(nsecs));
  return (double)nsecs * __target_impl_get_wtick();
}

INLINE uint32_t __kmpc_impl_ffs(uint32_t x) { return __ffs(x); }

INLINE uint32_t __kmpc_impl_popc(uint32_t x) { return __popc(x); }

template <typename T> INLINE T __kmpc_impl_min(T x, T y) {
  return min(x, y);
}

#ifndef CUDA_VERSION
#error CUDA_VERSION macro is undefined, something wrong with cuda.
#endif

// In Cuda 9.0, __ballot(1) from Cuda 8.0 is replaced with __activemask().

INLINE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
#if CUDA_VERSION >= 9000
  return __activemask();
#else
  return __ballot(1);
#endif
}

// In Cuda 9.0, the *_sync() version takes an extra argument 'mask'.

INLINE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t Mask, int32_t Var,
                                     int32_t SrcLane) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(Mask, Var, SrcLane);
#else
  return __shfl(Var, SrcLane);
#endif // CUDA_VERSION
}

INLINE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t Mask,
                                          int32_t Var, uint32_t Delta,
                                          int32_t Width) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(Mask, Var, Delta, Width);
#else
  return __shfl_down(Var, Delta, Width);
#endif // CUDA_VERSION
}

INLINE void __kmpc_impl_syncthreads() {
  // Use original __syncthreads if compiled by nvcc or clang >= 9.0.
#if !defined(__clang__) || __clang_major__ >= 9
  __syncthreads();
#else
  asm volatile("bar.sync %0;" : : "r"(0) : "memory");
#endif // __clang__
}

INLINE void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t Mask) {
#if CUDA_VERSION >= 9000
  __syncwarp(Mask);
#else
  // In Cuda < 9.0 no need to sync threads in warps.
#endif // CUDA_VERSION
}

INLINE void __kmpc_impl_named_sync(int barrier, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;"
               :
               : "r"(barrier), "r"(num_threads)
               : "memory");
}

INLINE void __kmpc_impl_threadfence(void) { __threadfence(); }
INLINE void __kmpc_impl_threadfence_block(void) { __threadfence_block(); }
INLINE void __kmpc_impl_threadfence_system(void) { __threadfence_system(); }

// Calls to the NVPTX layer (assuming 1D layout)
INLINE int GetThreadIdInBlock() { return threadIdx.x; }
INLINE int GetBlockIdInKernel() { return blockIdx.x; }
INLINE int GetNumberOfBlocksInKernel() { return gridDim.x; }
INLINE int GetNumberOfThreadsInBlock() { return blockDim.x; }

// Return true if this is the first active thread in the warp.
INLINE bool __kmpc_impl_is_first_active_thread() {
  unsigned long long Mask = __kmpc_impl_activemask();
  unsigned long long ShNum = WARPSIZE - (GetThreadIdInBlock() % WARPSIZE);
  unsigned long long Sh = Mask << ShNum;
  // Truncate Sh to the 32 lower bits
  return (unsigned)Sh == 0;
}

// Locks
EXTERN void __kmpc_impl_init_lock(omp_lock_t *lock);
EXTERN void __kmpc_impl_destroy_lock(omp_lock_t *lock);
EXTERN void __kmpc_impl_set_lock(omp_lock_t *lock);
EXTERN void __kmpc_impl_unset_lock(omp_lock_t *lock);
EXTERN int __kmpc_impl_test_lock(omp_lock_t *lock);

// Memory
INLINE void *__kmpc_impl_malloc(size_t x) { return malloc(x); }
INLINE void __kmpc_impl_free(void *x) { free(x); }

#endif
