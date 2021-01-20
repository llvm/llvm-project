//===---------- target_impl.cu - NVPTX OpenMP GPU options ------- CUDA -*-===//
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
#pragma omp declare target

#include "target_impl.h"
#include "common/debug.h"
#include "common/target_atomic.h"

#include <cuda.h>

DEVICE void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi) {
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));
}

DEVICE uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi) {
  uint64_t val;
  asm volatile("mov.b64 %0, {%1,%2};" : "=l"(val) : "r"(lo), "r"(hi));
  return val;
}

DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(res));
  return res;
}

DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(res));
  return res;
}

DEVICE uint32_t __kmpc_impl_smid() {
  uint32_t id;
  asm("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

DEVICE double __kmpc_impl_get_wtick() {
  // Timer precision is 1ns
  return ((double)1E-9);
}

DEVICE double __kmpc_impl_get_wtime() {
  unsigned long long nsecs;
  asm("mov.u64  %0, %%globaltimer;" : "=l"(nsecs));
  return (double)nsecs * __kmpc_impl_get_wtick();
}

// In Cuda 9.0, __ballot(1) from Cuda 8.0 is replaced with __activemask().
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
#if CUDA_VERSION >= 9000
  return __activemask();
#else
  return __ballot(1);
#endif
}

// In Cuda 9.0, the *_sync() version takes an extra argument 'mask'.
DEVICE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t Mask, int32_t Var,
                                     int32_t SrcLane) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(Mask, Var, SrcLane);
#else
  return __shfl(Var, SrcLane);
#endif // CUDA_VERSION
}

DEVICE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t Mask,
                                          int32_t Var, uint32_t Delta,
                                          int32_t Width) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(Mask, Var, Delta, Width);
#else
  return __shfl_down(Var, Delta, Width);
#endif // CUDA_VERSION
}

DEVICE void __kmpc_impl_syncthreads() { __syncthreads(); }

DEVICE void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t Mask) {
#if CUDA_VERSION >= 9000
  __syncwarp(Mask);
#else
  // In Cuda < 9.0 no need to sync threads in warps.
#endif // CUDA_VERSION
}

// NVPTX specific kernel initialization
DEVICE void __kmpc_impl_target_init() { /* nvptx needs no extra setup */
}

// Barrier until num_threads arrive.
DEVICE void __kmpc_impl_named_sync(uint32_t num_threads) {
  // The named barrier for active parallel threads of a team in an L1 parallel
  // region to synchronize with each other.
  int barrier = 1;
  asm volatile("bar.sync %0, %1;"
               :
               : "r"(barrier), "r"(num_threads)
               : "memory");
}

DEVICE void __kmpc_impl_threadfence() { __threadfence(); }
DEVICE void __kmpc_impl_threadfence_block() { __threadfence_block(); }
DEVICE void __kmpc_impl_threadfence_system() { __threadfence_system(); }

// Calls to the NVPTX layer (assuming 1D layout)
DEVICE int GetThreadIdInBlock() { return __nvvm_read_ptx_sreg_tid_x(); }
DEVICE int GetBlockIdInKernel() { return __nvvm_read_ptx_sreg_ctaid_x(); }
DEVICE int GetNumberOfBlocksInKernel() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}
DEVICE int GetNumberOfThreadsInBlock() { return __nvvm_read_ptx_sreg_ntid_x(); }
DEVICE unsigned GetWarpId() { return GetThreadIdInBlock() / WARPSIZE; }
DEVICE unsigned GetLaneId() { return GetThreadIdInBlock() & (WARPSIZE - 1); }

#define __OMP_SPIN 1000
#define UNSET 0u
#define SET 1u

DEVICE void __kmpc_impl_init_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

DEVICE void __kmpc_impl_destroy_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

DEVICE void __kmpc_impl_set_lock(omp_lock_t *lock) {
  // TODO: not sure spinning is a good idea here..
  while (__kmpc_atomic_cas(lock, UNSET, SET) != UNSET) {
    int32_t start = __nvvm_read_ptx_sreg_clock();
    int32_t now;
    for (;;) {
      now = __nvvm_read_ptx_sreg_clock();
      int32_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= __OMP_SPIN * GetBlockIdInKernel()) {
        break;
      }
    }
  } // wait for 0 to be the read value
}

DEVICE void __kmpc_impl_unset_lock(omp_lock_t *lock) {
  (void)__kmpc_atomic_exchange(lock, UNSET);
}

DEVICE int __kmpc_impl_test_lock(omp_lock_t *lock) {
  return __kmpc_atomic_add(lock, 0u);
}

DEVICE void *__kmpc_impl_malloc(size_t x) { return malloc(x); }
DEVICE void __kmpc_impl_free(void *x) { free(x); }

#pragma omp end declare target
