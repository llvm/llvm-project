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

#include "common/debug.h"
#include "target_impl.h"

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

DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  unsigned int Mask;
  asm volatile("activemask.b32 %0;" : "=r"(Mask));
  return Mask;
}

DEVICE void __kmpc_impl_syncthreads() { __syncthreads(); }

DEVICE void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t Mask) {
  __nvvm_bar_warp_sync(Mask);
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

DEVICE void __kmpc_impl_threadfence() { __nvvm_membar_gl(); }
DEVICE void __kmpc_impl_threadfence_block() { __nvvm_membar_cta(); }
DEVICE void __kmpc_impl_threadfence_system() { __nvvm_membar_sys(); }

// Calls to the NVPTX layer (assuming 1D layout)
DEVICE int GetThreadIdInBlock() { return __nvvm_read_ptx_sreg_tid_x(); }
DEVICE int GetBlockIdInKernel() { return __nvvm_read_ptx_sreg_ctaid_x(); }
DEVICE int GetNumberOfBlocksInKernel() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}
DEVICE int GetNumberOfThreadsInBlock() { return __nvvm_read_ptx_sreg_ntid_x(); }
DEVICE unsigned GetWarpId() { return GetThreadIdInBlock() / WARPSIZE; }
DEVICE unsigned GetLaneId() { return GetThreadIdInBlock() & (WARPSIZE - 1); }

// Atomics
DEVICE uint32_t __kmpc_atomic_add(uint32_t *Address, uint32_t Val) {
  return __atomic_fetch_add(Address, Val, __ATOMIC_SEQ_CST);
}
DEVICE uint32_t __kmpc_atomic_inc(uint32_t *Address, uint32_t Val) {
  return __nvvm_atom_inc_gen_ui(Address, Val);
}

DEVICE uint32_t __kmpc_atomic_max(uint32_t *Address, uint32_t Val) {
  return __atomic_fetch_max(Address, Val, __ATOMIC_SEQ_CST);
}

DEVICE uint32_t __kmpc_atomic_exchange(uint32_t *Address, uint32_t Val) {
  uint32_t R;
  __atomic_exchange(Address, &Val, &R, __ATOMIC_SEQ_CST);
  return R;
}

DEVICE uint32_t __kmpc_atomic_cas(uint32_t *Address, uint32_t Compare,
                                  uint32_t Val) {
  (void)__atomic_compare_exchange(Address, &Compare, &Val, false,
                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return Compare;
}

DEVICE unsigned long long __kmpc_atomic_exchange(unsigned long long *Address,
                                                 unsigned long long Val) {
  unsigned long long R;
  __atomic_exchange(Address, &Val, &R, __ATOMIC_SEQ_CST);
  return R;
}

DEVICE unsigned long long __kmpc_atomic_add(unsigned long long *Address,
                                            unsigned long long Val) {
  return __atomic_fetch_add(Address, Val, __ATOMIC_SEQ_CST);
}

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
