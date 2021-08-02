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
#include "target_interface.h"

EXTERN void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi) {
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));
}

EXTERN uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi) {
  uint64_t val;
  asm volatile("mov.b64 %0, {%1,%2};" : "=l"(val) : "r"(lo), "r"(hi));
  return val;
}

EXTERN __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(res));
  return res;
}

EXTERN __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  __kmpc_impl_lanemask_t res;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(res));
  return res;
}

EXTERN uint32_t __kmpc_impl_smid() {
  uint32_t id;
  asm("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

EXTERN double __kmpc_impl_get_wtick() {
  // Timer precision is 1ns
  return ((double)1E-9);
}

EXTERN double __kmpc_impl_get_wtime() {
  unsigned long long nsecs;
  asm("mov.u64  %0, %%globaltimer;" : "=l"(nsecs));
  return (double)nsecs * __kmpc_impl_get_wtick();
}

EXTERN __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  unsigned int Mask;
  asm volatile("activemask.b32 %0;" : "=r"(Mask));
  return Mask;
}

EXTERN void __kmpc_impl_syncthreads() {
  int barrier = 2;
  asm volatile("barrier.sync %0;"
               :
               : "r"(barrier)
               : "memory");
}

EXTERN void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t Mask) {
  __nvvm_bar_warp_sync(Mask);
}

// NVPTX specific kernel initialization
EXTERN void __kmpc_impl_target_init() { /* nvptx needs no extra setup */
}

// Barrier until num_threads arrive.
EXTERN void __kmpc_impl_named_sync(uint32_t num_threads) {
  // The named barrier for active parallel threads of a team in an L1 parallel
  // region to synchronize with each other.
  int barrier = 1;
  asm volatile("barrier.sync %0, %1;"
               :
               : "r"(barrier), "r"(num_threads)
               : "memory");
}

EXTERN void __kmpc_impl_threadfence() { __nvvm_membar_gl(); }
EXTERN void __kmpc_impl_threadfence_block() { __nvvm_membar_cta(); }
EXTERN void __kmpc_impl_threadfence_system() { __nvvm_membar_sys(); }

// Calls to the NVPTX layer (assuming 1D layout)
EXTERN int __kmpc_get_hardware_thread_id_in_block() {
  return __nvvm_read_ptx_sreg_tid_x();
}
EXTERN int GetBlockIdInKernel() { return __nvvm_read_ptx_sreg_ctaid_x(); }
EXTERN int __kmpc_get_hardware_num_blocks() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}
EXTERN int __kmpc_get_hardware_num_threads_in_block() {
  return __nvvm_read_ptx_sreg_ntid_x();
}
EXTERN unsigned GetWarpId() {
  return __kmpc_get_hardware_thread_id_in_block() / WARPSIZE;
}
EXTERN unsigned GetWarpSize() { return WARPSIZE; }
EXTERN unsigned GetLaneId() {
  return __kmpc_get_hardware_thread_id_in_block() & (WARPSIZE - 1);
}

// Atomics
uint32_t __kmpc_atomic_add(uint32_t *Address, uint32_t Val) {
  return __atomic_fetch_add(Address, Val, __ATOMIC_SEQ_CST);
}
uint32_t __kmpc_atomic_inc(uint32_t *Address, uint32_t Val) {
  return __nvvm_atom_inc_gen_ui(Address, Val);
}

uint32_t __kmpc_atomic_max(uint32_t *Address, uint32_t Val) {
  return __atomic_fetch_max(Address, Val, __ATOMIC_SEQ_CST);
}

uint32_t __kmpc_atomic_exchange(uint32_t *Address, uint32_t Val) {
  uint32_t R;
  __atomic_exchange(Address, &Val, &R, __ATOMIC_SEQ_CST);
  return R;
}

uint32_t __kmpc_atomic_cas(uint32_t *Address, uint32_t Compare, uint32_t Val) {
  (void)__atomic_compare_exchange(Address, &Compare, &Val, false,
                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return Compare;
}

unsigned long long __kmpc_atomic_exchange(unsigned long long *Address,
                                          unsigned long long Val) {
  unsigned long long R;
  __atomic_exchange(Address, &Val, &R, __ATOMIC_SEQ_CST);
  return R;
}

unsigned long long __kmpc_atomic_add(unsigned long long *Address,
                                     unsigned long long Val) {
  return __atomic_fetch_add(Address, Val, __ATOMIC_SEQ_CST);
}

#define __OMP_SPIN 1000
#define UNSET 0u
#define SET 1u

EXTERN void __kmpc_impl_init_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

EXTERN void __kmpc_impl_destroy_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

EXTERN void __kmpc_impl_set_lock(omp_lock_t *lock) {
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

EXTERN void __kmpc_impl_unset_lock(omp_lock_t *lock) {
  (void)__kmpc_atomic_exchange(lock, UNSET);
}

EXTERN int __kmpc_impl_test_lock(omp_lock_t *lock) {
  return __kmpc_atomic_add(lock, 0u);
}

EXTERN void *__kmpc_impl_malloc(size_t x) { return malloc(x); }
EXTERN void __kmpc_impl_free(void *x) { free(x); }

#pragma omp end declare target
