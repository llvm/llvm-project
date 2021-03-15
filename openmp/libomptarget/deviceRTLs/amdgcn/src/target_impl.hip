//===------- target_impl.hip - AMDGCN OpenMP GPU implementation --- HIP -*-===//
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

#include "common/omptarget.h"
#include "target_impl.h"
#include "target_interface.h"

// Implementations initially derived from hcc

// Initialized with a 64-bit mask with bits set in positions less than the
// thread's lane number in the warp
EXTERN __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  uint32_t lane = GetLaneId();
  int64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = ((uint64_t)1 << lane) - (uint64_t)1;
  return mask & ballot;
}

// Initialized with a 64-bit mask with bits set in positions greater than the
// thread's lane number in the warp
EXTERN __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  uint32_t lane = GetLaneId();
  if (lane == (WARPSIZE - 1))
    return 0;
  uint64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = (~((uint64_t)0)) << (lane + 1);
  return mask & ballot;
}

EXTERN double __kmpc_impl_get_wtick() { return ((double)1E-9); }

EXTERN double __kmpc_impl_get_wtime() {
  // The intrinsics for measuring time have undocumented frequency
  // This will probably need to be found by measurement on a number of
  // architectures. Until then, return 0, which is very inaccurate as a
  // timer but resolves the undefined symbol at link time.
  return 0;
}

// Warp vote function
EXTERN __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  return __builtin_amdgcn_read_exec();
}

uint32_t __kmpc_L1_Barrier [[clang::loader_uninitialized]];
#pragma allocate(__kmpc_L1_Barrier) allocator(omp_pteam_mem_alloc)

EXTERN void __kmpc_impl_target_init() {
  // Don't have global ctors, and shared memory is not zero init
  __atomic_store_n(&__kmpc_L1_Barrier, 0u, __ATOMIC_RELEASE);
}

EXTERN void __kmpc_impl_named_sync(uint32_t num_threads) {
  __atomic_thread_fence(__ATOMIC_ACQUIRE);

  uint32_t num_waves = num_threads / WARPSIZE;

  // Partial barrier implementation for amdgcn.
  // Uses two 16 bit unsigned counters. One for the number of waves to have
  // reached the barrier, and one to count how many times the barrier has been
  // passed. These are packed in a single atomically accessed 32 bit integer.
  // Low bits for the number of waves, assumed zero before this call.
  // High bits to count the number of times the barrier has been passed.

  // precondition: num_waves != 0;
  // invariant: num_waves * WARPSIZE == num_threads;
  // precondition: num_waves < 0xffffu;

  // Increment the low 16 bits once, using the lowest active thread.
  uint64_t lowestActiveThread = __kmpc_impl_ffs(__kmpc_impl_activemask()) - 1;
  bool isLowest = GetLaneId() == lowestActiveThread;

  if (isLowest) {
    uint32_t load = __atomic_fetch_add(&__kmpc_L1_Barrier, 1,
                                       __ATOMIC_RELAXED); // commutative

    // Record the number of times the barrier has been passed
    uint32_t generation = load & 0xffff0000u;

    if ((load & 0x0000ffffu) == (num_waves - 1)) {
      // Reached num_waves in low bits so this is the last wave.
      // Set low bits to zero and increment high bits
      load += 0x00010000u; // wrap is safe
      load &= 0xffff0000u; // because bits zeroed second

      // Reset the wave counter and release the waiting waves
      __atomic_store_n(&__kmpc_L1_Barrier, load, __ATOMIC_RELAXED);
    } else {
      // more waves still to go, spin until generation counter changes
      do {
        __builtin_amdgcn_s_sleep(0);
        load = __atomic_load_n(&__kmpc_L1_Barrier, __ATOMIC_RELAXED);
      } while ((load & 0xffff0000u) == generation);
    }
  }
  __atomic_thread_fence(__ATOMIC_RELEASE);
}

namespace {
DEVICE uint32_t get_grid_dim(uint32_t n, uint16_t d) {
  uint32_t q = n / d;
  return q + (n > q * d);
}
DEVICE uint32_t get_workgroup_dim(uint32_t group_id, uint32_t grid_size,
                                  uint16_t group_size) {
  uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}
} // namespace

EXTERN int GetNumberOfBlocksInKernel() {
  return get_grid_dim(__builtin_amdgcn_grid_size_x(),
                      __builtin_amdgcn_workgroup_size_x());
}

EXTERN int GetNumberOfThreadsInBlock() {
  return get_workgroup_dim(__builtin_amdgcn_workgroup_id_x(),
                           __builtin_amdgcn_grid_size_x(),
                           __builtin_amdgcn_workgroup_size_x());
}

EXTERN unsigned GetWarpId() { return GetThreadIdInBlock() / WARPSIZE; }
EXTERN unsigned GetWarpSize() { return WARPSIZE; }
EXTERN unsigned GetLaneId() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

EXTERN uint32_t __kmpc_amdgcn_gpu_num_threads() {
  return GetNumberOfThreadsInBlock();
}

// Atomics
DEVICE uint32_t __kmpc_atomic_add(uint32_t *Address, uint32_t Val) {
  return __atomic_fetch_add(Address, Val, __ATOMIC_SEQ_CST);
}
DEVICE uint32_t __kmpc_atomic_inc(uint32_t *Address, uint32_t Val) {
  return __builtin_amdgcn_atomic_inc32(Address, Val, __ATOMIC_SEQ_CST, "");
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
                                  __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
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

// Stub implementations
EXTERN void *__kmpc_impl_malloc(size_t) { return nullptr; }
EXTERN void __kmpc_impl_free(void *) {}

EXTERN void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi) {
  lo = (uint32_t)(val & UINT64_C(0x00000000FFFFFFFF));
  hi = (uint32_t)((val & UINT64_C(0xFFFFFFFF00000000)) >> 32);
}

EXTERN uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi) {
  return (((uint64_t)hi) << 32) | (uint64_t)lo;
}

EXTERN void __kmpc_impl_syncthreads() { __builtin_amdgcn_s_barrier(); }

EXTERN void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t) {
  // AMDGCN doesn't need to sync threads in a warp
}

EXTERN void __kmpc_impl_threadfence() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

EXTERN void __kmpc_impl_threadfence_block() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
}

EXTERN void __kmpc_impl_threadfence_system() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}

// Calls to the AMDGCN layer (assuming 1D layout)
EXTERN int GetThreadIdInBlock() { return __builtin_amdgcn_workitem_id_x(); }
EXTERN int GetBlockIdInKernel() { return __builtin_amdgcn_workgroup_id_x(); }

#pragma omp end declare target
