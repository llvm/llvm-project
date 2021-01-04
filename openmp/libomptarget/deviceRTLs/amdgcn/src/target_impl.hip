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

#include "target_impl.h"

// Implementations initially derived from hcc

// Initialized with a 64-bit mask with bits set in positions less than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt() {
  uint32_t lane = GetLaneId();
  int64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = ((uint64_t)1 << lane) - (uint64_t)1;
  return mask & ballot;
}

// Initialized with a 64-bit mask with bits set in positions greater than the
// thread's lane number in the warp
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt() {
  uint32_t lane = GetLaneId();
  if (lane == (WARPSIZE - 1))
    return 0;
  uint64_t ballot = __kmpc_impl_activemask();
  uint64_t mask = (~((uint64_t)0)) << (lane + 1);
  return mask & ballot;
}

DEVICE double __kmpc_impl_get_wtick() { return ((double)1E-9); }

DEVICE double __kmpc_impl_get_wtime() {
  // The intrinsics for measuring time have undocumented frequency
  // This will probably need to be found by measurement on a number of
  // architectures. Until then, return 0, which is very inaccurate as a
  // timer but resolves the undefined symbol at link time.
  return 0;
}

// Warp vote function
DEVICE __kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  return __builtin_amdgcn_read_exec();
}

DEVICE int32_t __kmpc_impl_shfl_sync(__kmpc_impl_lanemask_t, int32_t var,
                                     int32_t srcLane) {
  int width = WARPSIZE;
  int self = GetLaneId();
  int index = srcLane + (self & ~(width - 1));
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

DEVICE int32_t __kmpc_impl_shfl_down_sync(__kmpc_impl_lanemask_t, int32_t var,
                                          uint32_t laneDelta, int32_t width) {
  int self = GetLaneId();
  int index = self + laneDelta;
  index = (int)(laneDelta + (self & (width - 1))) >= width ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

static DEVICE SHARED uint32_t L1_Barrier;

DEVICE void __kmpc_impl_target_init() {
  // Don't have global ctors, and shared memory is not zero init
  __atomic_store_n(&L1_Barrier, 0u, __ATOMIC_RELEASE);
}

DEVICE void __kmpc_impl_named_sync(uint32_t num_threads) {
  __atomic_thread_fence(__ATOMIC_ACQUIRE);

  uint32_t num_waves = num_threads / WARPSIZE;

  // Partial barrier implementation for amdgcn.
  // Uses two 16 bit unsigned counters. One for the number of waves to have
  // reached the barrier, and one to count how many times the barrier has been
  // passed. These are packed in a single atomically accessed 32 bit integer.
  // Low bits for the number of waves, assumed zero before this call.
  // High bits to count the number of times the barrier has been passed.

  assert(num_waves != 0);
  assert(num_waves * WARPSIZE == num_threads);
  assert(num_waves < 0xffffu);

  // Increment the low 16 bits once, using the lowest active thread.
  uint64_t lowestActiveThread = __kmpc_impl_ffs(__kmpc_impl_activemask()) - 1;
  bool isLowest = GetLaneId() == lowestActiveThread;

  if (isLowest) {
    uint32_t load =
        __atomic_fetch_add(&L1_Barrier, 1, __ATOMIC_RELAXED); // commutative

    // Record the number of times the barrier has been passed
    uint32_t generation = load & 0xffff0000u;

    if ((load & 0x0000ffffu) == (num_waves - 1)) {
      // Reached num_waves in low bits so this is the last wave.
      // Set low bits to zero and increment high bits
      load += 0x00010000u; // wrap is safe
      load &= 0xffff0000u; // because bits zeroed second

      // Reset the wave counter and release the waiting waves
      __atomic_store_n(&L1_Barrier, load, __ATOMIC_RELAXED);
    } else {
      // more waves still to go, spin until generation counter changes
      do {
        __builtin_amdgcn_s_sleep(0);
        load = __atomic_load_n(&L1_Barrier, __ATOMIC_RELAXED);
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

DEVICE int GetNumberOfBlocksInKernel() {
  return get_grid_dim(__builtin_amdgcn_grid_size_x(), __builtin_amdgcn_workgroup_size_x());
}

DEVICE int GetNumberOfThreadsInBlock() {
  return get_workgroup_dim(__builtin_amdgcn_workgroup_id_x(), __builtin_amdgcn_grid_size_x(),
                           __builtin_amdgcn_workgroup_size_x());
}

DEVICE unsigned GetWarpId() { return GetThreadIdInBlock() / WARPSIZE; }
DEVICE unsigned GetLaneId() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

EXTERN uint32_t __kmpc_amdgcn_gpu_num_threads() {
  return GetNumberOfThreadsInBlock();
}

// Stub implementations
DEVICE void *__kmpc_impl_malloc(size_t) { return nullptr; }
DEVICE void __kmpc_impl_free(void *) {}
