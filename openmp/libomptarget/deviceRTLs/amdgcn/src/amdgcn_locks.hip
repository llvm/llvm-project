//===-- amdgcn_locks.hip - AMDGCN OpenMP GPU lock implementation -- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of openmp lock functions
// A 'thread' maps onto a lane of the wavefront. This means a per-thread lock
// cannot be implemented - if one thread gets the lock, it can't continue on to
// the next instruction in order to do anything as the other threads are waiting
// to take the lock
// The closest approximatation we can implement is to lock per-wavefront.
//
//===----------------------------------------------------------------------===//

#include "common/support.h"
#include "common/target_atomic.h"
#include "target_impl.h"

#define UNSET 0u
#define SET 1u

DEVICE void __kmpc_impl_init_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

DEVICE void __kmpc_impl_destroy_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

DEVICE void __kmpc_impl_set_lock(omp_lock_t *lock) {
  uint64_t lowestActiveThread = __kmpc_impl_ffs(__kmpc_impl_activemask()) - 1;
  if (GetLaneId() == lowestActiveThread) {
    while (__kmpc_atomic_cas(lock, UNSET, SET) != UNSET) {
      __builtin_amdgcn_s_sleep(0);
    }
  }
  // test_lock will now return true for any thread in the warp
}

DEVICE void __kmpc_impl_unset_lock(omp_lock_t *lock) {
  // Could be an atomic store of UNSET
  (void)__kmpc_atomic_exchange(lock, UNSET);
}

DEVICE int __kmpc_impl_test_lock(omp_lock_t *lock) {
  // Could be an atomic load
  return __kmpc_atomic_add(lock, 0u);
}
