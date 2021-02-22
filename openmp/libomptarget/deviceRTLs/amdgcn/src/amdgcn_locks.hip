//===-- amdgcn_locks.hip - AMDGCN OpenMP GPU lock implementation -- HIP -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A 'thread' maps onto a lane of the wavefront. This means a per-thread lock
// cannot be implemented - if one thread gets the lock, it can't continue on to
// the next instruction in order to do anything as the other threads are waiting
// to take the lock.
// These functions will be implemented to provide the documented semantics for
// a SIMD => wavefront mapping once that is implemented.
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/debug.h"

static DEVICE void warn() {
  PRINT0(LD_ALL, "Locks are not supported in this thread mapping model");
}

DEVICE void __kmpc_impl_init_lock(omp_lock_t *) { warn(); }
DEVICE void __kmpc_impl_destroy_lock(omp_lock_t *) { warn(); }
DEVICE void __kmpc_impl_set_lock(omp_lock_t *) { warn(); }
DEVICE void __kmpc_impl_unset_lock(omp_lock_t *) { warn(); }
DEVICE int __kmpc_impl_test_lock(omp_lock_t *lock) {
  warn();
  return 0;
}

#pragma omp end declare target
