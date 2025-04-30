//===-- atomic.c - Implement support functions for atomic operations.------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This code is partially copied from compiler-rt/lib/builtins/atomic.c.
//  atomic.c in next-ir-lib should be used to implement atomic operations
//  which are not supported in the NextSilicon hardware.
//
//===----------------------------------------------------------------------===//
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef SPINLOCK_COUNT
#define SPINLOCK_COUNT (1 << 10)
#endif
static const long SPINLOCK_MASK = SPINLOCK_COUNT - 1;

typedef _Atomic(uintptr_t) Lock;
/// locks for atomic operations
static Lock locks[SPINLOCK_COUNT];
// The locks array should be unique for application and each pointer should have
// a unique lock element. To achieve this, we use a hash function to get the
// lock element for a given pointer. In Nextsilicon system, it may be best to
// CodeGen the size of locks array based on the number of converted libcalls.

/// Unlock a lock.  This is a release operation.
__inline static void unlock(Lock *l) {
  __c11_atomic_store(l, 0, __ATOMIC_RELEASE);
}
/// Locks a lock.  In the current implementation, this is potentially
/// unbounded in the contended case.
__inline static void lock(Lock *l) {
  uintptr_t old = 0;
  while (!__c11_atomic_compare_exchange_weak(l, &old, 1, __ATOMIC_ACQUIRE,
                                             __ATOMIC_RELAXED))
    old = 0;
}

/// Returns a lock to use for a given pointer.
static __inline Lock *lock_for_pointer(void *ptr) {
  intptr_t hash = (intptr_t)ptr;
  // Disregard the lowest 4 bits.  We want all values that may be part of the
  // same memory operation to hash to the same value and therefore use the same
  // lock.
  hash >>= 4;
  // Use the next bits as the basis for the hash
  intptr_t low = hash & SPINLOCK_MASK;
  // Now use the high(er) set of bits to perturb the hash, so that we don't
  // get collisions from atomic fields in a single object
  hash >>= 16;
  hash ^= low;
  // Return a pointer to the word to use
  return locks + (hash & SPINLOCK_MASK);
}

#define OPTIMISED_CASE(n, type)                                                \
  bool __atomic_compare_exchange_##n(type *ptr, type *expected, type desired,  \
                                     int success, int failure) {               \
    Lock *l = lock_for_pointer(ptr);                                           \
    lock(l);                                                                   \
    if (*ptr == *expected) {                                                   \
      *ptr = desired;                                                          \
      unlock(l);                                                               \
      return true;                                                             \
    }                                                                          \
    *expected = *ptr;                                                          \
    unlock(l);                                                                 \
    return false;                                                              \
  }
OPTIMISED_CASE(16, __uint128_t)
#undef OPTIMISED_CASE

// Wrapper to match the signature of the libcalls introduced by the compiler
// (IRFixup pass).
bool __ns_atomic_compare_exchange_16(__uint128_t *ptr, __uint128_t *expected,
                                     __uint128_t desired) {
  return __atomic_compare_exchange_16(ptr, expected, desired, 0, 0);
}

#ifdef __cplusplus
}
#endif
