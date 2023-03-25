//===- Synchronization.h - OpenMP synchronization utilities ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_SYNCHRONIZATION_H
#define OMPTARGET_DEVICERTL_SYNCHRONIZATION_H

#include "Types.h"

namespace ompx {

namespace atomic {

enum OrderingTy {
  relaxed = __ATOMIC_RELAXED,
  aquire = __ATOMIC_ACQUIRE,
  release = __ATOMIC_RELEASE,
  acq_rel = __ATOMIC_ACQ_REL,
  seq_cst = __ATOMIC_SEQ_CST,
};

/// Atomically increment \p *Addr and wrap at \p V with \p Ordering semantics.
uint32_t inc(uint32_t *Addr, uint32_t V, OrderingTy Ordering);

/// Atomically perform <op> on \p V and \p *Addr with \p Ordering semantics. The
/// result is stored in \p *Addr;
/// {

#define ATOMIC_COMMON_OP(TY)                                                   \
  TY add(TY *Addr, TY V, OrderingTy Ordering);                                 \
  TY mul(TY *Addr, TY V, OrderingTy Ordering);                                 \
  TY load(TY *Addr, OrderingTy Ordering);                                      \
  void store(TY *Addr, TY V, OrderingTy Ordering);                             \
  bool cas(TY *Addr, TY ExpectedV, TY DesiredV, OrderingTy OrderingSucc,       \
           OrderingTy OrderingFail);

#define ATOMIC_FP_ONLY_OP(TY)                                                  \
  TY min(TY *Addr, TY V, OrderingTy Ordering);                                 \
  TY max(TY *Addr, TY V, OrderingTy Ordering);

#define ATOMIC_INT_ONLY_OP(TY)                                                 \
  TY min(TY *Addr, TY V, OrderingTy Ordering);                                 \
  TY max(TY *Addr, TY V, OrderingTy Ordering);                                 \
  TY bit_or(TY *Addr, TY V, OrderingTy Ordering);                              \
  TY bit_and(TY *Addr, TY V, OrderingTy Ordering);                             \
  TY bit_xor(TY *Addr, TY V, OrderingTy Ordering);

#define ATOMIC_FP_OP(TY)                                                       \
  ATOMIC_FP_ONLY_OP(TY)                                                        \
  ATOMIC_COMMON_OP(TY)

#define ATOMIC_INT_OP(TY)                                                      \
  ATOMIC_INT_ONLY_OP(TY)                                                       \
  ATOMIC_COMMON_OP(TY)

// This needs to be kept in sync with the header. Also the reason we don't use
// templates here.
ATOMIC_INT_OP(int8_t)
ATOMIC_INT_OP(int16_t)
ATOMIC_INT_OP(int32_t)
ATOMIC_INT_OP(int64_t)
ATOMIC_INT_OP(uint8_t)
ATOMIC_INT_OP(uint16_t)
ATOMIC_INT_OP(uint32_t)
ATOMIC_INT_OP(uint64_t)
ATOMIC_FP_OP(float)
ATOMIC_FP_OP(double)

#undef ATOMIC_INT_ONLY_OP
#undef ATOMIC_FP_ONLY_OP
#undef ATOMIC_COMMON_OP
#undef ATOMIC_INT_OP
#undef ATOMIC_FP_OP

///}

} // namespace atomic

namespace synchronize {

/// Initialize the synchronization machinery. Must be called by all threads.
void init(bool IsSPMD);

/// Synchronize all threads in a warp identified by \p Mask.
void warp(LaneMaskTy Mask);

/// Synchronize all threads in a block and perform a fence before and after the
/// barrier according to \p Ordering. Note that the fence might be part of the
/// barrier.
void threads(atomic::OrderingTy Ordering);

/// Synchronizing threads is allowed even if they all hit different instances of
/// `synchronize::threads()`. However, `synchronize::threadsAligned()` is more
/// restrictive in that it requires all threads to hit the same instance. The
/// noinline is removed by the openmp-opt pass and helps to preserve the
/// information till then.
///{
#pragma omp begin assumes ext_aligned_barrier

/// Synchronize all threads in a block, they are reaching the same instruction
/// (hence all threads in the block are "aligned"). Also perform a fence before
/// and after the barrier according to \p Ordering. Note that the
/// fence might be part of the barrier if the target offers this.
__attribute__((noinline)) void threadsAligned(atomic::OrderingTy Ordering);

#pragma omp end assumes
///}

} // namespace synchronize

namespace fence {

/// Memory fence with \p Ordering semantics for the team.
void team(atomic::OrderingTy Ordering);

/// Memory fence with \p Ordering semantics for the contention group.
void kernel(atomic::OrderingTy Ordering);

/// Memory fence with \p Ordering semantics for the system.
void system(atomic::OrderingTy Ordering);

} // namespace fence

} // namespace ompx

#endif
