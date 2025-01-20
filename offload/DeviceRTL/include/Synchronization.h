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

#include "DeviceTypes.h"
#include "DeviceUtils.h"

#pragma omp begin declare target device_type(nohost)

namespace ompx {
namespace atomic {

enum OrderingTy {
  relaxed = __ATOMIC_RELAXED,
  aquire = __ATOMIC_ACQUIRE,
  release = __ATOMIC_RELEASE,
  acq_rel = __ATOMIC_ACQ_REL,
  seq_cst = __ATOMIC_SEQ_CST,
};

enum ScopeTy {
  system = __MEMORY_SCOPE_SYSTEM,
  device_ = __MEMORY_SCOPE_DEVICE,
  workgroup = __MEMORY_SCOPE_WRKGRP,
  wavefront = __MEMORY_SCOPE_WVFRNT,
  single = __MEMORY_SCOPE_SINGLE,
};

enum MemScopeTy {
  all,    // All threads on all devices
  device, // All threads on the device
  cgroup  // All threads in the contention group, e.g. the team
};

/// Atomically increment \p *Addr and wrap at \p V with \p Ordering semantics.
uint32_t inc(uint32_t *Addr, uint32_t V, OrderingTy Ordering,
             MemScopeTy MemScope = MemScopeTy::all);

/// Atomically perform <op> on \p V and \p *Addr with \p Ordering semantics. The
/// result is stored in \p *Addr;
/// {

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
bool cas(Ty *Address, V ExpectedV, V DesiredV, atomic::OrderingTy OrderingSucc,
         atomic::OrderingTy OrderingFail) {
  return __scoped_atomic_compare_exchange(Address, &ExpectedV, &DesiredV, false,
                                          OrderingSucc, OrderingFail,
                                          __MEMORY_SCOPE_DEVICE);
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
V add(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_add(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
V load(Ty *Address, atomic::OrderingTy Ordering) {
  return __scoped_atomic_load_n(Address, Ordering, __MEMORY_SCOPE_DEVICE);
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
void store(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  __scoped_atomic_store_n(Address, Val, Ordering, __MEMORY_SCOPE_DEVICE);
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
V mul(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  Ty TypedCurrentVal, TypedResultVal, TypedNewVal;
  bool Success;
  do {
    TypedCurrentVal = atomic::load(Address, Ordering);
    TypedNewVal = TypedCurrentVal * Val;
    Success = atomic::cas(Address, TypedCurrentVal, TypedNewVal, Ordering,
                          atomic::relaxed);
  } while (!Success);
  return TypedResultVal;
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
utils::enable_if_t<!utils::is_floating_point_v<V>, V>
max(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_max(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
utils::enable_if_t<utils::is_same_v<V, float>, V>
max(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  if (Val >= 0)
    return utils::bitCast<float>(
        max((int32_t *)Address, utils::bitCast<int32_t>(Val), Ordering));
  return utils::bitCast<float>(
      min((uint32_t *)Address, utils::bitCast<uint32_t>(Val), Ordering));
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
utils::enable_if_t<utils::is_same_v<V, double>, V>
max(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  if (Val >= 0)
    return utils::bitCast<double>(
        max((int64_t *)Address, utils::bitCast<int64_t>(Val), Ordering));
  return utils::bitCast<double>(
      min((uint64_t *)Address, utils::bitCast<uint64_t>(Val), Ordering));
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
utils::enable_if_t<!utils::is_floating_point_v<V>, V>
min(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_min(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

// TODO: Implement this with __atomic_fetch_max and remove the duplication.
template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
utils::enable_if_t<utils::is_same_v<V, float>, V>
min(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  if (Val >= 0)
    return utils::bitCast<float>(
        min((int32_t *)Address, utils::bitCast<int32_t>(Val), Ordering));
  return utils::bitCast<float>(
      max((uint32_t *)Address, utils::bitCast<uint32_t>(Val), Ordering));
}

// TODO: Implement this with __atomic_fetch_max and remove the duplication.
template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
utils::enable_if_t<utils::is_same_v<V, double>, V>
min(Ty *Address, utils::remove_addrspace_t<Ty> Val,
    atomic::OrderingTy Ordering) {
  if (Val >= 0)
    return utils::bitCast<double>(
        min((int64_t *)Address, utils::bitCast<int64_t>(Val), Ordering));
  return utils::bitCast<double>(
      max((uint64_t *)Address, utils::bitCast<uint64_t>(Val), Ordering));
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
V bit_or(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_or(Address, Val, Ordering,
                                  __MEMORY_SCOPE_DEVICE);
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
V bit_and(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_and(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

template <typename Ty, typename V = utils::remove_addrspace_t<Ty>>
V bit_xor(Ty *Address, V Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_xor(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

static inline uint32_t atomicExchange(uint32_t *Address, uint32_t Val,
                                      atomic::OrderingTy Ordering) {
  uint32_t R;
  __scoped_atomic_exchange(Address, &Val, &R, Ordering, __MEMORY_SCOPE_DEVICE);
  return R;
}

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
[[gnu::noinline]] void threadsAligned(atomic::OrderingTy Ordering);

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

#pragma omp end declare target

#endif
