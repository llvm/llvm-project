//===-- EJitAtomic.h - Atomic wrappers for the SRE taskpool ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Centralized atomic-access wrapper for the EmbeddedJIT SRE taskpool.
//
//  The target platform (aarch64_be SRE) has no C++ thread library and the raw
//  compiler atomic builtins are not suitable to scatter throughout business
//  logic — they will eventually be redirected to platform-specific intrinsics
//  via a wrapper stub. Therefore *all* atomic accesses used by the taskpool go
//  through this single header. Taskpool logic files must never name a compiler
//  atomic builtin nor include <atomic> directly.
//
//  Implementation note: this first version is built on the GCC/Clang
//  __atomic_* builtins (NOT <atomic>) so it works in EJIT_FREESTANDING builds
//  that cannot depend on the C++ <atomic> header. To retarget a new platform,
//  replace only the small set of __atomic_* calls below; the taskpool data
//  structures and control flow stay unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITATOMIC_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITATOMIC_H

#include <cstdint>

namespace llvm {
namespace ejit {

//===----------------------------------------------------------------------===//
// EJitAtomic<T>
//
// Minimal, fixed-width atomic cell. Only the integer/pointer-sized types that
// the taskpool needs are exercised (uint32_t, uint64_t, uintptr_t). Memory
// orders are spelled out explicitly at every call site below so a port can
// reason about ordering when swapping in platform intrinsics.
//===----------------------------------------------------------------------===//
template <typename T> class EJitAtomic {
public:
  // Trivial default constructor: leaves value_ UNINITIALIZED for automatic /
  // dynamic storage, but ZERO for static storage (a namespace-scope object of a
  // type built from EJitAtomic lands in .bss and is zero-filled by the loader,
  // with NO C++ dynamic initializer / .init_array / startup memset). This is
  // what lets the cross-core shared state blob (EJitSharedTaskPoolState) be a
  // real, implicit-lifetime object that no core's ctor ever re-zeros — only the
  // elected owner field-initializes it (spec §11). Holders that need a defined
  // initial value on the stack/heap MUST value-initialize ({}) or store
  // explicitly; the taskpool data structures already do.
  EJitAtomic() = default;
  explicit EJitAtomic(T init) : value_(init) {}

  // Atomic cells are not copyable/movable: they identify a fixed memory slot.
  EJitAtomic(const EJitAtomic &) = delete;
  EJitAtomic &operator=(const EJitAtomic &) = delete;

  /// Acquire load — pairs with storeRelease() to publish dependent writes.
  T loadAcquire() const { return __atomic_load_n(&value_, __ATOMIC_ACQUIRE); }

  /// Relaxed load — no ordering, for diagnostics / approximate reads.
  T loadRelaxed() const { return __atomic_load_n(&value_, __ATOMIC_RELAXED); }

  /// Release store — publishes prior writes to an acquiring reader.
  void storeRelease(T v) { __atomic_store_n(&value_, v, __ATOMIC_RELEASE); }

  /// Relaxed store — no ordering.
  void storeRelaxed(T v) { __atomic_store_n(&value_, v, __ATOMIC_RELAXED); }

  /// Strong compare-exchange. On success the cell becomes \p desired and the
  /// call returns true (acq_rel). On failure \p expected is updated with the
  /// observed value and the call returns false (acquire).
  bool compareExchange(T &expected, T desired) {
    return __atomic_compare_exchange_n(&value_, &expected, desired,
                                       /*weak=*/false, __ATOMIC_ACQ_REL,
                                       __ATOMIC_ACQUIRE);
  }

  /// Atomic add, returning the previous value (acq_rel).
  T fetchAdd(T v) { return __atomic_fetch_add(&value_, v, __ATOMIC_ACQ_REL); }

  /// Atomic subtract, returning the previous value (acq_rel).
  T fetchSub(T v) { return __atomic_fetch_sub(&value_, v, __ATOMIC_ACQ_REL); }

  /// Atomic bitwise OR, returning the previous value (acq_rel).
  T fetchOr(T v) { return __atomic_fetch_or(&value_, v, __ATOMIC_ACQ_REL); }

private:
  // mutable so const load helpers can form a non-const pointer for the builtin.
  mutable T value_;
};

//===----------------------------------------------------------------------===//
// Fixed-width convenience aliases used by the taskpool data structures. Keeping
// the concrete widths explicit avoids endian/padding surprises on aarch64_be:
// every atomic field is a single naturally-aligned scalar, never a bitfield.
//===----------------------------------------------------------------------===//
using EJitAtomicU8 = EJitAtomic<uint8_t>;
using EJitAtomicU32 = EJitAtomic<uint32_t>;
using EJitAtomicU64 = EJitAtomic<uint64_t>;
using EJitAtomicUPtr = EJitAtomic<uintptr_t>;

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITATOMIC_H
