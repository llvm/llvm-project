//===-- Shared/RefCnt.h - Helper to keep track of references --- C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_REF_CNT_H
#define OMPTARGET_SHARED_REF_CNT_H

#include <atomic>
#include <cassert>
#include <limits>
#include <memory>

namespace llvm {
namespace omp {
namespace target {

/// Utility class for thread-safe reference counting. Any class that needs
/// objects' reference counting can inherit from this entity or have it as a
/// class data member.
template <typename Ty = uint32_t,
          std::memory_order MemoryOrder = std::memory_order_relaxed>
struct RefCountTy {
  /// Create a refcount object initialized to zero.
  RefCountTy() : Refs(0) {}

  ~RefCountTy() { assert(Refs == 0 && "Destroying with non-zero refcount"); }

  /// Increase the reference count atomically by \p Amount.
  void increase(Ty Amount = 1) { Refs.fetch_add(Amount, MemoryOrder); }

  /// Decrease the reference count by \p Amount and return whether it became
  /// zero. Decreasing the counter by more than it was previously increased
  /// results in undefined behavior.
  bool decrease(Ty Amount = 1) {
    Ty Prev = Refs.fetch_sub(Amount, MemoryOrder);
    assert(Prev >= Amount && "Invalid refcount");
    return (Prev == Amount);
  }

  Ty get() const { return Refs.load(MemoryOrder); }

private:
  /// The atomic reference counter.
  std::atomic<Ty> Refs;
};
} // namespace target
} // namespace omp
} // namespace llvm

#endif
