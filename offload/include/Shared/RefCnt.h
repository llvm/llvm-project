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

  /// Increase the reference count atomically.
  void increase() { Refs.fetch_add(1, MemoryOrder); }

  /// Decrease the reference count and return whether it became zero. Decreasing
  /// the counter in more units than it was previously increased results in
  /// undefined behavior.
  bool decrease() {
    Ty Prev = Refs.fetch_sub(1, MemoryOrder);
    assert(Prev > 0 && "Invalid refcount");
    return (Prev == 1);
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
