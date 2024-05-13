//===-- Shared/Utils.h - Target independent OpenMP target RTL -- C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Routines and classes used to provide useful functionalities like string
// parsing and environment variables.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_UTILS_H
#define OMPTARGET_SHARED_UTILS_H

#include "llvm/ADT/StringRef.h"

#include "Debug.h"

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

/// Return the difference (in bytes) between \p Begin and \p End.
template <typename Ty = char>
ptrdiff_t getPtrDiff(const void *End, const void *Begin) {
  return reinterpret_cast<const Ty *>(End) -
         reinterpret_cast<const Ty *>(Begin);
}

/// Return \p Ptr advanced by \p Offset bytes.
template <typename Ty> Ty *advanceVoidPtr(Ty *Ptr, int64_t Offset) {
  static_assert(std::is_void<Ty>::value);
  return const_cast<char *>(reinterpret_cast<const char *>(Ptr) + Offset);
}

/// Return \p Ptr aligned to \p Alignment bytes.
template <typename Ty> Ty *alignPtr(Ty *Ptr, int64_t Alignment) {
  size_t Space = std::numeric_limits<size_t>::max();
  return std::align(Alignment, sizeof(char), Ptr, Space);
}

/// Round up \p V to a \p Boundary.
template <typename Ty> inline Ty roundUp(Ty V, Ty Boundary) {
  return (V + Boundary - 1) / Boundary * Boundary;
}

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_SHARED_UTILS_H
