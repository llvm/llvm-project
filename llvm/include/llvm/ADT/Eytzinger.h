//===- Eytzinger.h - Eytzinger Search Tree Span -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the EytzingerTableSpan class, a non-owning view of a
/// buffer formatted as a complete binary search tree in Eytzinger
/// (breadth-first) order.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_EYTZINGER_H
#define LLVM_ADT_EYTZINGER_H

#include "llvm/ADT/bit.h"
#include <cassert>
#include <cstddef>
#include <optional>

namespace llvm {

/// Non-owning view of a buffer formatted as a complete binary search tree in
/// Eytzinger (breadth-first) order.
template <typename T> class EytzingerTableSpan {
public:
  EytzingerTableSpan() = default;
  EytzingerTableSpan(const T *Data, size_t NumEntries)
      : Data(Data), NumEntries(NumEntries) {}

  [[nodiscard]] const T *data() const { return Data; }
  [[nodiscard]] bool empty() const { return !Data || NumEntries == 0; }
  [[nodiscard]] size_t size() const { return NumEntries; }
  [[nodiscard]] const T &operator[](size_t Idx) const {
    assert(Idx < NumEntries && "Index out of bounds");
    return Data[Idx];
  }

  /// Search this Eytzinger table for Target using branchless binary search.
  /// Returns the 0-based array index if found.
  ///
  /// KeyT enables heterogeneous lookups, allowing callers to search tables of
  /// endian-specific wrappers (e.g., support::ulittle64_t) using native integer
  /// keys without explicit conversions at the call site.
  template <typename KeyT = T>
  [[nodiscard]] std::optional<size_t> findIndex(const KeyT &Target) const {
    if (empty())
      return std::nullopt;
    size_t K = 1;
    while (K <= NumEntries)
      K = 2 * K + (Data[K - 1] < Target);
    K >>= llvm::countr_one(K) + 1;
    if (K >= 1 && Data[K - 1] == Target)
      return K - 1;
    return std::nullopt;
  }

private:
  const T *Data = nullptr;
  size_t NumEntries = 0;
};

} // namespace llvm

#endif // LLVM_ADT_EYTZINGER_H
