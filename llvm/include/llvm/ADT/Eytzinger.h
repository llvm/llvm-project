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

#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

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

  /// Search this Eytzinger table for Target. Returns the 0-based array index if
  /// found.
  ///
  /// KeyT enables heterogeneous lookups, allowing callers to search tables of
  /// endian-specific wrappers (e.g., support::ulittle64_t) using native integer
  /// keys without explicit conversions at the call site.
  template <typename KeyT = T>
  [[nodiscard]] std::optional<size_t> findIndex(const KeyT &Target) const {
    size_t I = 0;
    while (I < NumEntries) {
      if (Data[I] == Target)
        return I;
      I = 2 * I + 1 + (Data[I] < Target);
    }
    return std::nullopt;
  }

  /// Check if this Eytzinger table contains Target.
  template <typename KeyT = T>
  [[nodiscard]] bool contains(const KeyT &Target) const {
    return findIndex(Target).has_value();
  }

  /// Verify whether the buffer satisfies strictly ascending binary search tree
  /// order in Eytzinger layout. Runs iteratively in O(N) time and O(1) space.
  [[nodiscard]] bool isSorted() const {
    if (empty())
      return true;

    auto Left = [](size_t I) { return 2 * I + 1; };
    auto Right = [](size_t I) { return 2 * I + 2; };
    auto Parent = [](size_t I) { return (I - 1) / 2; };
    auto IsRightChild = [](size_t I) { return I > 0 && I % 2 == 0; };
    auto HasLeft = [&](size_t I) { return Left(I) < NumEntries; };
    auto HasRight = [&](size_t I) { return Right(I) < NumEntries; };

    // Start at the leftmost leaf (in-order minimum).
    size_t Curr = 0;
    while (HasLeft(Curr))
      Curr = Left(Curr);

    const T *Prev = nullptr;
    while (Curr < NumEntries) {
      if (Prev && !(*Prev < Data[Curr]))
        return false;
      Prev = &Data[Curr];

      // Step to the in-order successor of Curr.
      if (HasRight(Curr)) {
        // If Curr has a right subtree, successor is its leftmost leaf.
        Curr = Right(Curr);
        while (HasLeft(Curr))
          Curr = Left(Curr);
      } else {
        // Otherwise, walk upward while we are in the right branch.
        while (IsRightChild(Curr))
          Curr = Parent(Curr);
        // Traversed the entire tree; done.
        if (Curr == 0)
          break;
        // Step up from left child to parent.
        Curr = Parent(Curr);
      }
    }
    return true;
  }

private:
  const T *Data = nullptr;
  size_t NumEntries = 0;
};

/// Owning container that stores elements in a complete binary search tree
/// formatted in Eytzinger (breadth-first) order.
template <typename T> class EytzingerTable {
  std::vector<T> Storage;

  explicit EytzingerTable(std::vector<T> Buffer) : Storage(std::move(Buffer)) {}

public:
  EytzingerTable() = default;

  /// Construct an Eytzinger search tree from a vector of keys by sorting,
  /// deduplicating, and reordering elements into breadth-first layout.
  /// KeyT may differ from T (e.g., creating EytzingerTable<ulittle64_t> from
  /// a vector of uint64_t).
  template <typename KeyT = T>
  static EytzingerTable<T> create(std::vector<KeyT> Keys) {
    llvm::sort(Keys);
    Keys.erase(llvm::unique(Keys), Keys.end());

    std::vector<T> Eytzinger(Keys.size());
    size_t InIdx = 0;
    auto EytzingerInOrder = [&](auto &Self, size_t K) -> void {
      if (K > Keys.size())
        return;
      Self(Self, 2 * K);
      Eytzinger[K - 1] = T(std::move(Keys[InIdx++]));
      Self(Self, 2 * K + 1);
    };
    EytzingerInOrder(EytzingerInOrder, 1);
    return EytzingerTable<T>(std::move(Eytzinger));
  }

  [[nodiscard]] EytzingerTableSpan<T> asSpan() const {
    return EytzingerTableSpan<T>(Storage.data(), Storage.size());
  }

  template <typename KeyT = T>
  [[nodiscard]] std::optional<size_t> findIndex(const KeyT &Target) const {
    return asSpan().findIndex(Target);
  }

  template <typename KeyT = T>
  [[nodiscard]] bool contains(const KeyT &Target) const {
    return asSpan().contains(Target);
  }

  [[nodiscard]] bool isSorted() const { return asSpan().isSorted(); }

  [[nodiscard]] const T *data() const { return Storage.data(); }
  [[nodiscard]] size_t size() const { return Storage.size(); }
  [[nodiscard]] bool empty() const { return Storage.empty(); }
  [[nodiscard]] const T &operator[](size_t Idx) const {
    assert(Idx < Storage.size() && "Index out of bounds");
    return Storage[Idx];
  }
};

} // namespace llvm

#endif // LLVM_ADT_EYTZINGER_H
