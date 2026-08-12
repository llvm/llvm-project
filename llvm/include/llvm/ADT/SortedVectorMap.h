//===- llvm/ADT/SortedVectorMap.h - Map backed by SmallVector *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a map backed by a sorted SmallVector. It provides a
/// std::map-like interface with binary search lookup while maintaining
/// contiguous memory layout and dense L1 cache density.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SORTEDVECTORMAP_H
#define LLVM_ADT_SORTEDVECTORMAP_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <functional>
#include <utility>

namespace llvm {

/// A map implementation backed by a sorted SmallVector.
/// Key-value pairs are stored in contiguous memory ordered by \p KeyCompare.
template <typename KeyT, typename ValueT, unsigned N = 0,
          typename KeyCompare = std::less<KeyT>>
class SortedVectorMap {
public:
  using key_type = KeyT;
  using mapped_type = ValueT;
  using value_type = std::pair<KeyT, ValueT>;
  using VectorType = SmallVector<value_type, N>;
  using size_type = typename VectorType::size_type;

  using iterator = typename VectorType::iterator;
  using const_iterator = typename VectorType::const_iterator;

private:
  VectorType Vector;
  LLVM_NO_UNIQUE_ADDRESS KeyCompare Comp;

  template <typename K1, typename K2>
  bool is_equal(const K1 &A, const K2 &B) const {
    return !Comp(A, B) && !Comp(B, A);
  }

  template <typename K> const_iterator lower_bound(const K &Key) const {
    return llvm::lower_bound(Vector, Key,
                             [this](const value_type &E, const K &KeyVal) {
                               return Comp(E.first, KeyVal);
                             });
  }

  template <typename K>
  std::pair<const_iterator, bool> find_or_insert_location(const K &Key) const {
    if (!Vector.empty() && Comp(Vector.back().first, Key))
      return {Vector.end(), false};
    auto It = lower_bound(Key);
    bool Found = (It != Vector.end() && is_equal(Key, It->first));
    return {It, Found};
  }

  template <typename K>
  std::pair<iterator, bool> find_or_insert_location(const K &Key) {
    auto [ConstIt, Found] = std::as_const(*this).find_or_insert_location(Key);
    return {Vector.begin() + (ConstIt - Vector.begin()), Found};
  }

public:
  SortedVectorMap() = default;

  // Iterators
  iterator begin() { return Vector.begin(); }
  iterator end() { return Vector.end(); }
  const_iterator begin() const { return Vector.begin(); }
  const_iterator end() const { return Vector.end(); }

  // Capacity
  [[nodiscard]] bool empty() const { return Vector.empty(); }
  size_type size() const { return Vector.size(); }
  size_type capacity() const { return Vector.capacity(); }
  void reserve(size_type Cap) { Vector.reserve(Cap); }

  // Element Access & Lookups

  template <typename K> const_iterator find(const K &Key) const {
    auto [It, Found] = find_or_insert_location(Key);
    return Found ? It : Vector.end();
  }

  template <typename K> iterator find(const K &Key) {
    auto [It, Found] = find_or_insert_location(Key);
    return Found ? It : Vector.end();
  }

  ValueT &operator[](const KeyT &Key) {
    auto [It, Found] = find_or_insert_location(Key);
    if (Found)
      return It->second;
    return Vector.insert(It, value_type(Key, ValueT()))->second;
  }

  ValueT &operator[](KeyT &&Key) {
    auto [It, Found] = find_or_insert_location(Key);
    if (Found)
      return It->second;
    return Vector.insert(It, value_type(std::move(Key), ValueT()))->second;
  }

  iterator erase(iterator Pos) { return Vector.erase(Pos); }
  iterator erase(const_iterator Pos) { return Vector.erase(Pos); }

  bool operator==(const SortedVectorMap &Other) const {
    return Vector == Other.Vector;
  }
};

} // namespace llvm

#endif // LLVM_ADT_SORTEDVECTORMAP_H
