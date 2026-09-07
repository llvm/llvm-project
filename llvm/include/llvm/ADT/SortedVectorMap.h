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
/// contiguous memory layout and dense cache locality.
///
/// SortedVectorMap is intended for:
/// - Small maps where memory footprint is a primary concern. In particular, it
///   avoids the initial bucket overhead of DenseMap (e.g. 64 buckets by
///   default) when only a few elements are stored.
/// - Use cases that require iteration in sorted key order.
///
/// Trade-offs:
/// - Lookups take O(log N) time via binary search rather than O(1) in DenseMap.
/// - Insertions and deletions take O(N) time due to shifting elements in the
///   underlying vector, making it best suited for small N or mostly-read data.
/// - Compared to std::map, elements are stored contiguously, eliminating
///   per-node heap allocations and pointer chasing.
/// - Compared to MapVector, elements are ordered by key rather than insertion
///   order, with zero auxiliary hash table overhead.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SORTEDVECTORMAP_H
#define LLVM_ADT_SORTEDVECTORMAP_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <functional>
#include <tuple>
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
  using reverse_iterator = typename VectorType::reverse_iterator;
  using const_reverse_iterator = typename VectorType::const_reverse_iterator;

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

  template <typename KeyArgT, typename... Ts>
  std::pair<iterator, bool> try_emplace_impl(KeyArgT &&Key, Ts &&...Args) {
    auto [It, Found] = find_or_insert_location(Key);
    if (Found)
      return {It, false};
    It = Vector.insert(
        It, value_type(std::piecewise_construct,
                       std::forward_as_tuple(std::forward<KeyArgT>(Key)),
                       std::forward_as_tuple(std::forward<Ts>(Args)...)));
    return {It, true};
  }

public:
  SortedVectorMap() = default;

  // Iterators
  iterator begin() { return Vector.begin(); }
  iterator end() { return Vector.end(); }
  const_iterator begin() const { return Vector.begin(); }
  const_iterator end() const { return Vector.end(); }
  const_iterator cbegin() const { return Vector.begin(); }
  const_iterator cend() const { return Vector.end(); }

  reverse_iterator rbegin() { return Vector.rbegin(); }
  reverse_iterator rend() { return Vector.rend(); }
  const_reverse_iterator rbegin() const { return Vector.rbegin(); }
  const_reverse_iterator rend() const { return Vector.rend(); }
  const_reverse_iterator crbegin() const { return Vector.rbegin(); }
  const_reverse_iterator crend() const { return Vector.rend(); }

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

  template <typename... Ts>
  std::pair<iterator, bool> try_emplace(const KeyT &Key, Ts &&...Args) {
    return try_emplace_impl(Key, std::forward<Ts>(Args)...);
  }

  template <typename... Ts>
  std::pair<iterator, bool> try_emplace(KeyT &&Key, Ts &&...Args) {
    return try_emplace_impl(std::move(Key), std::forward<Ts>(Args)...);
  }

  std::pair<iterator, bool> insert(const value_type &KV) {
    return try_emplace_impl(KV.first, KV.second);
  }

  std::pair<iterator, bool> insert(value_type &&KV) {
    return try_emplace_impl(std::move(KV.first), std::move(KV.second));
  }

  ValueT &operator[](const KeyT &Key) {
    return try_emplace_impl(Key).first->second;
  }

  ValueT &operator[](KeyT &&Key) {
    return try_emplace_impl(std::move(Key)).first->second;
  }

  iterator erase(iterator Pos) { return Vector.erase(Pos); }
  iterator erase(const_iterator Pos) { return Vector.erase(Pos); }

  bool operator==(const SortedVectorMap &Other) const {
    return Vector == Other.Vector;
  }
};
} // namespace llvm

#endif // LLVM_ADT_SORTEDVECTORMAP_H
