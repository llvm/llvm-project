//===- llvm/ADT/IndexedMap.h - An index map implementation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements an indexed map. The index map template takes two
/// types. The first is the mapped type and the second is a functor
/// that maps its argument to a size_t. On instantiation a "null" value
/// can be provided to be used as a "does not exist" indicator in the
/// map. A member function grow() is provided that given the value of
/// the maximally indexed key (the argument of the functor) makes sure
/// the map has enough space for it.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_INDEXEDMAP_H
#define LLVM_ADT_INDEXEDMAP_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <type_traits>

namespace llvm {

namespace detail {
// Helper to compute the IndexT for IndexedMap.
template <typename ToIndexT, typename = void> struct DeduceIndexType {
  using type =
      std::conditional_t<std::is_same_v<ToIndexT, llvm::identity_cxx20>,
                         unsigned, void>;
};

template <typename ToIndexT>
struct DeduceIndexType<ToIndexT,
                       std::void_t<typename ToIndexT::argument_type>> {
  using type = typename ToIndexT::argument_type;
};
} // namespace detail

template <typename T, typename ToIndexT = llvm::identity_cxx20,
          typename IndexT = typename detail::DeduceIndexType<ToIndexT>::type>
class IndexedMap {
  static_assert(!std::is_same_v<IndexT, void>,
                "Could not deduce index type from the provided functor.");
  // Prefer SmallVector with zero inline storage over std::vector. IndexedMaps
  // can grow very large and SmallVector grows more efficiently as long as T
  // is trivially copyable.
  using StorageT = SmallVector<T, 0>;

  StorageT storage_;
  T nullVal_;
  ToIndexT toIndex_;

public:
  IndexedMap() : nullVal_(T()) {}

  explicit IndexedMap(const T &val) : nullVal_(val) {}

  typename StorageT::reference operator[](IndexT n) {
    assert(toIndex_(n) < storage_.size() && "index out of bounds!");
    return storage_[toIndex_(n)];
  }

  typename StorageT::const_reference operator[](IndexT n) const {
    assert(toIndex_(n) < storage_.size() && "index out of bounds!");
    return storage_[toIndex_(n)];
  }

  void reserve(typename StorageT::size_type s) { storage_.reserve(s); }

  void resize(typename StorageT::size_type s) { storage_.resize(s, nullVal_); }

  void clear() { storage_.clear(); }

  void grow(IndexT n) {
    unsigned NewSize = toIndex_(n) + 1;
    if (NewSize > storage_.size())
      resize(NewSize);
  }

  bool inBounds(IndexT n) const { return toIndex_(n) < storage_.size(); }

  typename StorageT::size_type size() const { return storage_.size(); }
};

} // namespace llvm

#endif // LLVM_ADT_INDEXEDMAP_H
