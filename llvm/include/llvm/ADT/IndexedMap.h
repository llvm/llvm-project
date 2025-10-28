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
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace llvm {

namespace detail {
template <class Ty> struct IdentityIndex {
  using argument_type = Ty;

  Ty &operator()(Ty &self) const { return self; }
  const Ty &operator()(const Ty &self) const { return self; }
};
} // namespace detail

template <typename T, typename ToIndexT = detail::IdentityIndex<unsigned>>
class IndexedMap {
  using IndexT = typename ToIndexT::argument_type;
  // Prefer SmallVector with zero inline storage over std::vector. IndexedMaps
  // can grow very large and SmallVector grows more efficiently as long as T
  // is trivially copyable.
  using StorageT = SmallVector<T, 0>;

  StorageT Storage;
  T NullVal = T();
  ToIndexT ToIndex;

public:
  IndexedMap() = default;

  explicit IndexedMap(const T &Val) : NullVal(Val) {}

  typename StorageT::reference operator[](IndexT N) {
    assert(ToIndex(N) < Storage.size() && "index out of bounds!");
    return Storage[ToIndex(N)];
  }

  typename StorageT::const_reference operator[](IndexT N) const {
    assert(ToIndex(N) < Storage.size() && "index out of bounds!");
    return Storage[ToIndex(N)];
  }

  void reserve(typename StorageT::size_type S) { Storage.reserve(S); }

  void resize(typename StorageT::size_type S) { Storage.resize(S, NullVal); }

  void clear() { Storage.clear(); }

  void grow(IndexT N) {
    unsigned NewSize = ToIndex(N) + 1;
    if (NewSize > Storage.size())
      resize(NewSize);
  }

  bool inBounds(IndexT N) const { return ToIndex(N) < Storage.size(); }

  typename StorageT::size_type size() const { return Storage.size(); }
};

} // namespace llvm

#endif // LLVM_ADT_INDEXEDMAP_H
