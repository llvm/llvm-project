//===- llvm/ADT/EnumeratedArray.h - Enumerated Array-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines an array type that can be indexed using scoped enum
/// values.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ENUMERATEDARRAY_H
#define LLVM_ADT_ENUMERATEDARRAY_H

#include "llvm/ADT/STLExtras.h"
#include <array>
#include <cassert>

namespace llvm {

template <typename ValueType, typename Enumeration,
          Enumeration LargestEnum = Enumeration::Last, typename IndexType = int,
          IndexType Size = 1 + static_cast<IndexType>(LargestEnum)>
class EnumeratedArray {
  static_assert(Size > 0);
  using ArrayTy = std::array<ValueType, Size>;
  ArrayTy Underlying;

public:
  using iterator = typename ArrayTy::iterator;
  using const_iterator = typename ArrayTy::const_iterator;
  using reverse_iterator = typename ArrayTy::reverse_iterator;
  using const_reverse_iterator = typename ArrayTy::const_reverse_iterator;

  using value_type = ValueType;
  using reference = ValueType &;
  using const_reference = const ValueType &;
  using pointer = ValueType *;
  using const_pointer = const ValueType *;

  EnumeratedArray() = default;
  EnumeratedArray(ValueType V) { Underlying.fill(V); }
  EnumeratedArray(std::initializer_list<ValueType> Init) {
    assert(Init.size() == Size && "Incorrect initializer size");
    llvm::copy(Init, Underlying.begin());
  }

  const ValueType &operator[](Enumeration Index) const {
    auto IX = static_cast<IndexType>(Index);
    assert(IX >= 0 && IX < Size && "Index is out of bounds.");
    return Underlying[IX];
  }
  ValueType &operator[](Enumeration Index) {
    return const_cast<ValueType &>(
        static_cast<const EnumeratedArray &>(*this)[Index]);
  }
  IndexType size() const { return Size; }
  bool empty() const { return size() == 0; }

  iterator begin() { return Underlying.begin(); }
  const_iterator begin() const { return Underlying.begin(); }
  iterator end() { return Underlying.end(); }
  const_iterator end() const { return Underlying.end(); }

  reverse_iterator rbegin() { return Underlying.rbegin(); }
  const_reverse_iterator rbegin() const { return Underlying.rbegin(); }
  reverse_iterator rend() { return Underlying.rend(); }
  const_reverse_iterator rend() const { return Underlying.rend(); }
};

} // namespace llvm

#endif // LLVM_ADT_ENUMERATEDARRAY_H
