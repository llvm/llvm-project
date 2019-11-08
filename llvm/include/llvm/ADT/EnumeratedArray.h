//===- llvm/ADT/EnumeratedArray.h - Enumerated Array-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an array type that can be indexed using scoped enum values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ENUMERATEDARRAY_H
#define LLVM_ADT_ENUMERATEDARRAY_H

#include <cassert>

namespace llvm {

template <typename ValueType, typename Enumeration,
          Enumeration LargestEnum = Enumeration::Last, typename IndexType = int,
          IndexType Size = 1 + static_cast<IndexType>(LargestEnum)>
class EnumeratedArray {
public:
  EnumeratedArray() = default;
  EnumeratedArray(ValueType V) {
    for (IndexType IX = 0; IX < Size; ++IX) {
      Underlying[IX] = V;
    }
  }
  inline const ValueType &operator[](const Enumeration Index) const {
    auto IX = static_cast<const IndexType>(Index);
    assert(IX >= 0 && IX < Size && "Index is out of bounds.");
    return Underlying[IX];
  }
  inline ValueType &operator[](const Enumeration Index) {
    return const_cast<ValueType &>(
        static_cast<const EnumeratedArray<ValueType, Enumeration, LargestEnum,
                                          IndexType, Size> &>(*this)[Index]);
  }

private:
  ValueType Underlying[Size];
};

} // namespace llvm

#endif // LLVM_ADT_ENUMERATEDARRAY_H
