//===- llvm/ADT/PackedVector.h - Packed values vector -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the PackedVector class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_PACKEDVECTOR_H
#define LLVM_ADT_PACKEDVECTOR_H

#include "llvm/ADT/BitVector.h"
#include <cassert>
#include <limits>

namespace llvm {

/// Store a vector of values using a specific number of bits for each
/// value. Both signed and unsigned types can be used, e.g
/// @code
///   PackedVector<signed, 2> vec;
/// @endcode
/// will create a vector accepting values -2, -1, 0, 1. Any other value will hit
/// an assertion.
template <typename T, unsigned BitNum, typename BitVectorTy = BitVector>
class PackedVector {
  static_assert(BitNum > 0, "BitNum must be > 0");

  BitVectorTy Bits;
  // Keep track of the number of elements on our own.
  // We always maintain Bits.size() == NumElements * BitNum.
  // Used to avoid an integer division in size().
  unsigned NumElements = 0;

  static T getValue(const BitVectorTy &Bits, unsigned Idx) {
    if constexpr (std::numeric_limits<T>::is_signed) {
      T val = T();
      for (unsigned i = 0; i != BitNum - 1; ++i)
        val = T(val | ((Bits[(Idx * BitNum) + i] ? 1UL : 0UL) << i));
      if (Bits[(Idx * BitNum) + BitNum - 1])
        val = ~val;
      return val;
    } else {
      T val = T();
      for (unsigned i = 0; i != BitNum; ++i)
        val = T(val | ((Bits[(Idx * BitNum) + i] ? 1UL : 0UL) << i));
      return val;
    }
  }

  static void setValue(BitVectorTy &Bits, unsigned Idx, T val) {
    if constexpr (std::numeric_limits<T>::is_signed) {
      if (val < 0) {
        val = ~val;
        Bits.set((Idx * BitNum) + BitNum - 1);
      } else {
        Bits.reset((Idx * BitNum) + BitNum - 1);
      }
      assert((val >> (BitNum - 1)) == 0 && "value is too big");
      for (unsigned i = 0; i != BitNum - 1; ++i)
        Bits[(Idx * BitNum) + i] = val & (T(1) << i);
    } else {
      assert((val >> BitNum) == 0 && "value is too big");
      for (unsigned i = 0; i != BitNum; ++i)
        Bits[(Idx * BitNum) + i] = val & (T(1) << i);
    }
  }

public:
  class reference {
    PackedVector &Vec;
    const unsigned Idx;

  public:
    reference() = delete;
    reference(PackedVector &vec, unsigned idx) : Vec(vec), Idx(idx) {}

    reference &operator=(T val) {
      Vec.setValue(Vec.Bits, Idx, val);
      return *this;
    }

    operator T() const { return Vec.getValue(Vec.Bits, Idx); }
  };

  PackedVector() = default;
  explicit PackedVector(unsigned size)
      : Bits(size * BitNum), NumElements(size) {}

  bool empty() const { return NumElements == 0; }

  unsigned size() const { return NumElements; }

  void clear() {
    Bits.clear();
    NumElements = 0;
  }

  void resize(unsigned N) {
    Bits.resize(N * BitNum);
    NumElements = N;
  }

  void reserve(unsigned N) { Bits.reserve(N * BitNum); }

  PackedVector &reset() {
    Bits.reset();
    return *this;
  }

  void push_back(T val) {
    resize(size() + 1);
    (*this)[size() - 1] = val;
  }

  reference operator[](unsigned Idx) { return reference(*this, Idx); }

  T operator[](unsigned Idx) const { return getValue(Bits, Idx); }

  bool operator==(const PackedVector &RHS) const { return Bits == RHS.Bits; }

  bool operator!=(const PackedVector &RHS) const { return Bits != RHS.Bits; }

  PackedVector &operator|=(const PackedVector &RHS) {
    Bits |= RHS.Bits;
    return *this;
  }

  const BitVectorTy &raw_bits() const { return Bits; }
  BitVectorTy &raw_bits() { return Bits; }
};

} // end namespace llvm

#endif // LLVM_ADT_PACKEDVECTOR_H
