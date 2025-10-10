//=== llvm/ADT/Bitset.h - constexpr std::bitset -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a std::bitset like container that can be used in constexprs.
// That constructor and many of the methods are constexpr. std::bitset doesn't
// get constexpr methods until C++23. This class also provides a constexpr
// constructor that accepts an initializer_list of bits to set.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_BITSET_H
#define LLVM_ADT_BITSET_H

#include <llvm/ADT/STLExtras.h>
#include <array>
#include <climits>
#include <cstdint>

namespace llvm {

/// This is a constexpr reimplementation of a subset of std::bitset. It would be
/// nice to use std::bitset directly, but it doesn't support constant
/// initialization.
template <unsigned NumBits>
class Bitset {
  using BitWord = uintptr_t;

  static constexpr unsigned BitwordBits = sizeof(BitWord) * CHAR_BIT;

  static_assert(BitwordBits == 64 || BitwordBits == 32,
                "Unsupported word size");

  static constexpr unsigned NumWords =
      (NumBits + BitwordBits - 1) / BitwordBits;

  using StorageType = std::array<BitWord, NumWords>;
  StorageType Bits{};

protected:
  constexpr Bitset(const std::array<uint64_t, (NumBits + 63) / 64> &B) {
    if constexpr (sizeof(BitWord) == sizeof(uint64_t)) {
      for (size_t I = 0; I != B.size(); ++I)
        Bits[I] = B[I];
    } else {
      for (size_t I = 0; I != B.size(); ++I) {
        uint64_t Elt = B[I];
        Bits[2 * I] = static_cast<uint32_t>(Elt);
        Bits[2 * I + 1] = static_cast<uint32_t>(Elt >> 32);
      }
    }
  }

public:
  constexpr Bitset() = default;
  constexpr Bitset(std::initializer_list<unsigned> Init) {
    for (auto I : Init)
      set(I);
  }

  Bitset &set() {
    llvm::fill(Bits, -BitWord(0));
    return *this;
  }

  constexpr Bitset &set(unsigned I) {
    Bits[I / BitwordBits] |= BitWord(1) << (I % BitwordBits);
    return *this;
  }

  constexpr Bitset &reset(unsigned I) {
    Bits[I / BitwordBits] &= ~(BitWord(1) << (I % BitwordBits));
    return *this;
  }

  constexpr Bitset &flip(unsigned I) {
    Bits[I / BitwordBits] ^= BitWord(1) << (I % BitwordBits);
    return *this;
  }

  constexpr bool operator[](unsigned I) const {
    BitWord Mask = BitWord(1) << (I % BitwordBits);
    return (Bits[I / BitwordBits] & Mask) != 0;
  }

  constexpr bool test(unsigned I) const { return (*this)[I]; }

  constexpr size_t size() const { return NumBits; }

  bool any() const {
    return llvm::any_of(Bits, [](BitWord I) { return I != 0; });
  }
  bool none() const { return !any(); }
  size_t count() const {
    size_t Count = 0;
    for (auto B : Bits)
      Count += llvm::popcount(B);
    return Count;
  }

  constexpr Bitset &operator^=(const Bitset &RHS) {
    for (unsigned I = 0, E = Bits.size(); I != E; ++I) {
      Bits[I] ^= RHS.Bits[I];
    }
    return *this;
  }
  constexpr Bitset operator^(const Bitset &RHS) const {
    Bitset Result = *this;
    Result ^= RHS;
    return Result;
  }

  constexpr Bitset &operator&=(const Bitset &RHS) {
    for (unsigned I = 0, E = Bits.size(); I != E; ++I)
      Bits[I] &= RHS.Bits[I];
    return *this;
  }
  constexpr Bitset operator&(const Bitset &RHS) const {
    Bitset Result = *this;
    Result &= RHS;
    return Result;
  }

  constexpr Bitset &operator|=(const Bitset &RHS) {
    for (unsigned I = 0, E = Bits.size(); I != E; ++I) {
      Bits[I] |= RHS.Bits[I];
    }
    return *this;
  }
  constexpr Bitset operator|(const Bitset &RHS) const {
    Bitset Result = *this;
    Result |= RHS;
    return Result;
  }

  constexpr Bitset operator~() const {
    Bitset Result = *this;
    for (auto &B : Result.Bits)
      B = ~B;
    return Result;
  }

  bool operator==(const Bitset &RHS) const {
    return std::equal(std::begin(Bits), std::end(Bits), std::begin(RHS.Bits));
  }

  bool operator!=(const Bitset &RHS) const { return !(*this == RHS); }

  bool operator < (const Bitset &Other) const {
    for (unsigned I = 0, E = size(); I != E; ++I) {
      bool LHS = test(I), RHS = Other.test(I);
      if (LHS != RHS)
        return LHS < RHS;
    }
    return false;
  }
};

} // end namespace llvm

#endif
