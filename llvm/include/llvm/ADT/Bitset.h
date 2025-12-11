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

#include "llvm/ADT/bit.h"
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
  static constexpr unsigned RemainderNumBits = NumBits % BitwordBits;
  static constexpr BitWord RemainderMask =
      RemainderNumBits == 0 ? ~BitWord(0)
                            : ((BitWord(1) << RemainderNumBits) - 1);

  static_assert(BitwordBits == 64 || BitwordBits == 32,
                "Unsupported word size");

  static constexpr unsigned NumWords =
      (NumBits + BitwordBits - 1) / BitwordBits;

  using StorageType = std::array<BitWord, NumWords>;
  StorageType Bits{};

  constexpr void maskLastWord() {
    if constexpr (RemainderNumBits != 0)
      Bits[NumWords - 1] &= RemainderMask;
  }

protected:
  constexpr Bitset(const std::array<uint64_t, (NumBits + 63) / 64> &B) {
    if constexpr (sizeof(BitWord) == sizeof(uint64_t)) {
      for (size_t I = 0; I != B.size(); ++I)
        Bits[I] = B[I];
    } else {
      unsigned BitsToAssign = NumBits;
      for (size_t I = 0; I != B.size() && BitsToAssign; ++I) {
        uint64_t Elt = B[I];
        // On a 32-bit system the storage type will be 32-bit, so we may only
        // need half of a uint64_t.
        for (size_t Offset = 0; Offset != 2 && BitsToAssign; ++Offset) {
          Bits[2 * I + Offset] = static_cast<uint32_t>(Elt >> (32 * Offset));
          BitsToAssign = BitsToAssign >= 32 ? BitsToAssign - 32 : 0;
        }
      }
    }
    maskLastWord();
  }

public:
  constexpr Bitset() = default;
  constexpr Bitset(std::initializer_list<unsigned> Init) {
    for (auto I : Init)
      set(I);
  }

  constexpr Bitset &set() {
    constexpr const BitWord AllOnes = ~BitWord(0);
    for (BitWord &B : Bits)
      B = AllOnes;
    maskLastWord();
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

  constexpr bool any() const {
    for (unsigned I = 0; I < NumWords - 1; ++I)
      if (Bits[I] != 0)
        return true;
    return (Bits[NumWords - 1] & RemainderMask) != 0;
  }

  constexpr bool none() const { return !any(); }

  constexpr bool all() const {
    constexpr const BitWord AllOnes = ~BitWord(0);
    for (unsigned I = 0; I < NumWords - 1; ++I)
      if (Bits[I] != AllOnes)
        return false;
    return (Bits[NumWords - 1] & RemainderMask) == RemainderMask;
  }

  constexpr size_t count() const {
    size_t Count = 0;
    for (unsigned I = 0; I < NumWords - 1; ++I)
      Count += popcount(Bits[I]);
    Count += popcount(Bits[NumWords - 1] & RemainderMask);
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
    Result.maskLastWord();
    return Result;
  }

  constexpr bool operator==(const Bitset &RHS) const {
    for (unsigned I = 0; I < NumWords - 1; ++I)
      if (Bits[I] != RHS.Bits[I])
        return false;
    return (Bits[NumWords - 1] & RemainderMask) ==
           (RHS.Bits[NumWords - 1] & RemainderMask);
  }

  constexpr bool operator!=(const Bitset &RHS) const { return !(*this == RHS); }

  constexpr bool operator<(const Bitset &Other) const {
    for (unsigned I = 0, E = size(); I != E; ++I) {
      bool LHS = test(I), RHS = Other.test(I);
      if (LHS != RHS)
        return LHS < RHS;
    }
    return false;
  }

  constexpr Bitset &operator<<=(unsigned N) {
    if (N == 0)
      return *this;
    if (N >= NumBits) {
      return *this = Bitset();
    }
    const unsigned WordShift = N / BitwordBits;
    const unsigned BitShift = N % BitwordBits;
    if (BitShift == 0) {
      for (int I = NumWords - 1; I >= static_cast<int>(WordShift); --I)
        Bits[I] = Bits[I - WordShift];
    } else {
      const unsigned CarryShift = BitwordBits - BitShift;
      for (int I = NumWords - 1; I > static_cast<int>(WordShift); --I) {
        Bits[I] = (Bits[I - WordShift] << BitShift) |
                  (Bits[I - WordShift - 1] >> CarryShift);
      }
      Bits[WordShift] = Bits[0] << BitShift;
    }
    for (unsigned I = 0; I < WordShift; ++I)
      Bits[I] = 0;
    maskLastWord();
    return *this;
  }

  constexpr Bitset operator<<(unsigned N) const {
    Bitset Result(*this);
    Result <<= N;
    return Result;
  }

  constexpr Bitset &operator>>=(unsigned N) {
    if (N == 0)
      return *this;
    if (N >= NumBits) {
      return *this = Bitset();
    }
    const unsigned WordShift = N / BitwordBits;
    const unsigned BitShift = N % BitwordBits;
    if (BitShift == 0) {
      for (unsigned I = 0; I < NumWords - WordShift; ++I)
        Bits[I] = Bits[I + WordShift];
    } else {
      const unsigned CarryShift = BitwordBits - BitShift;
      for (unsigned I = 0; I < NumWords - WordShift - 1; ++I) {
        Bits[I] = (Bits[I + WordShift] >> BitShift) |
                  (Bits[I + WordShift + 1] << CarryShift);
      }
      Bits[NumWords - WordShift - 1] = Bits[NumWords - 1] >> BitShift;
    }
    for (unsigned I = NumWords - WordShift; I < NumWords; ++I)
      Bits[I] = 0;
    maskLastWord();
    return *this;
  }

  constexpr Bitset operator>>(unsigned N) const {
    Bitset Result(*this);
    Result >>= N;
    return Result;
  }
};

} // end namespace llvm

#endif
