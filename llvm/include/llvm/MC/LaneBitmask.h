//===- llvm/MC/LaneBitmask.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A common definition of LaneBitmask for use in TableGen and CodeGen.
///
/// A lane mask is a bitmask representing the covering of a register with
/// sub-registers.
///
/// This is typically used to track liveness at sub-register granularity.
/// Lane masks for sub-register indices are similar to register units for
/// physical registers. The individual bits in a lane mask can't be assigned
/// any specific meaning. They can be used to check if two sub-register
/// indices overlap.
///
/// Iff the target has a register such that:
///
///   getSubReg(Reg, A) overlaps getSubReg(Reg, B)
///
/// then:
///
///   (getSubRegIndexLaneMask(A) & getSubRegIndexLaneMask(B)) != 0

#ifndef LLVM_MC_LANEBITMASK_H
#define LLVM_MC_LANEBITMASK_H

#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cassert>

namespace llvm::detail {
template <unsigned NumBits> struct LaneBitmaskImpl {
  static constexpr unsigned BitWidth = NumBits;
  /// Number of 64-bit words needed to hold all lanes.
  static constexpr unsigned NumWords64 = Bitset<NumBits>::getNumWords64();

  constexpr LaneBitmaskImpl() = default;
  constexpr LaneBitmaskImpl(const LaneBitmaskImpl &) = default;
  explicit constexpr LaneBitmaskImpl(uint64_t V)
      : Storage(std::array<uint64_t, NumWords64>{V}) {}
  explicit constexpr LaneBitmaskImpl(const std::array<uint64_t, NumWords64> &B)
      : Storage(B) {}
  // Delete the initializer_list constructor to avoid ambiguity with the
  // std::array constructor.
  LaneBitmaskImpl(std::initializer_list<unsigned>) = delete;
  constexpr LaneBitmaskImpl &operator=(const LaneBitmaskImpl &) = default;

  constexpr bool operator==(const LaneBitmaskImpl &Other) const {
    return Storage == Other.Storage;
  }
  constexpr bool operator!=(const LaneBitmaskImpl &Other) const {
    return Storage != Other.Storage;
  }
  /// Compare as unsigned integers (most-significant word first). This differs
  /// from Bitset::operator< which compares bit-by-bit from LSB.
  constexpr bool operator<(const LaneBitmaskImpl &Other) const {
    for (int I = NumWords64 - 1; I >= 0; --I) {
      if (Storage.getWord64(I) != Other.Storage.getWord64(I))
        return Storage.getWord64(I) < Other.Storage.getWord64(I);
    }
    return false;
  }

  constexpr bool none() const { return Storage.none(); }
  constexpr bool any() const { return Storage.any(); }
  constexpr bool all() const { return Storage.all(); }

  constexpr LaneBitmaskImpl operator~() const {
    LaneBitmaskImpl Result;
    Result.Storage = ~Storage;
    return Result;
  }
  constexpr LaneBitmaskImpl operator|(const LaneBitmaskImpl &M) const {
    LaneBitmaskImpl Result;
    Result.Storage = Storage | M.Storage;
    return Result;
  }
  constexpr LaneBitmaskImpl operator&(const LaneBitmaskImpl &M) const {
    LaneBitmaskImpl Result;
    Result.Storage = Storage & M.Storage;
    return Result;
  }
  constexpr LaneBitmaskImpl &operator|=(const LaneBitmaskImpl &M) {
    Storage |= M.Storage;
    return *this;
  }
  constexpr LaneBitmaskImpl &operator&=(const LaneBitmaskImpl &M) {
    Storage &= M.Storage;
    return *this;
  }

  /// Return the I-th 64-bit word of the bitmask from least significant to most
  /// significant.
  constexpr uint64_t getWord64(unsigned I) const {
    return Storage.getWord64(I);
  }

  constexpr size_t getNumLanes() const { return Storage.count(); }

  unsigned getHighestLane() const {
    int Result = Storage.findLastSet();
    assert(Result >= 0 && "getHighestLane called on empty mask");
    return static_cast<unsigned>(Result);
  }

  constexpr LaneBitmaskImpl operator<<(unsigned S) const {
    LaneBitmaskImpl Result;
    Result.Storage = Storage << S;
    return Result;
  }
  constexpr LaneBitmaskImpl operator>>(unsigned S) const {
    LaneBitmaskImpl Result;
    Result.Storage = Storage >> S;
    return Result;
  }

  /// Rotate bits left by \p S positions.
  constexpr LaneBitmaskImpl rotateLeft(unsigned S) const {
    S = S % NumBits;
    if (S == 0)
      return *this;
    return (*this << S) | (*this >> (NumBits - S));
  }

  /// Rotate bits right by \p S positions.
  constexpr LaneBitmaskImpl rotateRight(unsigned S) const {
    S = S % NumBits;
    if (S == 0)
      return *this;
    return (*this >> S) | (*this << (NumBits - S));
  }

  static constexpr LaneBitmaskImpl getNone() { return LaneBitmaskImpl(); }

  static constexpr LaneBitmaskImpl getAll() {
    LaneBitmaskImpl Result;
    Result.Storage.set();
    return Result;
  }

  static constexpr LaneBitmaskImpl getLane(unsigned Lane) {
    LaneBitmaskImpl Result;
    Result.Storage.set(Lane);
    return Result;
  }

private:
  Bitset<NumBits> Storage;
};

} // end namespace llvm::detail

namespace llvm {
using LaneBitmask = detail::LaneBitmaskImpl<64>;

/// Create Printable object to print LaneBitmasks on a \ref raw_ostream.
template <unsigned NumBits>
inline Printable PrintLaneMask(detail::LaneBitmaskImpl<NumBits> LaneMask) {
  return Printable([LaneMask](raw_ostream &OS) {
    using T = detail::LaneBitmaskImpl<NumBits>;
    // Print as hex using 64-bit words from most significant to least.
    // Only print the first 64 bits if all upper words are zero.
    constexpr unsigned HexWidth = 64 / 4; // One hex digit per 4 bits.
    bool UpperWordsZero = (~T(~0ULL) & LaneMask).none();
    for (int I = UpperWordsZero ? 0 : T::NumWords64 - 1; I >= 0; --I)
      OS << format_hex_no_prefix(LaneMask.getWord64(I), HexWidth,
                                 /*Upper=*/true);
  });
}

template <unsigned NumBits>
inline hash_code hash_value(const detail::LaneBitmaskImpl<NumBits> &LM) {
  constexpr unsigned NumWords = detail::LaneBitmaskImpl<NumBits>::NumWords64;
  if constexpr (NumWords == 1)
    return hash_value(LM.getWord64(0));
  else if constexpr (NumWords == 2)
    return hash_combine(LM.getWord64(0), LM.getWord64(1));
  else {
    hash_code H = hash_value(LM.getWord64(0));
    for (unsigned I = 1; I < NumWords; ++I)
      H = hash_combine(H, LM.getWord64(I));
    return H;
  }
}

} // end namespace llvm

namespace std {

template <unsigned NumBits>
struct hash<llvm::detail::LaneBitmaskImpl<NumBits>> {
  size_t operator()(const llvm::detail::LaneBitmaskImpl<NumBits> &LM) const {
    return llvm::hash_value(LM);
  }
};

} // end namespace std

#endif // LLVM_MC_LANEBITMASK_H
