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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cassert>

namespace llvm::detail {
template <unsigned NumBits> struct LaneBitmaskImpl {
  static constexpr unsigned BitWidth = NumBits;

  constexpr LaneBitmaskImpl() = default;
  constexpr LaneBitmaskImpl(const LaneBitmaskImpl &) = default;
  explicit constexpr LaneBitmaskImpl(uint64_t V)
      : Storage(std::array<uint64_t, (NumBits + 63) / 64>{V}) {}
  explicit constexpr LaneBitmaskImpl(
      const std::array<uint64_t, (NumBits + 63) / 64> &B)
      : Storage(B) {}
  explicit LaneBitmaskImpl(const APInt &N) : Storage(convertAPIntToArray(N)) {}
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
    for (int I = Storage.getNumWords64() - 1; I >= 0; --I) {
      if (Storage.getWord(I) != Other.Storage.getWord(I))
        return Storage.getWord(I) < Other.Storage.getWord(I);
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
  constexpr uint64_t getWord(unsigned I) const { return Storage.getWord(I); }

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

  /// Helper to convert APInt to array format for Bitset constructor.
  static std::array<uint64_t, (NumBits + 63) / 64>
  convertAPIntToArray(const APInt &N) {
    static_assert(std::is_same_v<APInt::WordType, uint64_t>,
                  "APInt::WordType needs to be uint64_t for word-level copy.");
    assert(N.getBitWidth() <= NumBits &&
           "Cannot convert to LaneBitmask. The input APInt has "
           "more bits than LaneBitmask can hold.");
    std::array<uint64_t, (NumBits + 63) / 64> Result{};
    const uint64_t *RawData = N.getRawData();
    const size_t NumWords = N.getNumWords();
    for (size_t I = 0; I < NumWords && I < Result.size(); ++I)
      Result[I] = RawData[I];
    return Result;
  }
};

} // end namespace llvm::detail

namespace llvm {
using LaneBitmask = detail::LaneBitmaskImpl<64>;

template <unsigned NumBits>
struct format_provider<detail::LaneBitmaskImpl<NumBits>> {
  using T = detail::LaneBitmaskImpl<NumBits>;
  static void format(const T &V, raw_ostream &Stream, StringRef Style) {
    // Print as hex using 64-bit words from most significant to least.
    // Only print the first 64 bits if all upper words are zero.
    constexpr unsigned HexWidth = 16; // 16 hex digits per 64-bit word.
    constexpr unsigned NumWords = Bitset<NumBits>::getNumWords64();
    T UpperWords = ~T(~0ULL) & V;
    if (UpperWords.none())
      Stream << format_hex_no_prefix(V.getWord(0), HexWidth, true);
    else
      for (int I = NumWords - 1; I >= 0; --I)
        Stream << format_hex_no_prefix(V.getWord(I), HexWidth, true);
  }
};

/// Create Printable object to print LaneBitmasks on a \ref raw_ostream.
template <unsigned NumBits>
inline Printable PrintLaneMask(detail::LaneBitmaskImpl<NumBits> LaneMask) {
  return Printable(
      [LaneMask](raw_ostream &OS) { OS << formatv("{0}", LaneMask); });
}

template <unsigned NumBits>
inline hash_code hash_value(const detail::LaneBitmaskImpl<NumBits> &LM) {
  constexpr unsigned NumWords = Bitset<NumBits>::getNumWords64();
  if constexpr (NumWords == 1)
    return hash_value(LM.getWord(0));
  else if constexpr (NumWords == 2)
    return hash_combine(LM.getWord(0), LM.getWord(1));
  else {
    hash_code H = hash_value(LM.getWord(0));
    for (unsigned I = 1; I < NumWords; ++I)
      H = hash_combine(H, LM.getWord(I));
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
