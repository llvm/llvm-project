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
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cassert>

namespace llvm::detail {
template <unsigned NumBits> struct LaneBitmaskImpl : public Bitset<NumBits> {
  static constexpr unsigned BitWidth = NumBits;

  constexpr LaneBitmaskImpl() = default;
  constexpr LaneBitmaskImpl(const LaneBitmaskImpl &) = default;
  explicit constexpr LaneBitmaskImpl(uint64_t V)
      : Bitset<NumBits>(std::array<uint64_t, (NumBits + 63) / 64>{V}) {}
  explicit constexpr LaneBitmaskImpl(
      const std::array<uint64_t, (NumBits + 63) / 64> &B)
      : Bitset<NumBits>(B) {}
  explicit LaneBitmaskImpl(const APInt &N)
      : Bitset<NumBits>(convertAPIntToArray(N)) {}
  // Delete the initializer_list constructor to avoid ambiguity with the
  // std::array constructor.
  LaneBitmaskImpl(std::initializer_list<unsigned>) = delete;
  constexpr LaneBitmaskImpl &operator=(const LaneBitmaskImpl &) = default;

  /// Compare as unsigned integers (most-significant word first). This differs
  /// from Bitset::operator< which compares bit-by-bit from LSB.
  constexpr bool operator<(const LaneBitmaskImpl &Other) const {
    const auto &ThisBits = this->getData();
    const auto &OtherBits = Other.getData();
    for (int I = ThisBits.size() - 1; I >= 0; --I) {
      if (ThisBits[I] != OtherBits[I])
        return ThisBits[I] < OtherBits[I];
    }
    return false;
  }

  constexpr LaneBitmaskImpl operator~() const {
    return Bitset<NumBits>::operator~();
  }
  constexpr LaneBitmaskImpl operator|(LaneBitmaskImpl M) const {
    return Bitset<NumBits>::operator|(M);
  }
  constexpr LaneBitmaskImpl operator&(LaneBitmaskImpl M) const {
    return Bitset<NumBits>::operator&(M);
  }
  constexpr LaneBitmaskImpl &operator|=(LaneBitmaskImpl M) {
    Bitset<NumBits>::operator|=(M);
    return *this;
  }
  constexpr LaneBitmaskImpl &operator&=(LaneBitmaskImpl M) {
    Bitset<NumBits>::operator&=(M);
    return *this;
  }
  constexpr LaneBitmaskImpl operator^(LaneBitmaskImpl M) const {
    return Bitset<NumBits>::operator^(M);
  }
  constexpr LaneBitmaskImpl &operator^=(LaneBitmaskImpl M) {
    Bitset<NumBits>::operator^=(M);
    return *this;
  }

  constexpr size_t getNumLanes() const { return this->count(); }

  unsigned getHighestLane() const {
    assert(this->any() && "getHighestLane called on empty mask");
    const auto &Bits = this->getData();
    constexpr size_t WordBits = sizeof(decltype(Bits[0])) * 8;
    for (int I = Bits.size() - 1; I >= 0; --I)
      if (Bits[I] != 0)
        return I * WordBits + Log2_64(Bits[I]);
    llvm_unreachable("should have found a set bit");
  }

  /// Shift bits left by \p S positions. Zeroes are shifted in from the right.
  constexpr LaneBitmaskImpl operator<<(unsigned S) const {
    return Bitset<NumBits>::operator<<(S);
  }

  /// Shift bits right by \p S positions. Zeroes are shifted in from the left.
  constexpr LaneBitmaskImpl operator>>(unsigned S) const {
    return Bitset<NumBits>::operator>>(S);
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
    Result.set();
    return Result;
  }

  static constexpr LaneBitmaskImpl getLane(unsigned Lane) {
    LaneBitmaskImpl Result;
    Result.set(Lane);
    return Result;
  }

private:
  constexpr LaneBitmaskImpl(const Bitset<NumBits> &B) : Bitset<NumBits>(B) {}

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

  template <typename, typename> friend struct llvm::format_provider;
};

} // end namespace llvm::detail

namespace llvm {
using LaneBitmask = detail::LaneBitmaskImpl<64>;

template <unsigned NumBits>
struct format_provider<detail::LaneBitmaskImpl<NumBits>> {
  using T = detail::LaneBitmaskImpl<NumBits>;
  static void format(const T &V, raw_ostream &Stream, StringRef Style) {
    // Print as hex using platform words from most significant to least.
    // Only print the first 64 bits if all upper words are zero.
    const auto &Data = V.getData();
    constexpr unsigned SizeOfBitword = sizeof(Data[0]);
    constexpr unsigned HexWidth = SizeOfBitword * 2;
    constexpr unsigned NumWordsIn64Bits = 8 / SizeOfBitword;
    T UpperWords = ~T(~0ULL) & V;
    if (UpperWords.none())
      for (int I = NumWordsIn64Bits - 1; I >= 0; --I)
        Stream << format_hex_no_prefix(Data[I], HexWidth, true);
    else
      for (int I = Data.size() - 1; I >= 0; --I)
        Stream << format_hex_no_prefix(Data[I], HexWidth, true);
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
  return hash_value(static_cast<const Bitset<NumBits> &>(LM));
}

} // end namespace llvm

namespace std {

template <unsigned NumBits>
struct hash<llvm::detail::LaneBitmaskImpl<NumBits>> {
  size_t operator()(const llvm::detail::LaneBitmaskImpl<NumBits> &LM) const {
    return hash<llvm::Bitset<NumBits>>{}(LM);
  }
};

} // end namespace std

#endif // LLVM_MC_LANEBITMASK_H
