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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace llvm {

struct LaneBitmask {
  static constexpr unsigned int BitWidth = 128;

  explicit LaneBitmask(APInt V) {
    switch (V.getBitWidth()) {
    case BitWidth:
      Mask[0] = V.getRawData()[0];
      Mask[1] = V.getRawData()[1];
      break;
    default:
      llvm_unreachable("Unsupported bitwidth");
    }
  }
  constexpr explicit LaneBitmask(uint64_t Lo = 0, uint64_t Hi = 0) : Mask{Lo, Hi} {}

  constexpr bool operator==(LaneBitmask M) const {
    return Mask[0] == M.Mask[0] && Mask[1] == M.Mask[1];
  }
  constexpr bool operator!=(LaneBitmask M) const {
    return Mask[0] != M.Mask[0] || Mask[1] != M.Mask[1];
  }
  constexpr bool operator<(LaneBitmask M) const {
    return Mask[1] < M.Mask[1] || Mask[0] < M.Mask[0];
  }
  constexpr bool none() const { return Mask[0] == 0 && Mask[1] == 0; }
  constexpr bool any() const { return Mask[0] != 0 || Mask[1] != 0; }
  constexpr bool all() const { return ~Mask[0] == 0 && ~Mask[1] == 0; }

  constexpr LaneBitmask operator~() const { return LaneBitmask(~Mask[0], ~Mask[1]); }
  constexpr LaneBitmask operator|(LaneBitmask M) const {
    return LaneBitmask(Mask[0] | M.Mask[0], Mask[1] | M.Mask[1]);
  }
  constexpr LaneBitmask operator&(LaneBitmask M) const {
    return LaneBitmask(Mask[0] & M.Mask[0], Mask[1] & M.Mask[1]);
  }
  LaneBitmask &operator|=(LaneBitmask M) {
    Mask[0] |= M.Mask[0];
    Mask[1] |= M.Mask[1];
    return *this;
  }
  LaneBitmask &operator&=(LaneBitmask M) {
    Mask[0] &= M.Mask[0];
    Mask[1] &= M.Mask[1];
    return *this;
  }

  APInt getAsAPInt() const { return APInt(BitWidth, {Mask[0], Mask[1]}); }
  constexpr std::pair<uint64_t, uint64_t> getAsPair() const { return {Mask[0], Mask[1]}; }

  unsigned getNumLanes() const {
    return Mask[1] ? llvm::popcount(Mask[1]) + llvm::popcount(Mask[0])
                   : llvm::popcount(Mask[0]);
  }
  unsigned getHighestLane() const {
    return Mask[1] ? Log2_64(Mask[1]) + 64 : Log2_64(Mask[0]);
  }

  static constexpr LaneBitmask getNone() { return LaneBitmask(0, 0); }
  static constexpr LaneBitmask getAll() { return ~LaneBitmask(0, 0); }
  static constexpr LaneBitmask getLane(unsigned Lane) {
    return Lane >= 64 ? LaneBitmask(0, 1ULL << (Lane % 64))
                      : LaneBitmask(1ULL << Lane, 0);
  }

private:
  uint64_t Mask[2];
};

/// Create Printable object to print LaneBitmasks on a \ref raw_ostream.
/// If \p FormatAsCLiterals is true, it will print the bitmask as
/// a hexadecimal C literal with zero padding, or a list of such C literals if
/// the value cannot be represented in 64 bits.
/// For example (FormatAsCliterals == true)
///   bitmask '1'       => "0x0000000000000001"
///   bitmask '1 << 64' => "0x0000000000000000,0x0000000000000001"
/// (FormatAsCLiterals == false)
///   bitmask '1'       => "00000000000000000000000000000001"
///   bitmask '1 << 64' => "00000000000000010000000000000000"
inline Printable PrintLaneMask(LaneBitmask LaneMask,
                               bool FormatAsCLiterals = false) {
  return Printable([LaneMask, FormatAsCLiterals](raw_ostream &OS) {
    SmallString<64> Buffer;
    APInt V = LaneMask.getAsAPInt();
    while (true) {
      unsigned Bitwidth = FormatAsCLiterals ? 64 : LaneBitmask::BitWidth;
      APInt VToPrint = V.trunc(Bitwidth);

      Buffer.clear();
      VToPrint.toString(Buffer, 16, /*Signed=*/false,
                        /*formatAsCLiteral=*/false);
      unsigned NumZeroesToPad =
          (VToPrint.countLeadingZeros() / 4) - VToPrint.isZero();
      OS << (FormatAsCLiterals ? "0x" : "") << std::string(NumZeroesToPad, '0')
         << Buffer.str();
      V = V.lshr(Bitwidth);
      if (V.getActiveBits())
        OS << ",";
      else
        break;
    }
  });
}

} // end namespace llvm

#endif // LLVM_MC_LANEBITMASK_H
