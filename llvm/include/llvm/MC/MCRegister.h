//===-- llvm/MC/Register.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCREGISTER_H
#define LLVM_MC_MCREGISTER_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include <cassert>
#include <limits>

namespace llvm {

/// An unsigned integer type large enough to represent all physical registers,
/// but not necessarily virtual registers.
using MCPhysReg = uint16_t;

/// Register units are used to compute register aliasing. Every register has at
/// least one register unit, but it can have more. Two registers overlap if and
/// only if they have a common register unit.
///
/// A target with a complicated sub-register structure will typically have many
/// fewer register units than actual registers. MCRI::getNumRegUnits() returns
/// the number of register units in the target.
enum class MCRegUnit : unsigned;

struct MCRegUnitToIndex {
  using argument_type = MCRegUnit;

  unsigned operator()(MCRegUnit Unit) const {
    return static_cast<unsigned>(Unit);
  }
};

/// Wrapper class representing physical registers. Should be passed by value.
class MCRegister {
  friend hash_code hash_value(const MCRegister &);
  unsigned Reg;

public:
  constexpr MCRegister(unsigned Val = 0) : Reg(Val) {}

  // Register numbers can represent physical registers, virtual registers, and
  // sometimes stack slots. The unsigned values are divided into these ranges:
  //
  //   0           Not a register, can be used as a sentinel.
  //   [1;2^30)    Physical registers assigned by TableGen.
  //   [2^30;2^31) Stack slots. (Rarely used.)
  //   [2^31;2^32) Virtual registers assigned by MachineRegisterInfo.
  //
  // Further sentinels can be allocated from the small negative integers.
  // DenseMapInfo<unsigned> uses -1u and -2u.
  static_assert(std::numeric_limits<decltype(Reg)>::max() >= 0xFFFFFFFF,
                "Reg isn't large enough to hold full range.");
  static constexpr unsigned NoRegister = 0u;
  static constexpr unsigned FirstPhysicalReg = 1u;
  static constexpr unsigned LastPhysicalReg = (1u << 30) - 1;

  /// Return true if the specified register number is in
  /// the physical register namespace.
  static constexpr bool isPhysicalRegister(unsigned Reg) {
    return FirstPhysicalReg <= Reg && Reg <= LastPhysicalReg;
  }

  /// Return true if the specified register number is in the physical register
  /// namespace.
  constexpr bool isPhysical() const { return isPhysicalRegister(Reg); }

  constexpr operator unsigned() const { return Reg; }

  /// Check the provided unsigned value is a valid MCRegister.
  static MCRegister from(unsigned Val) {
    assert(Val == NoRegister || isPhysicalRegister(Val));
    return MCRegister(Val);
  }

  constexpr unsigned id() const { return Reg; }

  constexpr bool isValid() const { return Reg != NoRegister; }

  /// Comparisons between register objects
  constexpr bool operator==(const MCRegister &Other) const {
    return Reg == Other.Reg;
  }
  constexpr bool operator!=(const MCRegister &Other) const {
    return Reg != Other.Reg;
  }

  /// Comparisons against register constants. E.g.
  /// * R == AArch64::WZR
  /// * R == 0
  constexpr bool operator==(unsigned Other) const { return Reg == Other; }
  constexpr bool operator!=(unsigned Other) const { return Reg != Other; }
  constexpr bool operator==(int Other) const { return Reg == unsigned(Other); }
  constexpr bool operator!=(int Other) const { return Reg != unsigned(Other); }
  // MSVC requires that we explicitly declare these two as well.
  constexpr bool operator==(MCPhysReg Other) const {
    return Reg == unsigned(Other);
  }
  constexpr bool operator!=(MCPhysReg Other) const {
    return Reg != unsigned(Other);
  }
};

/// A block of registers spread evenly over one register sequence, each
/// spanning the same number of its members.
///
/// The registers of a block are enumerated in the order of the members they
/// begin at, so naming the first of them and saying how many follow describes
/// them all, and none of them needs an enumerator of its own.
struct MCRegisterSequenceBlock {
  /// The first register of the block.
  MCRegister FirstReg;

  /// The number of registers in the block.
  unsigned Count;

  /// The member-index distance between the members that adjacent registers
  /// begin at. Not every member begins a register: where the step is four,
  /// only every fourth one does.
  unsigned Step;

  /// Returns the register that begins at the given member of the sequence.
  /// The block name and the member read together as the name of that register:
  /// SGPR_64(30) is SGPR30_64.
  constexpr MCRegister operator()(unsigned Member) const {
    assert(Member % Step == 0 && "No register begins at this member.");
    unsigned Index = Member / Step;
    assert(Index < Count && "Register index out of range.");
    return FirstReg.id() + Index;
  }
};

// Provide DenseMapInfo for MCRegister
template <> struct DenseMapInfo<MCRegister> {
  static unsigned getHashValue(const MCRegister &Val) {
    return DenseMapInfo<unsigned>::getHashValue(Val.id());
  }
  static bool isEqual(const MCRegister &LHS, const MCRegister &RHS) {
    return LHS == RHS;
  }
};

inline hash_code hash_value(const MCRegister &Reg) {
  return hash_value(Reg.id());
}
} // namespace llvm

#endif // LLVM_MC_MCREGISTER_H
