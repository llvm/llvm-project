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
#include <cassert>
#include <limits>

namespace llvm {

/// An unsigned integer type large enough to represent all physical registers,
/// but not necessarily virtual registers.
using MCPhysReg = uint16_t;

/// Wrapper class representing physical registers. Should be passed by value.
class MCRegister {
  friend hash_code hash_value(const MCRegister &);
  unsigned Reg;

public:
  constexpr MCRegister(unsigned Val = 0): Reg(Val) {}

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
  static constexpr unsigned FirstStackSlot = 1u << 30;
  static constexpr unsigned VirtualRegFlag = 1u << 31;

  /// This is the portion of the positive number space that is not a physical
  /// register. StackSlot values do not exist in the MC layer, see
  /// Register::isStackSlot() for the more information on them.
  ///
  static bool isStackSlot(unsigned Reg) {
    return FirstStackSlot <= Reg && Reg < VirtualRegFlag;
  }

  /// Return true if the specified register number is in
  /// the physical register namespace.
  static bool isPhysicalRegister(unsigned Reg) {
    return FirstPhysicalReg <= Reg && Reg < FirstStackSlot;
  }

  constexpr operator unsigned() const {
    return Reg;
  }

  /// Check the provided unsigned value is a valid MCRegister.
  static MCRegister from(unsigned Val) {
    assert(Val == NoRegister || isPhysicalRegister(Val));
    return MCRegister(Val);
  }

  unsigned id() const {
    return Reg;
  }

  bool isValid() const { return Reg != NoRegister; }

  /// Comparisons between register objects
  bool operator==(const MCRegister &Other) const { return Reg == Other.Reg; }
  bool operator!=(const MCRegister &Other) const { return Reg != Other.Reg; }

  /// Comparisons against register constants. E.g.
  /// * R == AArch64::WZR
  /// * R == 0
  /// * R == VirtRegMap::NO_PHYS_REG
  bool operator==(unsigned Other) const { return Reg == Other; }
  bool operator!=(unsigned Other) const { return Reg != Other; }
  bool operator==(int Other) const { return Reg == unsigned(Other); }
  bool operator!=(int Other) const { return Reg != unsigned(Other); }
  // MSVC requires that we explicitly declare these two as well.
  bool operator==(MCPhysReg Other) const { return Reg == unsigned(Other); }
  bool operator!=(MCPhysReg Other) const { return Reg != unsigned(Other); }
};

// Provide DenseMapInfo for MCRegister
template<> struct DenseMapInfo<MCRegister> {
  static inline unsigned getEmptyKey() {
    return DenseMapInfo<unsigned>::getEmptyKey();
  }
  static inline unsigned getTombstoneKey() {
    return DenseMapInfo<unsigned>::getTombstoneKey();
  }
  static unsigned getHashValue(const MCRegister &Val) {
    return DenseMapInfo<unsigned>::getHashValue(Val.id());
  }
  static bool isEqual(const MCRegister &LHS, const MCRegister &RHS) {
    return DenseMapInfo<unsigned>::isEqual(LHS.id(), RHS.id());
  }
};

inline hash_code hash_value(const MCRegister &Reg) {
  return hash_value(Reg.id());
}
}

#endif // LLVM_MC_MCREGISTER_H
