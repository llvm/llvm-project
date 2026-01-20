//===-- llvm/CodeGen/Register.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTER_H
#define LLVM_CODEGEN_REGISTER_H

#include "llvm/MC/MCRegister.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace llvm {

/// Wrapper class representing virtual and physical registers. Should be passed
/// by value.
class Register {
  unsigned Reg;

public:
  constexpr Register(unsigned Val = 0) : Reg(Val) {}
  constexpr Register(MCRegister Val) : Reg(Val.id()) {}

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
  static constexpr unsigned MaxFrameIndexBitwidth = 30;
  static constexpr unsigned StackSlotZero = 1u << MaxFrameIndexBitwidth;
  static constexpr const unsigned StackSlotMask = StackSlotZero - 1;
  static_assert(StackSlotZero >= MCRegister::LastPhysicalReg);
  static constexpr unsigned VirtualRegFlag = 1u << 31;

  /// Return true if this is a stack slot.
  constexpr bool isStack() const {
    return Register::StackSlotZero <= Reg && Reg < Register::VirtualRegFlag;
  }

  /// Convert a non-negative frame index to a stack slot register value.
  static Register index2StackSlot(int FI) {
    assert(isInt<MaxFrameIndexBitwidth>(FI) &&
           "Frame index must be at most 30 bits.");
    unsigned FIMasked = FI & Register::StackSlotMask;
    return Register(FIMasked | Register::StackSlotZero);
  }

  /// Return true if the specified register number is in
  /// the physical register namespace.
  static constexpr bool isPhysicalRegister(unsigned Reg) {
    return MCRegister::isPhysicalRegister(Reg);
  }

  /// Return true if the specified register number is in
  /// the virtual register namespace.
  static constexpr bool isVirtualRegister(unsigned Reg) {
    return Reg & Register::VirtualRegFlag;
  }

  /// Convert a 0-based index to a virtual register number.
  /// This is the inverse operation of VirtReg2IndexFunctor below.
  static Register index2VirtReg(unsigned Index) {
    assert(Index < (1u << 31) && "Index too large for virtual register range.");
    return Index | Register::VirtualRegFlag;
  }

  /// Return true if the specified register number is in the virtual register
  /// namespace.
  constexpr bool isVirtual() const { return isVirtualRegister(Reg); }

  /// Return true if the specified register number is in the physical register
  /// namespace.
  constexpr bool isPhysical() const { return isPhysicalRegister(Reg); }

  /// Convert a virtual register number to a 0-based index. The first virtual
  /// register in a function will get the index 0.
  unsigned virtRegIndex() const {
    assert(isVirtual() && "Not a virtual register");
    return Reg & ~Register::VirtualRegFlag;
  }

  /// Compute the frame index from a register value representing a stack slot.
  int stackSlotIndex() const {
    assert(isStack() && "Not a stack slot");
    return SignExtend32<MaxFrameIndexBitwidth>(Reg & Register::StackSlotMask);
  }

  constexpr operator unsigned() const { return Reg; }

  constexpr unsigned id() const { return Reg; }

  constexpr operator MCRegister() const { return MCRegister(Reg); }

  /// Utility to check-convert this value to a MCRegister. The caller is
  /// expected to have already validated that this Register is, indeed,
  /// physical.
  MCRegister asMCReg() const {
    assert(!isValid() || isPhysical());
    return MCRegister(Reg);
  }

  constexpr bool isValid() const { return Reg != MCRegister::NoRegister; }

  /// Comparisons between register objects
  constexpr bool operator==(const Register &Other) const {
    return Reg == Other.Reg;
  }
  constexpr bool operator!=(const Register &Other) const {
    return Reg != Other.Reg;
  }
  constexpr bool operator==(const MCRegister &Other) const {
    return Reg == Other.id();
  }
  constexpr bool operator!=(const MCRegister &Other) const {
    return Reg != Other.id();
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

  /// Operators to move from one register to another nearby register by adding
  /// an offset.
  Register &operator++() {
    assert(isValid());
    ++Reg;
    return *this;
  }

  Register operator++(int) {
    Register R(*this);
    ++(*this);
    return R;
  }

  Register &operator+=(unsigned RHS) {
    assert(isValid());
    Reg += RHS;
    return *this;
  }
};

// Provide DenseMapInfo for Register
template <> struct DenseMapInfo<Register> {
  static inline Register getEmptyKey() {
    return DenseMapInfo<unsigned>::getEmptyKey();
  }
  static inline Register getTombstoneKey() {
    return DenseMapInfo<unsigned>::getTombstoneKey();
  }
  static unsigned getHashValue(const Register &Val) {
    return DenseMapInfo<unsigned>::getHashValue(Val.id());
  }
  static bool isEqual(const Register &LHS, const Register &RHS) {
    return LHS == RHS;
  }
};

/// Wrapper class representing a virtual register or register unit.
class VirtRegOrUnit {
  unsigned VRegOrUnit;

public:
  constexpr explicit VirtRegOrUnit(MCRegUnit Unit)
      : VRegOrUnit(static_cast<unsigned>(Unit)) {
    assert(!Register::isVirtualRegister(VRegOrUnit));
  }

  constexpr explicit VirtRegOrUnit(Register Reg) : VRegOrUnit(Reg.id()) {
    assert(Reg.isVirtual());
  }

  // Catches implicit conversions to Register.
  template <typename T> explicit VirtRegOrUnit(T) = delete;

  constexpr bool isVirtualReg() const {
    return Register::isVirtualRegister(VRegOrUnit);
  }

  constexpr MCRegUnit asMCRegUnit() const {
    assert(!isVirtualReg() && "Not a register unit");
    return static_cast<MCRegUnit>(VRegOrUnit);
  }

  constexpr Register asVirtualReg() const {
    assert(isVirtualReg() && "Not a virtual register");
    return Register(VRegOrUnit);
  }

  constexpr bool operator==(const VirtRegOrUnit &Other) const {
    return VRegOrUnit == Other.VRegOrUnit;
  }

  constexpr bool operator<(const VirtRegOrUnit &Other) const {
    return VRegOrUnit < Other.VRegOrUnit;
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_REGISTER_H
