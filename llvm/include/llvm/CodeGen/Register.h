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
#include <cassert>

namespace llvm {

/// Wrapper class representing virtual and physical registers. Should be passed
/// by value.
class Register {
  unsigned Reg;

public:
  Register(unsigned Val = 0): Reg(Val) {}

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

  /// isStackSlot - Sometimes it is useful the be able to store a non-negative
  /// frame index in a variable that normally holds a register. isStackSlot()
  /// returns true if Reg is in the range used for stack slots.
  ///
  /// Note that isVirtualRegister() and isPhysicalRegister() cannot handle stack
  /// slots, so if a variable may contains a stack slot, always check
  /// isStackSlot() first.
  ///
  static bool isStackSlot(unsigned Reg) {
    return MCRegister::isStackSlot(Reg);
  }

  /// Compute the frame index from a register value representing a stack slot.
  static int stackSlot2Index(unsigned Reg) {
    assert(isStackSlot(Reg) && "Not a stack slot");
    return int(Reg - (1u << 30));
  }

  /// Convert a non-negative frame index to a stack slot register value.
  static unsigned index2StackSlot(int FI) {
    assert(FI >= 0 && "Cannot hold a negative frame index.");
    return FI + (1u << 30);
  }

  /// Return true if the specified register number is in
  /// the physical register namespace.
  static bool isPhysicalRegister(unsigned Reg) {
    return MCRegister::isPhysicalRegister(Reg);
  }

  /// Return true if the specified register number is in
  /// the virtual register namespace.
  static bool isVirtualRegister(unsigned Reg) {
    assert(!isStackSlot(Reg) && "Not a register! Check isStackSlot() first.");
    return int(Reg) < 0;
  }

  /// Convert a virtual register number to a 0-based index.
  /// The first virtual register in a function will get the index 0.
  static unsigned virtReg2Index(unsigned Reg) {
    assert(isVirtualRegister(Reg) && "Not a virtual register");
    return Reg & ~(1u << 31);
  }

  /// Convert a 0-based index to a virtual register number.
  /// This is the inverse operation of VirtReg2IndexFunctor below.
  static unsigned index2VirtReg(unsigned Index) {
    return Index | (1u << 31);
  }

  /// Return true if the specified register number is in the virtual register
  /// namespace.
  bool isVirtual() const {
    return isVirtualRegister(Reg);
  }

  /// Return true if the specified register number is in the physical register
  /// namespace.
  bool isPhysical() const {
    return isPhysicalRegister(Reg);
  }

  /// Convert a virtual register number to a 0-based index. The first virtual
  /// register in a function will get the index 0.
  unsigned virtRegIndex() const {
    return virtReg2Index(Reg);
  }

  operator unsigned() const {
    return Reg;
  }

  bool isValid() const {
    return Reg != 0;
  }
};

}

#endif // ifndef LLVM_CODEGEN_REGISTER_H
