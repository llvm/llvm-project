//===- llvm/CodeGen/LiveRegUnits.h - Register Unit Set ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// A set of register units. It is intended for register liveness tracking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEREGUNITS_H
#define LLVM_CODEGEN_LIVEREGUNITS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cstdint>

namespace llvm {

class MachineInstr;
class MachineBasicBlock;

/// A set of register units used to track register liveness.
class LiveRegUnits {
  const TargetRegisterInfo *TRI = nullptr;
  BitVector Units;

public:
  /// Constructs a new empty LiveRegUnits set.
  LiveRegUnits() = default;

  /// Constructs and initialize an empty LiveRegUnits set.
  LiveRegUnits(const TargetRegisterInfo &TRI) {
    init(TRI);
  }

  /// Initialize and clear the set.
  void init(const TargetRegisterInfo &TRI) {
    this->TRI = &TRI;
    Units.reset();
    Units.resize(TRI.getNumRegUnits());
  }

  /// Clears the set.
  void clear() { Units.reset(); }

  /// Returns true if the set is empty.
  bool empty() const { return Units.empty(); }

  /// Adds register units covered by physical register \p Reg.
  void addReg(unsigned Reg) {
    for (MCRegUnitIterator Unit(Reg, TRI); Unit.isValid(); ++Unit)
      Units.set(*Unit);
  }

  /// \brief Adds register units covered by physical register \p Reg that are
  /// part of the lanemask \p Mask.
  void addRegMasked(unsigned Reg, LaneBitmask Mask) {
    for (MCRegUnitMaskIterator Unit(Reg, TRI); Unit.isValid(); ++Unit) {
      LaneBitmask UnitMask = (*Unit).second;
      if (UnitMask.none() || (UnitMask & Mask).any())
        Units.set((*Unit).first);
    }
  }

  /// Removes all register units covered by physical register \p Reg.
  void removeReg(unsigned Reg) {
    for (MCRegUnitIterator Unit(Reg, TRI); Unit.isValid(); ++Unit)
      Units.reset(*Unit);
  }

  /// Removes register units not preserved by the regmask \p RegMask.
  /// The regmask has the same format as the one in the RegMask machine operand.
  void removeRegsNotPreserved(const uint32_t *RegMask);

  /// Adds register units not preserved by the regmask \p RegMask.
  /// The regmask has the same format as the one in the RegMask machine operand.
  void addRegsInMask(const uint32_t *RegMask);

  /// Returns true if no part of physical register \p Reg is live.
  bool available(unsigned Reg) const {
    for (MCRegUnitIterator Unit(Reg, TRI); Unit.isValid(); ++Unit) {
      if (Units.test(*Unit))
        return false;
    }
    return true;
  }

  /// Updates liveness when stepping backwards over the instruction \p MI.
  /// This removes all register units defined or clobbered in \p MI and then
  /// adds the units used (as in use operands) in \p MI.
  void stepBackward(const MachineInstr &MI);

  /// Adds all register units used, defined or clobbered in \p MI.
  /// This is useful when walking over a range of instruction to find registers
  /// unused over the whole range.
  void accumulate(const MachineInstr &MI);

  /// Adds registers living out of block \p MBB.
  /// Live out registers are the union of the live-in registers of the successor
  /// blocks and pristine registers. Live out registers of the end block are the
  /// callee saved registers.
  void addLiveOuts(const MachineBasicBlock &MBB);

  /// Adds registers living into block \p MBB.
  void addLiveIns(const MachineBasicBlock &MBB);

  /// Adds all register units marked in the bitvector \p RegUnits.
  void addUnits(const BitVector &RegUnits) {
    Units |= RegUnits;
  }
  /// Removes all register units marked in the bitvector \p RegUnits.
  void removeUnits(const BitVector &RegUnits) {
    Units.reset(RegUnits);
  }
  /// Return the internal bitvector representation of the set.
  const BitVector &getBitVector() const {
    return Units;
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_LIVEREGUNITS_H
