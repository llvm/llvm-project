//==-- llvm/CodeGen/GlobalISel/InstructionSelector.h -------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API for the instruction selector.
/// This class is responsible for selecting machine instructions.
/// It's implemented by the target. It's used by the InstructionSelect pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H
#define LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H

#include "llvm/ADT/Optional.h"
#include <cstdint>

namespace llvm {
class MachineInstr;
class MachineOperand;
class MachineRegisterInfo;
class RegisterBankInfo;
class TargetInstrInfo;
class TargetRegisterInfo;

/// Provides the logic to select generic machine instructions.
class InstructionSelector {
public:
  virtual ~InstructionSelector() {}

  /// Select the (possibly generic) instruction \p I to only use target-specific
  /// opcodes. It is OK to insert multiple instructions, but they cannot be
  /// generic pre-isel instructions.
  ///
  /// \returns whether selection succeeded.
  /// \pre  I.getParent() && I.getParent()->getParent()
  /// \post
  ///   if returns true:
  ///     for I in all mutated/inserted instructions:
  ///       !isPreISelGenericOpcode(I.getOpcode())
  ///
  virtual bool select(MachineInstr &I) const = 0;

protected:
  InstructionSelector();

  /// Mutate the newly-selected instruction \p I to constrain its (possibly
  /// generic) virtual register operands to the instruction's register class.
  /// This could involve inserting COPYs before (for uses) or after (for defs).
  /// This requires the number of operands to match the instruction description.
  /// \returns whether operand regclass constraining succeeded.
  ///
  // FIXME: Not all instructions have the same number of operands. We should
  // probably expose a constrain helper per operand and let the target selector
  // constrain individual registers, like fast-isel.
  bool constrainSelectedInstRegOperands(MachineInstr &I,
                                        const TargetInstrInfo &TII,
                                        const TargetRegisterInfo &TRI,
                                        const RegisterBankInfo &RBI) const;

  Optional<int64_t> getConstantVRegVal(unsigned VReg,
                                       const MachineRegisterInfo &MRI) const;

  bool isOperandImmEqual(const MachineOperand &MO, int64_t Value,
                         const MachineRegisterInfo &MRI) const;

  bool isObviouslySafeToFold(MachineInstr &MI) const;
};

} // End namespace llvm.

#endif
