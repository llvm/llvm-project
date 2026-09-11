//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the base class used by the several RISC-V Pseudo
// Instruction Expansion passes. This avoids having to re-implement some of the
// boilerplate needed in these passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVEXPANDPSEUDOBASE_H
#define LLVM_LIB_TARGET_RISCV_RISCVEXPANDPSEUDOBASE_H

#include "RISCV.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class RISCVSubtarget;
class RISCVInstrInfo;

class RISCVExpandPseudoImplBase {
public:
  /// Expand a subset of pseudos in the current function. Returns whether the
  /// function was modified.
  ///
  /// This will assert if expansion increased the estimated size of the
  /// function.
  bool run(MachineFunction &MF);

  virtual ~RISCVExpandPseudoImplBase() = default;

protected:
  /// The Subtarget for the current function.
  const RISCVSubtarget *STI;

  /// The derived TargetInstrInfo for the current function.
  const RISCVInstrInfo *TII;

  /// This method should be implemented to expand the instruction at `*MBBI`.
  /// The iteration over the current basic block will continue at `NextMBBI`.
  /// This method should return `true` if it replaced the instruction at
  /// `*MBBI`.
  virtual bool expandMI(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI,
                        MachineBasicBlock::iterator &NextMBBI) const {
    reportFatalInternalError("Expand Pseudos not yet implemented.");
    return false;
  }

private:
  bool expandMBB(MachineBasicBlock &MBB) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVEXPANDPSEUDOBASE_H
