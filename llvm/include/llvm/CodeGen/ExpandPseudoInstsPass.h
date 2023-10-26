//===-- ExpandPseudoInstsPass.h - Pass for expanding pseudo insts-*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ExpandPseudoInstsPass class. It provides a default
// implementation for expandMBB to simplify target-specific pseudo instruction
// expansion passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXPANDPSEUDOINSTSPASS_H
#define LLVM_CODEGEN_EXPANDPSEUDOINSTSPASS_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

/// ExpandPseudoInstsPass - Helper class for expanding pseudo instructions.
class ExpandPseudoInstsPass : public MachineFunctionPass {
protected:
  explicit ExpandPseudoInstsPass(char &ID) : MachineFunctionPass(ID) {}

  /// If MBBI references a pseudo instruction that should be expanded here,
  /// do the expansion and return true.  Otherwise return false.
  virtual bool expandMI(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI,
                        MachineBasicBlock::iterator &NextMBBI) = 0;

  /// Iterate over the instructions in basic block MBB and expand any
  /// pseudo instructions.  Return true if anything was modified.
  bool expandMBB(MachineBasicBlock &MBB);
};

} // namespace llvm

#endif
