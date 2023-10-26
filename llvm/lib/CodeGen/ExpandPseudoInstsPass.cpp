//===- ExpandPseudoInstsPass.cpp - Pass for expanding pseudo insts*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ExpandPseudoInstsPass class. It provides a default
// implementation for expandMBB to simplify target-specific pseudo instruction
// expansion passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandPseudoInstsPass.h"

namespace llvm {

/// Iterate over the instructions in basic block MBB and expand any
/// pseudo instructions.  Return true if anything was modified.
bool ExpandPseudoInstsPass::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

} // namespace llvm
