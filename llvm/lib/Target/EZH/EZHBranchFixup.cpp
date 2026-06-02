//===-- EZHBranchFixup.cpp - EZH Branch Cleanup Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates redundant GOTO instructions that jump to their direct
// layout successors. Doing this before BitSliceInjection prevents redundant
// gotol_bs instructions from being injected.
//
//===----------------------------------------------------------------------===//

#include "EZH.h"
#include "EZHInstrInfo.h"
#include "EZHSubtarget.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {
class EZHBranchFixup : public MachineFunctionPass {
public:
  static char ID;
  EZHBranchFixup() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool Changed = false;

    for (auto &MBB : MF) {
      MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
      if (I == MBB.end())
        continue;

      if (I->getOpcode() == EZH::GOTO) {
        MachineBasicBlock *TargetMBB = I->getOperand(0).getMBB();
        if (MBB.isLayoutSuccessor(TargetMBB)) {
          I->eraseFromParent();
          Changed = true;
        }
      }
    }
    return Changed;
  }
};

char EZHBranchFixup::ID = 0;
} // namespace

namespace llvm {
FunctionPass *createEZHBranchFixupPass() { return new EZHBranchFixup(); }
} // namespace llvm
