//===-- PISAMarkConvergentNoMerge.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark convergent instructions with the NoMerge flag to prevent the branch
// folder from tail-merging blocks that contain them. Convergent instructions
// must execute with specific thread convergence guarantees, and tail merging
// can violate those guarantees by changing the control flow paths that reach
// the instruction.
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "pisa-mark-convergent-no-merge"
#define DEBUG_NAME "PISA mark convergent instructions as NoMerge"

using namespace llvm;

namespace {

class PISAMarkConvergentNoMerge : public MachineFunctionPass {
public:
  static char ID;

  PISAMarkConvergentNoMerge() : MachineFunctionPass(ID) {
    initializePISAMarkConvergentNoMergePass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return DEBUG_NAME; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool Changed = false;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (MI.isConvergent() && !MI.getFlag(MachineInstr::NoMerge)) {
          MI.setFlag(MachineInstr::NoMerge);
          Changed = true;
        }
      }
    }
    return Changed;
  }
};

} // end anonymous namespace

char PISAMarkConvergentNoMerge::ID = 0;
INITIALIZE_PASS(PISAMarkConvergentNoMerge, DEBUG_TYPE, DEBUG_NAME, false, false)

FunctionPass *llvm::createPISAMarkConvergentNoMerge() {
  return new PISAMarkConvergentNoMerge();
}
