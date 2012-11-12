//===- MachineDominators.cpp - Machine Dominator Calculation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators on machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"

using namespace llvm;

namespace llvm {
TEMPLATE_INSTANTIATION(class DomTreeNodeBase<MachineBasicBlock>);
TEMPLATE_INSTANTIATION(class DominatorTreeBase<MachineBasicBlock>);
}

char MachineDominatorTree::ID = 0;

INITIALIZE_PASS(MachineDominatorTree, "machinedomtree",
                "MachineDominator Tree Construction", true, true)

char &llvm::MachineDominatorsID = MachineDominatorTree::ID;

void MachineDominatorTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineDominatorTree::runOnMachineFunction(MachineFunction &F) {
  DT->recalculate(F);

  return false;
}

MachineDominatorTree::MachineDominatorTree()
    : MachineFunctionPass(ID) {
  initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
  DT = new DominatorTreeBase<MachineBasicBlock>(false);
}

MachineDominatorTree::~MachineDominatorTree() {
  delete DT;
}

void MachineDominatorTree::releaseMemory() {
  DT->releaseMemory();
}

void MachineDominatorTree::print(raw_ostream &OS, const Module*) const {
  DT->print(OS);
}
