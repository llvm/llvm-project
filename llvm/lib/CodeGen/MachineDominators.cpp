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

TEMPLATE_INSTANTIATION(class DomTreeNodeBase<MachineBasicBlock>);
TEMPLATE_INSTANTIATION(class DominatorTreeBase<MachineBasicBlock>);

char MachineDominatorTree::ID = 0;

static RegisterPass<MachineDominatorTree>
E("machinedomtree", "MachineDominator Tree Construction", true);

const PassInfo *const llvm::MachineDominatorsID = &E;

void MachineDominatorTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineDominatorTree::runOnMachineFunction(MachineFunction &F) {
  DT->recalculate(F);

  return false;
}

MachineDominatorTree::MachineDominatorTree()
    : MachineFunctionPass(intptr_t(&ID)) {
  DT = new DominatorTreeBase<MachineBasicBlock>(false);
}

MachineDominatorTree::~MachineDominatorTree() {
  DT->releaseMemory();
  delete DT;
}

void MachineDominatorTree::releaseMemory() {
  DT->releaseMemory();
}
