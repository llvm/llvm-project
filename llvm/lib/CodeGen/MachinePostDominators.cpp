//===- MachinePostDominators.cpp -Machine Post Dominator Calculation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// post dominators on machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePostDominators.h"

using namespace llvm;

namespace llvm {
template class DominatorTreeBase<MachineBasicBlock, true>; // PostDomTreeBase
}

char MachinePostDominatorTree::ID = 0;

//declare initializeMachinePostDominatorTreePass
INITIALIZE_PASS(MachinePostDominatorTree, "machinepostdomtree",
                "MachinePostDominator Tree Construction", true, true)

MachinePostDominatorTree::MachinePostDominatorTree() : MachineFunctionPass(ID) {
  initializeMachinePostDominatorTreePass(*PassRegistry::getPassRegistry());
  DT = new PostDomTreeBase<MachineBasicBlock>();
}

FunctionPass *
MachinePostDominatorTree::createMachinePostDominatorTreePass() {
  return new MachinePostDominatorTree();
}

bool
MachinePostDominatorTree::runOnMachineFunction(MachineFunction &F) {
  DT->recalculate(F);
  return false;
}

MachinePostDominatorTree::~MachinePostDominatorTree() {
  delete DT;
}

void
MachinePostDominatorTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void
MachinePostDominatorTree::print(llvm::raw_ostream &OS, const Module *M) const {
  DT->print(OS);
}
