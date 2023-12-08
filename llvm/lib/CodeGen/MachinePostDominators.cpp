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
#include "llvm/InitializePasses.h"

using namespace llvm;

namespace llvm {
template class DominatorTreeBase<MachineBasicBlock, true>; // PostDomTreeBase

extern bool VerifyMachineDomInfo;
} // namespace llvm

char MachinePostDominatorTree::ID = 0;

//declare initializeMachinePostDominatorTreePass
INITIALIZE_PASS(MachinePostDominatorTree, "machinepostdomtree",
                "MachinePostDominator Tree Construction", true, true)

MachinePostDominatorTree::MachinePostDominatorTree()
    : MachineFunctionPass(ID), PDT(nullptr) {
  initializeMachinePostDominatorTreePass(*PassRegistry::getPassRegistry());
}

FunctionPass *MachinePostDominatorTree::createMachinePostDominatorTreePass() {
  return new MachinePostDominatorTree();
}

bool MachinePostDominatorTree::runOnMachineFunction(MachineFunction &F) {
  PDT = std::make_unique<PostDomTreeT>();
  PDT->recalculate(F);
  return false;
}

void MachinePostDominatorTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachineBasicBlock *MachinePostDominatorTree::findNearestCommonDominator(
    ArrayRef<MachineBasicBlock *> Blocks) const {
  assert(!Blocks.empty());

  MachineBasicBlock *NCD = Blocks.front();
  for (MachineBasicBlock *BB : Blocks.drop_front()) {
    NCD = PDT->findNearestCommonDominator(NCD, BB);

    // Stop when the root is reached.
    if (PDT->isVirtualRoot(PDT->getNode(NCD)))
      return nullptr;
  }

  return NCD;
}

void MachinePostDominatorTree::verifyAnalysis() const {
  if (PDT && VerifyMachineDomInfo)
    if (!PDT->verify(PostDomTreeT::VerificationLevel::Basic)) {
      errs() << "MachinePostDominatorTree verification failed\n";

      abort();
    }
}

void MachinePostDominatorTree::print(llvm::raw_ostream &OS,
                                     const Module *M) const {
  PDT->print(OS);
}
