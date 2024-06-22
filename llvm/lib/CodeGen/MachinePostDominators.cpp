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
#include "llvm/Support/GenericDomTreeConstruction.h"

using namespace llvm;

namespace llvm {
template class DominatorTreeBase<MachineBasicBlock, true>; // PostDomTreeBase

namespace DomTreeBuilder {

template void Calculate<MBBPostDomTree>(MBBPostDomTree &DT);
template void InsertEdge<MBBPostDomTree>(MBBPostDomTree &DT,
                                         MachineBasicBlock *From,
                                         MachineBasicBlock *To);
template void DeleteEdge<MBBPostDomTree>(MBBPostDomTree &DT,
                                         MachineBasicBlock *From,
                                         MachineBasicBlock *To);
template void ApplyUpdates<MBBPostDomTree>(MBBPostDomTree &DT,
                                           MBBPostDomTreeGraphDiff &,
                                           MBBPostDomTreeGraphDiff *);
template bool Verify<MBBPostDomTree>(const MBBPostDomTree &DT,
                                     MBBPostDomTree::VerificationLevel VL);

} // namespace DomTreeBuilder
extern bool VerifyMachineDomInfo;
} // namespace llvm

char MachinePostDominatorTreeWrapperPass::ID = 0;

//declare initializeMachinePostDominatorTreePass
INITIALIZE_PASS(MachinePostDominatorTreeWrapperPass, "machinepostdomtree",
                "MachinePostDominator Tree Construction", true, true)

MachinePostDominatorTreeWrapperPass::MachinePostDominatorTreeWrapperPass()
    : MachineFunctionPass(ID), PDT() {
  initializeMachinePostDominatorTreeWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

bool MachinePostDominatorTreeWrapperPass::runOnMachineFunction(
    MachineFunction &F) {
  PDT = MachinePostDominatorTree();
  PDT->recalculate(F);
  return false;
}

void MachinePostDominatorTreeWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachineBasicBlock *MachinePostDominatorTree::findNearestCommonDominator(
    ArrayRef<MachineBasicBlock *> Blocks) const {
  assert(!Blocks.empty());

  MachineBasicBlock *NCD = Blocks.front();
  for (MachineBasicBlock *BB : Blocks.drop_front()) {
    NCD = Base::findNearestCommonDominator(NCD, BB);

    // Stop when the root is reached.
    if (isVirtualRoot(getNode(NCD)))
      return nullptr;
  }

  return NCD;
}

void MachinePostDominatorTreeWrapperPass::verifyAnalysis() const {
  if (VerifyMachineDomInfo && PDT &&
      !PDT->verify(MachinePostDominatorTree::VerificationLevel::Basic))
    report_fatal_error("MachinePostDominatorTree verification failed!");
}

void MachinePostDominatorTreeWrapperPass::print(llvm::raw_ostream &OS,
                                                const Module *M) const {
  PDT->print(OS);
}
