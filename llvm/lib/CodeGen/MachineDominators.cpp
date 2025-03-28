//===- MachineDominators.cpp - Machine Dominator Calculation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators on machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

using namespace llvm;

namespace llvm {
// Always verify dominfo if expensive checking is enabled.
#ifdef EXPENSIVE_CHECKS
bool VerifyMachineDomInfo = true;
#else
bool VerifyMachineDomInfo = false;
#endif
} // namespace llvm

static cl::opt<bool, true> VerifyMachineDomInfoX(
    "verify-machine-dom-info", cl::location(VerifyMachineDomInfo), cl::Hidden,
    cl::desc("Verify machine dominator info (time consuming)"));

namespace llvm {
template class DomTreeNodeBase<MachineBasicBlock>;
template class DominatorTreeBase<MachineBasicBlock, false>; // DomTreeBase

namespace DomTreeBuilder {
template void Calculate<MBBDomTree>(MBBDomTree &DT);
template void CalculateWithUpdates<MBBDomTree>(MBBDomTree &DT, MBBUpdates U);

template void InsertEdge<MBBDomTree>(MBBDomTree &DT, MachineBasicBlock *From,
                                     MachineBasicBlock *To);

template void DeleteEdge<MBBDomTree>(MBBDomTree &DT, MachineBasicBlock *From,
                                     MachineBasicBlock *To);

template void ApplyUpdates<MBBDomTree>(MBBDomTree &DT, MBBDomTreeGraphDiff &,
                                       MBBDomTreeGraphDiff *);

template bool Verify<MBBDomTree>(const MBBDomTree &DT,
                                 MBBDomTree::VerificationLevel VL);
} // namespace DomTreeBuilder
}

bool MachineDominatorTree::invalidate(
    MachineFunction &, const PreservedAnalyses &PA,
    MachineFunctionAnalysisManager::Invalidator &) {
  // Check whether the analysis, all analyses on machine functions, or the
  // machine function's CFG have been preserved.
  auto PAC = PA.getChecker<MachineDominatorTreeAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<AllAnalysesOn<MachineFunction>>() &&
         !PAC.preservedSet<CFGAnalyses>();
}

AnalysisKey MachineDominatorTreeAnalysis::Key;

MachineDominatorTreeAnalysis::Result
MachineDominatorTreeAnalysis::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &) {
  return MachineDominatorTree(MF);
}

PreservedAnalyses
MachineDominatorTreePrinterPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  OS << "MachineDominatorTree for machine function: " << MF.getName() << '\n';
  MFAM.getResult<MachineDominatorTreeAnalysis>(MF).print(OS);
  return PreservedAnalyses::all();
}

char MachineDominatorTreeWrapperPass::ID = 0;

INITIALIZE_PASS(MachineDominatorTreeWrapperPass, "machinedomtree",
                "MachineDominator Tree Construction", true, true)

MachineDominatorTreeWrapperPass::MachineDominatorTreeWrapperPass()
    : MachineFunctionPass(ID) {
  initializeMachineDominatorTreeWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

char &llvm::MachineDominatorsID = MachineDominatorTreeWrapperPass::ID;

bool MachineDominatorTreeWrapperPass::runOnMachineFunction(MachineFunction &F) {
  DT = MachineDominatorTree(F);
  return false;
}

void MachineDominatorTreeWrapperPass::releaseMemory() { DT.reset(); }

void MachineDominatorTreeWrapperPass::verifyAnalysis() const {
  if (VerifyMachineDomInfo && DT)
    if (!DT->verify(MachineDominatorTree::VerificationLevel::Basic))
      report_fatal_error("MachineDominatorTree verification failed!");
}

void MachineDominatorTreeWrapperPass::print(raw_ostream &OS,
                                            const Module *) const {
  if (DT)
    DT->print(OS);
}
