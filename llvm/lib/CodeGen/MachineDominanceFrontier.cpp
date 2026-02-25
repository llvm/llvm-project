//===- MachineDominanceFrontier.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

namespace llvm {
template class DominanceFrontierBase<MachineBasicBlock, false>;
}

char MachineDominanceFrontierWrapperPass::ID = 0;

INITIALIZE_PASS_BEGIN(MachineDominanceFrontierWrapperPass,
                      "machine-domfrontier",
                      "Machine Dominance Frontier Construction", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(MachineDominanceFrontierWrapperPass, "machine-domfrontier",
                    "Machine Dominance Frontier Construction", true, true)

MachineDominanceFrontierWrapperPass::MachineDominanceFrontierWrapperPass()
    : MachineFunctionPass(ID) {}

char &llvm::MachineDominanceFrontierID =
    MachineDominanceFrontierWrapperPass::ID;

bool MachineDominanceFrontier::invalidate(
    MachineFunction &F, const PreservedAnalyses &PA,
    MachineFunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<MachineDominanceFrontierAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<AllAnalysesOn<MachineFunction>>() &&
         !PAC.preservedSet<CFGAnalyses>();
}

bool MachineDominanceFrontierWrapperPass::runOnMachineFunction(
    MachineFunction &) {
  MDF.releaseMemory();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MDF.analyze(MDT);
  return false;
}

void MachineDominanceFrontierWrapperPass::releaseMemory() {
  MDF.releaseMemory();
}

void MachineDominanceFrontierWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AnalysisKey MachineDominanceFrontierAnalysis::Key;

MachineDominanceFrontierAnalysis::Result
MachineDominanceFrontierAnalysis::run(MachineFunction &MF,
                                      MachineFunctionAnalysisManager &MFAM) {
  MachineDominanceFrontier MDF;
  auto &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MDF.analyze(MDT);
  return MDF;
}
