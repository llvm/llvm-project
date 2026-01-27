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
template class DominanceFrontierBase<MachineBasicBlock, true>;
template class ForwardDominanceFrontierBase<MachineBasicBlock>;
}


char MachineDominanceFrontierWrapperPass::ID = 0;

INITIALIZE_PASS_BEGIN(MachineDominanceFrontierWrapperPass, "machine-domfrontier",
                "Machine Dominance Frontier Construction", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(MachineDominanceFrontierWrapperPass, "machine-domfrontier",
                "Machine Dominance Frontier Construction", true, true)

MachineDominanceFrontierWrapperPass::MachineDominanceFrontierWrapperPass()
    : MachineFunctionPass(ID) {}

char &llvm::MachineDominanceFrontierID = MachineDominanceFrontierWrapperPass::ID;

bool MachineDominanceFrontierWrapperPass::runOnMachineFunction(MachineFunction &) {
  auto& MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  return MDF.analyze(MDT);
}

bool MachineDominanceFrontier::analyze(MachineDominatorTree &MDT) {
  releaseMemory();
  Base.analyze(MDT);
  return false;
}

void MachineDominanceFrontierWrapperPass::releaseMemory() {
  MDF.releaseMemory();
}

void MachineDominanceFrontier::releaseMemory() {
  Base.releaseMemory();
}

void MachineDominanceFrontierWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AnalysisKey MachineDominanceFrontierAnalysis::Key;

MachineDominanceFrontierAnalysis::Result
MachineDominanceFrontierAnalysis::run(MachineFunction &MF,
                                      MachineFunctionAnalysisManager &MFAM) {
  auto& MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MDF.analyze(MDT);
  return MDF;
}
