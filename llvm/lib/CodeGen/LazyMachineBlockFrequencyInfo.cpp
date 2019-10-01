///===- LazyMachineBlockFrequencyInfo.cpp - Lazy Machine Block Frequency --===//
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
/// See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===---------------------------------------------------------------------===//
/// \file
/// This is an alternative analysis pass to MachineBlockFrequencyInfo.  The
/// difference is that with this pass the block frequencies are not computed
/// when the analysis pass is executed but rather when the BFI result is
/// explicitly requested by the analysis client.
///
///===---------------------------------------------------------------------===//

#include "llvm/CodeGen/LazyMachineBlockFrequencyInfo.h"

using namespace llvm;

#define DEBUG_TYPE "lazy-machine-block-freq"

INITIALIZE_PASS_BEGIN(LazyMachineBlockFrequencyInfoPass, DEBUG_TYPE,
                      "Lazy Machine Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(LazyMachineBlockFrequencyInfoPass, DEBUG_TYPE,
                    "Lazy Machine Block Frequency Analysis", true, true)

char LazyMachineBlockFrequencyInfoPass::ID = 0;

LazyMachineBlockFrequencyInfoPass::LazyMachineBlockFrequencyInfoPass()
    : MachineFunctionPass(ID) {
  initializeLazyMachineBlockFrequencyInfoPassPass(
      *PassRegistry::getPassRegistry());
}

void LazyMachineBlockFrequencyInfoPass::print(raw_ostream &OS,
                                              const Module *M) const {
  getBFI().print(OS, M);
}

void LazyMachineBlockFrequencyInfoPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LazyMachineBlockFrequencyInfoPass::releaseMemory() {
  OwnedMBFI.reset();
  OwnedMLI.reset();
  OwnedMDT.reset();
}

MachineBlockFrequencyInfo &
LazyMachineBlockFrequencyInfoPass::calculateIfNotAvailable() const {
  auto *MBFI = getAnalysisIfAvailable<MachineBlockFrequencyInfo>();
  if (MBFI) {
    LLVM_DEBUG(dbgs() << "MachineBlockFrequencyInfo is available\n");
    return *MBFI;
  }

  auto &MBPI = getAnalysis<MachineBranchProbabilityInfo>();
  auto *MLI = getAnalysisIfAvailable<MachineLoopInfo>();
  auto *MDT = getAnalysisIfAvailable<MachineDominatorTree>();
  LLVM_DEBUG(dbgs() << "Building MachineBlockFrequencyInfo on the fly\n");
  LLVM_DEBUG(if (MLI) dbgs() << "LoopInfo is available\n");

  if (!MLI) {
    LLVM_DEBUG(dbgs() << "Building LoopInfo on the fly\n");
    // First create a dominator tree.
    LLVM_DEBUG(if (MDT) dbgs() << "DominatorTree is available\n");

    if (!MDT) {
      LLVM_DEBUG(dbgs() << "Building DominatorTree on the fly\n");
      OwnedMDT = make_unique<MachineDominatorTree>();
      OwnedMDT->getBase().recalculate(*MF);
      MDT = OwnedMDT.get();
    }

    // Generate LoopInfo from it.
    OwnedMLI = make_unique<MachineLoopInfo>();
    OwnedMLI->getBase().analyze(MDT->getBase());
    MLI = OwnedMLI.get();
  }

  OwnedMBFI = make_unique<MachineBlockFrequencyInfo>();
  OwnedMBFI->calculate(*MF, MBPI, *MLI);
  return *OwnedMBFI.get();
}

bool LazyMachineBlockFrequencyInfoPass::runOnMachineFunction(
    MachineFunction &F) {
  MF = &F;
  return false;
}
