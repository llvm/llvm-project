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
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "lazy-machine-block-freq"

INITIALIZE_PASS_BEGIN(LazyMachineBlockFrequencyInfoPass, DEBUG_TYPE,
                      "Lazy Machine Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_END(LazyMachineBlockFrequencyInfoPass, DEBUG_TYPE,
                    "Lazy Machine Block Frequency Analysis", true, true)

char LazyMachineBlockFrequencyInfoPass::ID = 0;

LazyMachineBlockFrequencyInfoPass::LazyMachineBlockFrequencyInfoPass()
    : MachineFunctionPass(ID) {}

void LazyMachineBlockFrequencyInfoPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LazyMachineBlockFrequencyInfoPass::releaseMemory() {
  OwnedMBFI.reset();
  OwnedMCI.reset();
}

MachineBlockFrequencyInfo &
LazyMachineBlockFrequencyInfoPass::calculateIfNotAvailable() const {
  auto *MBFIWrapper =
      getAnalysisIfAvailable<MachineBlockFrequencyInfoWrapperPass>();
  if (MBFIWrapper) {
    LLVM_DEBUG(dbgs() << "MachineBlockFrequencyInfo is available\n");
    return MBFIWrapper->getMBFI();
  }

  auto &MBPI = getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  auto *MCIWrapper = getAnalysisIfAvailable<MachineCycleInfoWrapperPass>();
  auto *MCI = MCIWrapper ? &MCIWrapper->getCycleInfo() : nullptr;
  LLVM_DEBUG(dbgs() << "Building MachineBlockFrequencyInfo on the fly\n");
  LLVM_DEBUG(if (MCI) dbgs() << "CycleInfo is available\n");

  if (!MCI) {
    LLVM_DEBUG(dbgs() << "Building CycleInfo on the fly\n");
    OwnedMCI = std::make_unique<MachineCycleInfo>();
    OwnedMCI->compute(*MF);
    MCI = OwnedMCI.get();
  }

  OwnedMBFI = std::make_unique<MachineBlockFrequencyInfo>();
  OwnedMBFI->calculate(*MF, MBPI, *MCI);
  return *OwnedMBFI;
}

bool LazyMachineBlockFrequencyInfoPass::runOnMachineFunction(
    MachineFunction &F) {
  MF = &F;
  return false;
}
