//====----- MachineBlockFrequencyInfo.cpp - Machine Block Frequency Analysis ----====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/Analysis/BlockFrequencyImpl.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(MachineBlockFrequencyInfo, "machine-block-freq",
                      "Machine Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_END(MachineBlockFrequencyInfo, "machine-block-freq",
                    "Machine Block Frequency Analysis", true, true)

char MachineBlockFrequencyInfo::ID = 0;


MachineBlockFrequencyInfo::MachineBlockFrequencyInfo() : MachineFunctionPass(ID) {
  initializeMachineBlockFrequencyInfoPass(*PassRegistry::getPassRegistry());
  MBFI = new BlockFrequencyImpl<MachineBasicBlock, MachineFunction,
                                MachineBranchProbabilityInfo>();
}

MachineBlockFrequencyInfo::~MachineBlockFrequencyInfo() {
  delete MBFI;
}

void MachineBlockFrequencyInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineBlockFrequencyInfo::runOnMachineFunction(MachineFunction &F) {
  MachineBranchProbabilityInfo &MBPI = getAnalysis<MachineBranchProbabilityInfo>();
  MBFI->doFunction(&F, &MBPI);
  return false;
}

/// getblockFreq - Return block frequency. Return 0 if we don't have the
/// information. Please note that initial frequency is equal to 1024. It means
/// that we should not rely on the value itself, but only on the comparison to
/// the other block frequencies. We do this to avoid using of floating points.
///
BlockFrequency MachineBlockFrequencyInfo::
getBlockFreq(const MachineBasicBlock *MBB) const {
  return MBFI->getBlockFreq(MBB);
}
