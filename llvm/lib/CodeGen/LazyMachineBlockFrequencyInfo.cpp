///===- LazyMachineBlockFrequencyInfo.cpp - Lazy Machine Block Frequency --===//
///
///                     The LLVM Compiler Infrastructure
///
/// This file is distributed under the University of Illinois Open Source
/// License. See LICENSE.TXT for details.
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
  LMBFI.getCalculated().print(OS, M);
}

void LazyMachineBlockFrequencyInfoPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.addRequired<MachineLoopInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LazyMachineBlockFrequencyInfoPass::releaseMemory() {
  LMBFI.releaseMemory();
}

bool LazyMachineBlockFrequencyInfoPass::runOnMachineFunction(
    MachineFunction &MF) {
  auto &BPIPass = getAnalysis<MachineBranchProbabilityInfo>();
  auto &LI = getAnalysis<MachineLoopInfo>();
  LMBFI.setAnalysis(&MF, &BPIPass, &LI);
  return false;
}

void LazyMachineBlockFrequencyInfoPass::getLazyMachineBFIAnalysisUsage(
    AnalysisUsage &AU) {
  AU.addRequired<LazyMachineBlockFrequencyInfoPass>();
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.addRequired<MachineLoopInfo>();
}

void llvm::initializeLazyMachineBFIPassPass(PassRegistry &Registry) {
  INITIALIZE_PASS_DEPENDENCY(LazyMachineBlockFrequencyInfoPass);
  INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo);
  INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo);
}
