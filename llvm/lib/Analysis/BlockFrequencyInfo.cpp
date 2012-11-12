//=======-------- BlockFrequencyInfo.cpp - Block Frequency Analysis -------=======//
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
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(BlockFrequencyInfo, "block-freq", "Block Frequency Analysis",
                      true, true)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfo)
INITIALIZE_PASS_END(BlockFrequencyInfo, "block-freq", "Block Frequency Analysis",
                    true, true)

char BlockFrequencyInfo::ID = 0;


BlockFrequencyInfo::BlockFrequencyInfo() : FunctionPass(ID) {
  initializeBlockFrequencyInfoPass(*PassRegistry::getPassRegistry());
  BFI = new BlockFrequencyImpl<BasicBlock, Function, BranchProbabilityInfo>();
}

BlockFrequencyInfo::~BlockFrequencyInfo() {
  delete BFI;
}

void BlockFrequencyInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BranchProbabilityInfo>();
  AU.setPreservesAll();
}

bool BlockFrequencyInfo::runOnFunction(Function &F) {
  BranchProbabilityInfo &BPI = getAnalysis<BranchProbabilityInfo>();
  BFI->doFunction(&F, &BPI);
  return false;
}

void BlockFrequencyInfo::print(raw_ostream &O, const Module *) const {
  if (BFI) BFI->print(O);
}

/// getblockFreq - Return block frequency. Return 0 if we don't have the
/// information. Please note that initial frequency is equal to 1024. It means
/// that we should not rely on the value itself, but only on the comparison to
/// the other block frequencies. We do this to avoid using of floating points.
///
BlockFrequency BlockFrequencyInfo::getBlockFreq(const BasicBlock *BB) const {
  return BFI->getBlockFreq(BB);
}
