//===- LazyBlockFrequencyInfo.cpp - Lazy Block Frequency Analysis ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an alternative analysis pass to BlockFrequencyInfoWrapperPass.  The
// difference is that with this pass the block frequencies are not computed when
// the analysis pass is executed but rather when the BFI results is explicitly
// requested by the analysis client.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "lazy-block-freq"

INITIALIZE_PASS_BEGIN(LazyBlockFrequencyInfoPass, DEBUG_TYPE,
                      "Lazy Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LazyBlockFrequencyInfoPass, DEBUG_TYPE,
                    "Lazy Block Frequency Analysis", true, true)

char LazyBlockFrequencyInfoPass::ID = 0;

LazyBlockFrequencyInfoPass::LazyBlockFrequencyInfoPass() : FunctionPass(ID) {
  initializeLazyBlockFrequencyInfoPassPass(*PassRegistry::getPassRegistry());
}

void LazyBlockFrequencyInfoPass::print(raw_ostream &OS, const Module *) const {
  LBFI.getCalculated().print(OS);
}

void LazyBlockFrequencyInfoPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BranchProbabilityInfoWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.setPreservesAll();
}

void LazyBlockFrequencyInfoPass::releaseMemory() { LBFI.releaseMemory(); }

bool LazyBlockFrequencyInfoPass::runOnFunction(Function &F) {
  BranchProbabilityInfo &BPI =
      getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  LBFI.setAnalysis(&F, &BPI, &LI);
  return false;
}

void LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AnalysisUsage &AU) {
  AU.addRequired<BranchProbabilityInfoWrapperPass>();
  AU.addRequired<LazyBlockFrequencyInfoPass>();
  AU.addRequired<LoopInfoWrapperPass>();
}

void llvm::initializeLazyBFIPassPass(PassRegistry &Registry) {
  INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass);
  INITIALIZE_PASS_DEPENDENCY(LazyBlockFrequencyInfoPass);
  INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
}
