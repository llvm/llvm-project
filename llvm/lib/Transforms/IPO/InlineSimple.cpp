//===- InlineSimple.cpp - Code to perform simple function inlining --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InlinerPass.h"

using namespace llvm;

#define DEBUG_TYPE "inline"

namespace {

/// \brief Actual inliner pass implementation.
///
/// The common implementation of the inlining logic is shared between this
/// inliner pass and the always inliner pass. The two passes use different cost
/// analyses to determine when to inline.
class SimpleInliner : public Inliner {
  // This field is populated based on one of the following:
  //  * optimization or size-optimization levels,
  //  * the --inline-threshold flag, or
  //  * a user specified value.
  int DefaultThreshold;

public:
  SimpleInliner()
      : Inliner(ID), DefaultThreshold(llvm::getDefaultInlineThreshold()) {
    initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
  }

  explicit SimpleInliner(int Threshold)
      : Inliner(ID), DefaultThreshold(Threshold) {
    initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
  }

  static char ID; // Pass identification, replacement for typeid

  InlineCost getInlineCost(CallSite CS) override {
    Function *Callee = CS.getCalledFunction();
    TargetTransformInfo &TTI = TTIWP->getTTI(*Callee);
    return llvm::getInlineCost(CS, DefaultThreshold, TTI, ACT, PSI);
  }

  bool runOnSCC(CallGraphSCC &SCC) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  TargetTransformInfoWrapperPass *TTIWP;
};

} // end anonymous namespace

char SimpleInliner::ID = 0;
INITIALIZE_PASS_BEGIN(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)

Pass *llvm::createFunctionInliningPass() { return new SimpleInliner(); }

Pass *llvm::createFunctionInliningPass(int Threshold) {
  return new SimpleInliner(Threshold);
}

Pass *llvm::createFunctionInliningPass(unsigned OptLevel,
                                       unsigned SizeOptLevel) {
  return new SimpleInliner(
      llvm::computeThresholdFromOptLevels(OptLevel, SizeOptLevel));
}

bool SimpleInliner::runOnSCC(CallGraphSCC &SCC) {
  TTIWP = &getAnalysis<TargetTransformInfoWrapperPass>();
  return Inliner::runOnSCC(SCC);
}

void SimpleInliner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetTransformInfoWrapperPass>();
  Inliner::getAnalysisUsage(AU);
}
