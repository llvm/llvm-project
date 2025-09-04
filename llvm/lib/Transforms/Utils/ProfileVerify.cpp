//===- ProfileVerify.cpp - Verify profile info for testing ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ProfileVerify.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
static cl::opt<int64_t>
    DefaultFunctionEntryCount("profcheck-default-function-entry-count",
                              cl::init(1000));
static cl::opt<bool>
    AnnotateSelect("profcheck-annotate-select", cl::init(true),
                   cl::desc("Also inject (if missing) and verify MD_prof for "
                            "`select` instructions"));
static cl::opt<uint32_t> SelectTrueWeight(
    "profcheck-default-select-true-weight", cl::init(2U),
    cl::desc("When annotating `select` instructions, this value will be used "
             "for the first ('true') case."));
static cl::opt<uint32_t> SelectFalseWeight(
    "profcheck-default-select-false-weight", cl::init(3U),
    cl::desc("When annotating `select` instructions, this value will be used "
             "for the second ('false') case."));
namespace {
class ProfileInjector {
  Function &F;
  FunctionAnalysisManager &FAM;

public:
  static const Instruction *
  getTerminatorBenefitingFromMDProf(const BasicBlock &BB) {
    if (succ_size(&BB) < 2)
      return nullptr;
    auto *Term = BB.getTerminator();
    return (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
            isa<IndirectBrInst>(Term) || isa<CallBrInst>(Term))
               ? Term
               : nullptr;
  }

  static Instruction *getTerminatorBenefitingFromMDProf(BasicBlock &BB) {
    return const_cast<Instruction *>(
        getTerminatorBenefitingFromMDProf(const_cast<const BasicBlock &>(BB)));
  }

  ProfileInjector(Function &F, FunctionAnalysisManager &FAM) : F(F), FAM(FAM) {}
  bool inject();
};
} // namespace

// FIXME: currently this injects only for terminators. Select isn't yet
// supported.
bool ProfileInjector::inject() {
  // Get whatever branch probability info can be derived from the given IR -
  // whether it has or not metadata. The main intention for this pass is to
  // ensure that other passes don't drop or "forget" to update MD_prof. We do
  // this as a mode in which lit tests would run. We want to avoid changing the
  // behavior of those tests. A pass may use BPI (or BFI, which is computed from
  // BPI). If no metadata is present, BPI is guesstimated by
  // BranchProbabilityAnalysis. The injector (this pass) only persists whatever
  // information the analysis provides, in other words, the pass being tested
  // will get the same BPI it does if the injector wasn't running.
  auto &BPI = FAM.getResult<BranchProbabilityAnalysis>(F);

  // Inject a function count if there's none. It's reasonable for a pass to
  // want to clear the MD_prof of a function with zero entry count. If the
  // original profile (iFDO or AFDO) is empty for a function, it's simpler to
  // require assigning it the 0-entry count explicitly than to mark every branch
  // as cold (we do want some explicit information in the spirit of what this
  // verifier wants to achieve - make dropping / corrupting MD_prof
  // unit-testable)
  if (!F.getEntryCount(/*AllowSynthetic=*/true))
    F.setEntryCount(DefaultFunctionEntryCount);
  // If there is an entry count that's 0, then don't bother injecting. We won't
  // verify these either.
  if (F.getEntryCount(/*AllowSynthetic=*/true)->getCount() == 0)
    return false;
  bool Changed = false;
  for (auto &BB : F) {
    if (AnnotateSelect) {
      for (auto &I : BB) {
        if (isa<SelectInst>(I) && !I.getMetadata(LLVMContext::MD_prof))
          setBranchWeights(I, {SelectTrueWeight, SelectFalseWeight},
                           /*IsExpected=*/false);
      }
    }
    auto *Term = getTerminatorBenefitingFromMDProf(BB);
    if (!Term || Term->getMetadata(LLVMContext::MD_prof))
      continue;
    SmallVector<BranchProbability> Probs;
    Probs.reserve(Term->getNumSuccessors());
    for (auto I = 0U, E = Term->getNumSuccessors(); I < E; ++I)
      Probs.emplace_back(BPI.getEdgeProbability(&BB, Term->getSuccessor(I)));

    assert(llvm::find_if(Probs,
                         [](const BranchProbability &P) {
                           return P.isUnknown();
                         }) == Probs.end() &&
           "All branch probabilities should be valid");
    const auto *FirstZeroDenominator =
        find_if(Probs, [](const BranchProbability &P) {
          return P.getDenominator() == 0;
        });
    (void)FirstZeroDenominator;
    assert(FirstZeroDenominator == Probs.end());
    const auto *FirstNonZeroNumerator =
        find_if(Probs, [](const BranchProbability &P) { return !P.isZero(); });
    assert(FirstNonZeroNumerator != Probs.end());
    DynamicAPInt LCM(Probs[0].getDenominator());
    DynamicAPInt GCD(FirstNonZeroNumerator->getNumerator());
    for (const auto &Prob : drop_begin(Probs)) {
      if (!Prob.getNumerator())
        continue;
      LCM = llvm::lcm(LCM, DynamicAPInt(Prob.getDenominator()));
      GCD = llvm::gcd(GCD, DynamicAPInt(Prob.getNumerator()));
    }
    SmallVector<uint32_t> Weights;
    Weights.reserve(Term->getNumSuccessors());
    for (const auto &Prob : Probs) {
      DynamicAPInt W =
          (Prob.getNumerator() * LCM / GCD) / Prob.getDenominator();
      Weights.emplace_back(static_cast<uint32_t>((int64_t)W));
    }
    setBranchWeights(*Term, Weights, /*IsExpected=*/false);
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses ProfileInjectorPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  ProfileInjector PI(F, FAM);
  if (!PI.inject())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

PreservedAnalyses ProfileVerifierPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  const auto EntryCount = F.getEntryCount(/*AllowSynthetic=*/true);
  if (!EntryCount) {
    auto *MD = F.getMetadata(LLVMContext::MD_prof);
    if (!MD || !isExplicitlyUnknownProfileMetadata(*MD)) {
      F.getContext().emitError("Profile verification failed: function entry "
                               "count missing (set to 0 if cold)");
      return PreservedAnalyses::all();
    }
  } else if (EntryCount->getCount() == 0) {
    return PreservedAnalyses::all();
  }
  for (const auto &BB : F) {
    if (AnnotateSelect) {
      for (const auto &I : BB)
        if (isa<SelectInst>(I) && !I.getMetadata(LLVMContext::MD_prof))
          F.getContext().emitError(
              "Profile verification failed: select annotation missing");
    }
    if (const auto *Term =
            ProfileInjector::getTerminatorBenefitingFromMDProf(BB))
      if (!Term->getMetadata(LLVMContext::MD_prof))
        F.getContext().emitError(
            "Profile verification failed: branch annotation missing");
  }
  return PreservedAnalyses::all();
}
