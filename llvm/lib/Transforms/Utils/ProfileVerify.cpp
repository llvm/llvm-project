//===- ProfileVerify.cpp - Verify profile info for testing ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ProfileVerify.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/BranchProbability.h"

using namespace llvm;
namespace {
class ProfileInjector {
  Function &F;
  FunctionAnalysisManager &FAM;

public:
  static bool supportsBranchWeights(const Instruction &I) {
    return isa<BranchInst>(&I) ||

           isa<SwitchInst>(&I) ||

           isa<IndirectBrInst>(&I) || isa<SelectInst>(&I) ||
           isa<CallBrInst>(&I);
  }

  ProfileInjector(Function &F, FunctionAnalysisManager &FAM) : F(F), FAM(FAM) {}
  bool inject();
};
} // namespace

bool ProfileInjector::inject() {
  auto &BPI = FAM.getResult<BranchProbabilityAnalysis>(F);

  for (auto &BB : F) {
    if (succ_size(&BB) <= 1)
      continue;
    auto *Term = BB.getTerminator();
    assert(Term);
    if (Term->getMetadata(LLVMContext::MD_prof) ||
        !supportsBranchWeights(*Term))
      continue;
    SmallVector<BranchProbability> Probs;
    Probs.reserve(Term->getNumSuccessors());
    for (auto I = 0U, E = Term->getNumSuccessors(); I < E; ++I)
      Probs.emplace_back(BPI.getEdgeProbability(&BB, Term->getSuccessor(I)));

    const auto *FirstZeroDenominator =
        find_if(Probs, [](const BranchProbability &P) {
          return P.getDenominator() == 0;
        });
    assert(FirstZeroDenominator == Probs.end());
    const auto *FirstNonzeroNumerator =
        find_if(Probs, [](const BranchProbability &P) {
          return P.getNumerator() != 0;
        });
    assert(FirstNonzeroNumerator != Probs.end());
    DynamicAPInt LCM(Probs[0].getDenominator());
    DynamicAPInt GCD(FirstNonzeroNumerator->getNumerator());
    for (const auto &Prob : drop_begin(Probs)) {
      if (!Prob.getNumerator())
        continue;
      LCM = llvm::lcm(LCM, DynamicAPInt(Prob.getDenominator()));
      GCD = llvm::lcm(GCD, DynamicAPInt(Prob.getNumerator()));
    }
    SmallVector<uint32_t> Weights;
    Weights.reserve(Term->getNumSuccessors());
    for (const auto &Prob : Probs) {
      auto W = Prob.getNumerator() * LCM / GCD;
      Weights.emplace_back(static_cast<int32_t>((int64_t)W));
    }
    setBranchWeights(*Term, Weights, false);
  }
  return true;
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
  bool Changed = false;
  for (auto &BB : F)
    if (succ_size(&BB) >= 2)
      if (auto *Term = BB.getTerminator())
        if (ProfileInjector::supportsBranchWeights(*Term)) {
          if (!Term->getMetadata(LLVMContext::MD_prof)) {
            F.getContext().emitError("Profile verification failed");
          } else {
            Changed = true;
          }
        }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
