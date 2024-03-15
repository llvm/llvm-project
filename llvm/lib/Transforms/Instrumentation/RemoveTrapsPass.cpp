//===- RemoveTrapsPass.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/RemoveTrapsPass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include <memory>
#include <random>

using namespace llvm;

#define DEBUG_TYPE "remove-traps"

static cl::opt<int> HotPercentileCutoff(
    "remove-traps-percentile-cutoff-hot", cl::init(0),
    cl::desc("Alternative hot percentile cuttoff. By default "
             "`-profile-summary-cutoff-hot` is used."));

static cl::opt<float>
    RandomRate("remove-traps-random-rate", cl::init(0.0),
               cl::desc("Probability value in the range [0.0, 1.0] of "
                        "unconditional pseudo-random checks removal."));

STATISTIC(NumChecksTotal, "Number of checks");
STATISTIC(NumChecksRemoved, "Number of removed checks");

static bool removeUbsanTraps(Function &F, const BlockFrequencyInfo &BFI,
                             const ProfileSummaryInfo *PSI) {
  SmallVector<IntrinsicInst *, 16> Remove;
  std::unique_ptr<RandomNumberGenerator> Rng;

  auto ShouldRemove = [&](bool IsHot) {
    if (!RandomRate.getNumOccurrences())
      return IsHot;
    if (!Rng)
      Rng = F.getParent()->createRNG(F.getName());
    std::bernoulli_distribution D(RandomRate);
    return D(*Rng);
  };

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      auto ID = II->getIntrinsicID();
      switch (ID) {
      case Intrinsic::ubsantrap: {
        ++NumChecksTotal;

        bool IsHot = false;
        if (PSI) {
          uint64_t Count = 0;
          for (const auto *PR : predecessors(&BB))
            Count += BFI.getBlockProfileCount(PR).value_or(0);

          IsHot =
              HotPercentileCutoff.getNumOccurrences()
                  ? (HotPercentileCutoff > 0 &&
                     PSI->isHotCountNthPercentile(HotPercentileCutoff, Count))
                  : PSI->isHotCount(Count);
        }

        if (ShouldRemove(IsHot)) {
          Remove.push_back(II);
          ++NumChecksRemoved;
        }
        break;
      }
      default:
        break;
      }
    }
  }

  for (IntrinsicInst *I : Remove)
    I->eraseFromParent();

  return !Remove.empty();
}

PreservedAnalyses RemoveTrapsPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();
  auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  ProfileSummaryInfo *PSI =
      MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
  BlockFrequencyInfo &BFI = AM.getResult<BlockFrequencyAnalysis>(F);

  return removeUbsanTraps(F, BFI, PSI) ? PreservedAnalyses::none()
                                       : PreservedAnalyses::all();
}
