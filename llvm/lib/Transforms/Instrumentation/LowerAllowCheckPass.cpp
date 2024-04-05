//===- LowerAllowCheckPass.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/LowerAllowCheckPass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include <memory>
#include <random>

using namespace llvm;

#define DEBUG_TYPE "lower-allow-check"

static cl::opt<int>
    HotPercentileCutoff("lower-allow-check-percentile-cutoff-hot",
                        cl::desc("Hot percentile cuttoff."));

static cl::opt<float>
    RandomRate("lower-allow-check-random-rate",
               cl::desc("Probability value in the range [0.0, 1.0] of "
                        "unconditional pseudo-random checks removal."));

STATISTIC(NumChecksTotal, "Number of checks");
STATISTIC(NumChecksRemoved, "Number of removed checks");

static bool removeUbsanTraps(Function &F, const BlockFrequencyInfo &BFI,
                             const ProfileSummaryInfo *PSI) {
  SmallVector<std::pair<IntrinsicInst *, bool>, 16> ReplaceWithValue;
  std::unique_ptr<RandomNumberGenerator> Rng;

  // TODO:
  // https://github.com/llvm/llvm-project/pull/84858#discussion_r1520603139
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
      case Intrinsic::allow_ubsan_check:
      case Intrinsic::allow_runtime_check: {
        ++NumChecksTotal;

        bool IsHot = false;
        if (PSI) {
          uint64_t Count = BFI.getBlockProfileCount(&BB).value_or(0);
          IsHot = PSI->isHotCountNthPercentile(HotPercentileCutoff, Count);
        }

        bool ToRemove = ShouldRemove(IsHot);
        ReplaceWithValue.push_back({
            II,
            ToRemove,
        });
        if (ToRemove)
          ++NumChecksRemoved;
        break;
      }
      default:
        break;
      }
    }
  }

  for (auto [I, V] : ReplaceWithValue) {
    I->replaceAllUsesWith(ConstantInt::getBool(I->getType(), !V));
    I->eraseFromParent();
  }

  return !ReplaceWithValue.empty();
}

PreservedAnalyses LowerAllowCheckPass::run(Function &F,
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

bool LowerAllowCheckPass::IsRequested() {
  return RandomRate.getNumOccurrences() ||
         HotPercentileCutoff.getNumOccurrences();
}
