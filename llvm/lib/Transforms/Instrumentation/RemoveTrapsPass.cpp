//===- RemoveTrapsPass.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/RemoveTrapsPass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "remove-traps"

static constexpr unsigned MaxRandomRate = 1000;

static cl::opt<int> HotPercentileCutoff(
    "remove-traps-percentile-cutoff-hot", cl::init(0),
    cl::desc("Alternative hot percentile cuttoff. By default "
             "`-profile-summary-cutoff-hot` is used."));
static cl::opt<float> RandomRate(
    "remove-traps-random-rate", cl::init(0.0),
    cl::desc(
        "Probability to use for pseudorandom unconditional checks removal."));

STATISTIC(NumChecksTotal, "Number of checks");
STATISTIC(NumChecksRemoved, "Number of removed checks");

static SmallVector<IntrinsicInst *, 16>
removeUbsanTraps(Function &F, FunctionAnalysisManager &FAM,
                 ProfileSummaryInfo *PSI) {
  SmallVector<IntrinsicInst *, 16> Remove;

  if (F.isDeclaration())
    return {};

  auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);

  int BBCounter = 0;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      auto ID = II->getIntrinsicID();
      if (ID != Intrinsic::ubsantrap)
        continue;
      ++NumChecksTotal;

      bool IsHot = false;
      if (PSI) {
        uint64_t Count = 0;
        for (const auto *PR : predecessors(&BB))
          Count += BFI.getBlockProfileCount(PR).value_or(0);

        IsHot = HotPercentileCutoff.getNumOccurrences()
                    ? PSI->isHotCountNthPercentile(HotPercentileCutoff, Count)
                    : PSI->isHotCount(Count);
      }

      if ((IsHot) || ((F.getGUID() + BBCounter++) % MaxRandomRate) <
                         RandomRate * RandomRate) {
        Remove.push_back(II);
        ++NumChecksRemoved;
      }
    }
  }
  return Remove;
}

PreservedAnalyses RemoveTrapsPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();
  auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  ProfileSummaryInfo *PSI =
      MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());

  auto Remove = removeUbsanTraps(F, AM, PSI);
  for (auto *I : Remove)
    I->eraseFromParent();

  return Remove.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
