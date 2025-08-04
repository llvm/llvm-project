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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include <memory>
#include <random>

using namespace llvm;

#define DEBUG_TYPE "lower-allow-check"

static cl::opt<int>
    HotPercentileCutoff("lower-allow-check-percentile-cutoff-hot",
                        cl::desc("Hot percentile cutoff."));

static cl::opt<float>
    RandomRate("lower-allow-check-random-rate",
               cl::desc("Probability value in the range [0.0, 1.0] of "
                        "unconditional pseudo-random checks."));

STATISTIC(NumChecksTotal, "Number of checks");
STATISTIC(NumChecksRemoved, "Number of removed checks");

struct RemarkInfo {
  ore::NV Kind;
  ore::NV F;
  ore::NV BB;
  explicit RemarkInfo(IntrinsicInst *II)
      : Kind("Kind", II->getArgOperand(0)),
        F("Function", II->getParent()->getParent()),
        BB("Block", II->getParent()->getName()) {}
};

static void emitRemark(IntrinsicInst *II, OptimizationRemarkEmitter &ORE,
                       bool Removed) {
  if (Removed) {
    ORE.emit([&]() {
      RemarkInfo Info(II);
      return OptimizationRemark(DEBUG_TYPE, "Removed", II)
             << "Removed check: Kind=" << Info.Kind << " F=" << Info.F
             << " BB=" << Info.BB;
    });
  } else {
    ORE.emit([&]() {
      RemarkInfo Info(II);
      return OptimizationRemarkMissed(DEBUG_TYPE, "Allowed", II)
             << "Allowed check: Kind=" << Info.Kind << " F=" << Info.F
             << " BB=" << Info.BB;
    });
  }
}

static bool lowerAllowChecks(Function &F, const BlockFrequencyInfo &BFI,
                             const ProfileSummaryInfo *PSI,
                             OptimizationRemarkEmitter &ORE,
                             const LowerAllowCheckPass::Options &Opts) {
  SmallVector<std::pair<IntrinsicInst *, bool>, 16> ReplaceWithValue;
  std::unique_ptr<RandomNumberGenerator> Rng;

  auto GetRng = [&]() -> RandomNumberGenerator & {
    if (!Rng)
      Rng = F.getParent()->createRNG(F.getName());
    return *Rng;
  };

  auto GetCutoff = [&](const IntrinsicInst *II) -> unsigned {
    if (HotPercentileCutoff.getNumOccurrences())
      return HotPercentileCutoff;
    else if (II->getIntrinsicID() == Intrinsic::allow_ubsan_check) {
      auto *Kind = cast<ConstantInt>(II->getArgOperand(0));
      if (Kind->getZExtValue() < Opts.cutoffs.size())
        return Opts.cutoffs[Kind->getZExtValue()];
    } else if (II->getIntrinsicID() == Intrinsic::allow_runtime_check) {
      return Opts.runtime_check;
    }

    return 0;
  };

  auto ShouldRemoveHot = [&](const BasicBlock &BB, unsigned int cutoff) {
    return (cutoff == 1000000) ||
           (PSI && PSI->isHotCountNthPercentile(
                       cutoff, BFI.getBlockProfileCount(&BB).value_or(0)));
  };

  auto ShouldRemoveRandom = [&]() {
    return RandomRate.getNumOccurrences() &&
           !std::bernoulli_distribution(RandomRate)(GetRng());
  };

  auto ShouldRemove = [&](const IntrinsicInst *II) {
    unsigned int cutoff = GetCutoff(II);
    return ShouldRemoveRandom() || ShouldRemoveHot(*(II->getParent()), cutoff);
  };

  for (Instruction &I : instructions(F)) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    auto ID = II->getIntrinsicID();
    switch (ID) {
    case Intrinsic::allow_ubsan_check:
    case Intrinsic::allow_runtime_check: {
      ++NumChecksTotal;

      bool ToRemove = ShouldRemove(II);

      ReplaceWithValue.push_back({
          II,
          ToRemove,
      });
      if (ToRemove)
        ++NumChecksRemoved;
      emitRemark(II, ORE, ToRemove);
      break;
    }
    default:
      break;
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
  OptimizationRemarkEmitter &ORE =
      AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  return lowerAllowChecks(F, BFI, PSI, ORE, Opts)
             // We do not change the CFG, we only replace the intrinsics with
             // true or false.
             ? PreservedAnalyses::none().preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}

bool LowerAllowCheckPass::IsRequested() {
  return RandomRate.getNumOccurrences() ||
         HotPercentileCutoff.getNumOccurrences();
}

void LowerAllowCheckPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<LowerAllowCheckPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << "<";

  // Format is <cutoffs[0,1,2]=70000;cutoffs[5,6,8]=90000>
  // but it's equally valid to specify
  //   cutoffs[0]=70000;cutoffs[1]=70000;cutoffs[2]=70000;cutoffs[5]=90000;...
  // and that's what we do here. It is verbose but valid and easy to verify
  // correctness.
  // TODO: print shorter output by combining adjacent runs, etc.
  int i = 0;
  ListSeparator LS(";");
  for (unsigned int cutoff : Opts.cutoffs) {
    if (cutoff > 0)
      OS << LS << "cutoffs[" << i << "]=" << cutoff;
    i++;
  }
  if (Opts.runtime_check)
    OS << LS << "runtime_check=" << Opts.runtime_check;

  OS << '>';
}
