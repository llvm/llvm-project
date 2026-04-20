//===- IncrementalUpdateProfileAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a wrapper of BFI and BPI that, on invalidation, optionally verifies
// (which is time-consuming) that the BFI and BPI are equivalent to newly
// created instances.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IncrementalUpdateProfileAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
static cl::opt<bool> VerifyIncrementalProfileUpdate(
    "verify-incremental-profile-update",
    cl::desc("Verify that incremental profile updates match a recalculation"),
#ifdef EXPENSIVE_CHECKS
    cl::init(true),
#else
    cl::init(false),
#endif
    cl::Hidden);

AnalysisKey IncrementalUpdateProfileAnalysis::Key;

IncrementalProfDataVerifier::IncrementalProfDataVerifier(
    Function &F, FunctionAnalysisManager &FAM)
    : BPI(FAM.getResult<BranchProbabilityAnalysis>(F)),
      BFI(FAM.getResult<BlockFrequencyAnalysis>(F)), F(F), FAM(FAM) {}

std::unique_ptr<IncrementalProfDataVerifier>
IncrementalProfDataVerifier::create(Function &F, FunctionAnalysisManager &FAM) {
  if (auto EC = F.getEntryCount(); !EC || !EC->getCount())
    return nullptr;
  return std::unique_ptr<IncrementalProfDataVerifier>(
      new IncrementalProfDataVerifier(F, FAM));
}

void IncrementalProfDataVerifier::verify() {
  auto &Ctx = F.getContext();
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  BranchProbabilityInfo NewBPI(F, LI, &FAM.getResult<TargetLibraryAnalysis>(F),
                               &FAM.getResult<DominatorTreeAnalysis>(F),
                               &FAM.getResult<PostDominatorTreeAnalysis>(F));
  BlockFrequencyInfo NewBFI(F, NewBPI, LI);
  for (auto &BB : F) {
    if (BFI.getBlockFreq(&BB) != NewBFI.getBlockFreq(&BB))
      Ctx.emitError("Incremental profile info failure for " + BB.getName() +
                    " in " + F.getName());
    if (succ_size(&BB) < 2)
      continue;
    for (auto *Dest : successors(&BB))
      if (BPI.getEdgeProbability(&BB, Dest) !=
          NewBPI.getEdgeProbability(&BB, Dest))
        Ctx.emitError("Incremental profile info failure for " + BB.getName() +
                      " -> " + Dest->getName() + " in " + F.getName());
  }
}
