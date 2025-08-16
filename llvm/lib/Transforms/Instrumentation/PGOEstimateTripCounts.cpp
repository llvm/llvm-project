//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/PGOEstimateTripCounts.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "pgo-estimate-trip-counts"

static bool runOnLoop(Loop *L) {
  bool MadeChange = false;
  std::optional<unsigned> TC = getLoopEstimatedTripCount(
      L, /*EstimatedLoopInvocationWeight=*/nullptr, /*DbgForInit=*/true);
  MadeChange |= setLoopEstimatedTripCount(L, TC);
  for (Loop *SL : *L)
    MadeChange |= runOnLoop(SL);
  return MadeChange;
}

PreservedAnalyses PGOEstimateTripCountsPass::run(Function &F,
                                                 FunctionAnalysisManager &FAM) {
  bool MadeChange = false;
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": start\n");
  LoopInfo *LI = &FAM.getResult<LoopAnalysis>(F);
  if (!LI)
    return PreservedAnalyses::all();
  for (Loop *L : *LI)
    MadeChange |= runOnLoop(L);
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": end\n");
  if (MadeChange) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<LazyCallGraphAnalysis>();
    return PA;
  }
  return PreservedAnalyses::all();
}
