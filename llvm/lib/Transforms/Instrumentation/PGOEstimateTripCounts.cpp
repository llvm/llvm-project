//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/PGOEstimateTripCounts.h"
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

PreservedAnalyses PGOEstimateTripCountsPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool MadeChange = false;
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": start\n");
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    LoopInfo *LI = &FAM.getResult<LoopAnalysis>(F);
    if (!LI)
      continue;
    for (Loop *L : *LI)
      MadeChange |= runOnLoop(L);
  }
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": end\n");
  return MadeChange ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
