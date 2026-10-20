//===- PruneUnprofitable.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark a SCoP as unfeasible if not deemed profitable to optimize.
//
//===----------------------------------------------------------------------===//

#include "polly/PruneUnprofitable.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace polly;

#include "polly/Support/PollyDebug.h"
#define DEBUG_TYPE "polly-prune-unprofitable"

namespace {

STATISTIC(ScopsProcessed,
          "Number of SCoPs considered for unprofitability pruning");
STATISTIC(ScopsPruned, "Number of pruned SCoPs because it they cannot be "
                       "optimized in a significant way");
STATISTIC(ScopsSurvived, "Number of SCoPs after pruning");

STATISTIC(NumPrunedLoops, "Number of pruned loops");
STATISTIC(NumPrunedBoxedLoops, "Number of pruned boxed loops");
STATISTIC(NumPrunedAffineLoops, "Number of pruned affine loops");

STATISTIC(NumLoopsInScop, "Number of loops in scops after pruning");
STATISTIC(NumBoxedLoops, "Number of boxed loops in SCoPs after pruning");
STATISTIC(NumAffineLoops, "Number of affine loops in SCoPs after pruning");

static void updateStatistics(Scop &S, bool Pruned) {
  Scop::ScopStatistics ScopStats = S.getStatistics();
  if (Pruned) {
    ScopsPruned++;
    NumPrunedLoops += ScopStats.NumAffineLoops + ScopStats.NumBoxedLoops;
    NumPrunedBoxedLoops += ScopStats.NumBoxedLoops;
    NumPrunedAffineLoops += ScopStats.NumAffineLoops;
  } else {
    ScopsSurvived++;
    NumLoopsInScop += ScopStats.NumAffineLoops + ScopStats.NumBoxedLoops;
    NumBoxedLoops += ScopStats.NumBoxedLoops;
    NumAffineLoops += ScopStats.NumAffineLoops;
  }
}
} // namespace

bool polly::runPruneUnprofitable(Scop &S) {
  if (PollyProcessUnprofitable) {
    POLLY_DEBUG(
        dbgs() << "NOTE: -polly-process-unprofitable active, won't prune "
                  "anything\n");
    return false;
  }

  ScopsProcessed++;

  if (!S.isProfitable(true)) {
    POLLY_DEBUG(
        dbgs() << "SCoP pruned because it probably cannot be optimized in "
                  "a significant way\n");
    S.invalidate(PROFITABLE, DebugLoc());
    updateStatistics(S, true);
  } else {
    updateStatistics(S, false);
  }

  return false;
}
