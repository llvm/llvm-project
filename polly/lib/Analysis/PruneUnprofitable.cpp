//===- PruneUnprofitable.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Mark a SCoP as unfeasible if not deemed profitable to optimize.
//
//===----------------------------------------------------------------------===//

#include "polly/PruneUnprofitable.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace polly;

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

class PruneUnprofitable : public ScopPass {
private:
  void updateStatistics(Scop &S, bool Pruned) {
    auto ScopStats = S.getStatistics();
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

public:
  static char ID;

  explicit PruneUnprofitable() : ScopPass(ID) {}
  PruneUnprofitable(const PruneUnprofitable &) = delete;
  PruneUnprofitable &operator=(const PruneUnprofitable &) = delete;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScopInfoRegionPass>();
    AU.setPreservesAll();
  }

  bool runOnScop(Scop &S) override {
    if (PollyProcessUnprofitable) {
      LLVM_DEBUG(
          dbgs() << "NOTE: -polly-process-unprofitable active, won't prune "
                    "anything\n");
      return false;
    }

    ScopsProcessed++;

    if (!S.isProfitable(true)) {
      LLVM_DEBUG(
          dbgs() << "SCoP pruned because it probably cannot be optimized in "
                    "a significant way\n");
      S.invalidate(PROFITABLE, DebugLoc());
      updateStatistics(S, true);
    } else {
      updateStatistics(S, false);
    }

    return false;
  }
};
} // namespace

char PruneUnprofitable::ID;

Pass *polly::createPruneUnprofitablePass() { return new PruneUnprofitable(); }

INITIALIZE_PASS_BEGIN(PruneUnprofitable, "polly-prune-unprofitable",
                      "Polly - Prune unprofitable SCoPs", false, false)
INITIALIZE_PASS_END(PruneUnprofitable, "polly-prune-unprofitable",
                    "Polly - Prune unprofitable SCoPs", false, false)
