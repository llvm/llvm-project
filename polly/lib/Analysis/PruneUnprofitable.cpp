//===- PruneUnprofitable.cpp ------------------------------------*- C++ -*-===//
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
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#define DEBUG_TYPE "polly-prune-unprofitable"

using namespace polly;
using namespace llvm;

namespace {

STATISTIC(ScopsProcessed,
          "Number of SCoPs considered for unprofitability pruning");
STATISTIC(ScopsPruned, "Number of pruned SCoPs because it they cannot be "
                       "optimized in a significant way");

class PruneUnprofitable : public ScopPass {
private:
  PruneUnprofitable(const PruneUnprofitable &) = delete;
  const PruneUnprofitable &operator=(const PruneUnprofitable &) = delete;

public:
  static char ID;
  explicit PruneUnprofitable() : ScopPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScopInfoRegionPass>();
    AU.setPreservesAll();
  }

  virtual bool runOnScop(Scop &S) override {
    if (PollyProcessUnprofitable) {
      DEBUG(dbgs() << "NOTE: -polly-process-unprofitable active, won't prune "
                      "anything\n");
      return false;
    }

    ScopsProcessed++;

    if (!S.isProfitable(true)) {
      DEBUG(dbgs() << "SCoP pruned because it probably cannot be optimized in "
                      "a significant way\n");
      ScopsPruned++;
      S.invalidate(PROFITABLE, DebugLoc());
    }

    return false;
  }
};

char PruneUnprofitable::ID;
} // anonymous namespace

Pass *polly::createPruneUnprofitablePass() { return new PruneUnprofitable(); }

INITIALIZE_PASS_BEGIN(PruneUnprofitable, "polly-prune-unprofitable",
                      "Polly - Prune unprofitable SCoPs", false, false)
INITIALIZE_PASS_END(PruneUnprofitable, "polly-prune-unprofitable",
                    "Polly - Prune unprofitable SCoPs", false, false)
