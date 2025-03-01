//===- TransactionAcceptOrRevert.cpp - Check cost and accept/revert region ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/TransactionAcceptOrRevert.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"

namespace llvm {

static cl::opt<int> CostThreshold("sbvec-cost-threshold", cl::init(0),
                                  cl::Hidden,
                                  cl::desc("Vectorization cost threshold."));

namespace sandboxir {

bool TransactionAcceptOrRevert::runOnRegion(Region &Rgn, const Analyses &A) {
  const auto &SB = Rgn.getScoreboard();
  auto CostBefore = SB.getBeforeCost();
  auto CostAfter = SB.getAfterCost();
  InstructionCost CostAfterMinusBefore = SB.getAfterCost() - SB.getBeforeCost();
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Cost gain: " << CostAfterMinusBefore
                    << " (before/after/threshold: " << CostBefore << "/"
                    << CostAfter << "/" << CostThreshold << ")\n");
  // TODO: Print costs / write to remarks.
  auto &Tracker = Rgn.getContext().getTracker();
  if (CostAfterMinusBefore < -CostThreshold) {
    bool HasChanges = !Tracker.empty();
    Tracker.accept();
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "*** Transaction Accept ***\n");
    return HasChanges;
  }
  // Revert the IR.
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "*** Transaction Revert ***\n");
  Rgn.getContext().getTracker().revert();
  return false;
}

} // namespace sandboxir
} // namespace llvm
