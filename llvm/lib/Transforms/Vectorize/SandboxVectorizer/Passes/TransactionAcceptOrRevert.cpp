//===- TransactionAcceptOrRevert.cpp - Check cost and accept/revert region ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/TransactionAcceptOrRevert.h"
#include "../../VectorizeOptions.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/RegionWithScore.h"

namespace llvm {

namespace sandboxir {

bool TransactionAcceptOrRevert::runOnRegion(Region &Rgn, const Analyses &A) {
  const VectorizeOptions &Opts = VectorizeOptions::Global;
  const auto &SB = cast<RegionWithScore>(Rgn).getScoreboard();
  [[maybe_unused]] auto CostBefore = SB.getBeforeCost();
  [[maybe_unused]] auto CostAfter = SB.getAfterCost();
  InstructionCost CostAfterMinusBefore = SB.getAfterCost() - SB.getBeforeCost();
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Cost gain: " << CostAfterMinusBefore
                    << " (before/after/threshold: " << CostBefore << "/"
                    << CostAfter << "/" << Opts.sbvec_cost_threshold << ")\n");
  // TODO: Print costs / write to remarks.
  auto &Tracker = Rgn.getContext().getTracker();
  if (CostAfterMinusBefore < -Opts.sbvec_cost_threshold) {
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
