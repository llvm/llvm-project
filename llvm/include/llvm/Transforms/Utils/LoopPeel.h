//===- llvm/Transforms/Utils/LoopPeel.h ----- Peeling utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop peeling utilities. It does not define any
// actual pass or policy.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPPEEL_H
#define LLVM_TRANSFORMS_UTILS_LOOPPEEL_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

bool canPeel(const Loop *L);

/// Returns true if the last iteration of \p L can be peeled off. It makes sure
/// the loop exit condition can be adjusted when peeling and that the loop
/// executes at least 2 iterations.
bool canPeelLastIteration(const Loop &L, ScalarEvolution &SE);

/// VMap is the value-map that maps instructions from the original loop to
/// instructions in the last peeled-off iteration. If \p PeelLast is true, peel
/// off the last \p PeelCount iterations from \p L (canPeelLastIteration must be
/// true for \p L), otherwise peel off the first \p PeelCount iterations.
bool peelLoop(Loop *L, unsigned PeelCount, bool PeelLast, LoopInfo *LI,
              ScalarEvolution *SE, DominatorTree &DT, AssumptionCache *AC,
              bool PreserveLCSSA, ValueToValueMapTy &VMap);

TargetTransformInfo::PeelingPreferences
gatherPeelingPreferences(Loop *L, ScalarEvolution &SE,
                         const TargetTransformInfo &TTI,
                         std::optional<bool> UserAllowPeeling,
                         std::optional<bool> UserAllowProfileBasedPeeling,
                         bool UnrollingSpecficValues = false);

void computePeelCount(Loop *L, unsigned LoopSize,
                      TargetTransformInfo::PeelingPreferences &PP,
                      unsigned TripCount, DominatorTree &DT,
                      ScalarEvolution &SE, const TargetTransformInfo &TTI,
                      AssumptionCache *AC = nullptr,
                      unsigned Threshold = UINT_MAX);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPPEEL_H
