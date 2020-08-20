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

namespace llvm {

bool canPeel(Loop *L);

bool peelLoop(Loop *L, unsigned PeelCount, LoopInfo *LI, ScalarEvolution *SE,
              DominatorTree *DT, AssumptionCache *AC, bool PreserveLCSSA);

TargetTransformInfo::PeelingPreferences
gatherPeelingPreferences(Loop *L, ScalarEvolution &SE,
                         const TargetTransformInfo &TTI,
                         Optional<bool> UserAllowPeeling,
                         Optional<bool> UserAllowProfileBasedPeeling,
                         bool UnrollingSpecficValues = false);

void computePeelCount(Loop *L, unsigned LoopSize,
                      TargetTransformInfo::PeelingPreferences &PP,
                      unsigned &TripCount, ScalarEvolution &SE,
                      unsigned Threshold = UINT_MAX);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPPEEL_H
