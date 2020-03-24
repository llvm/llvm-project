//===- LoopStripMine.h - Tapir loop stripmining -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINE_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"

namespace llvm {

class AssumptionCache;
class DominatorTree;
class Loop;
class LoopInfo;
class MDNode;
class OptimizationRemarkEmitter;
class ScalarEvolution;
class TargetLibraryInfo;
class TaskInfo;

using NewLoopsMap = SmallDenseMap<const Loop *, Loop *, 4>;

void simplifyLoopAfterStripMine(Loop *L, bool SimplifyIVs, LoopInfo *LI,
                                ScalarEvolution *SE, DominatorTree *DT,
                                AssumptionCache *AC);

TargetTransformInfo::StripMiningPreferences gatherStripMiningPreferences(
    Loop *L, ScalarEvolution &SE, const TargetTransformInfo &TTI,
    Optional<unsigned> UserCount);

bool computeStripMineCount(Loop *L, const TargetTransformInfo &TTI,
                           int64_t LoopCost,
                           TargetTransformInfo::StripMiningPreferences &UP);

Loop *StripMineLoop(
    Loop *L, unsigned Count, bool AllowExpensiveTripCount,
    bool UnrollRemainder, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT,
    AssumptionCache *AC, TaskInfo *TI, OptimizationRemarkEmitter *ORE,
    bool PreserveLCSSA, bool ParallelEpilog, bool NeedNestedSync);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINE_H
