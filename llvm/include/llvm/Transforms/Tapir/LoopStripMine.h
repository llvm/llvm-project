//===- llvm/Transforms/Tapir/LoopStripMine.h - Loop stripmining -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
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

Loop *StripMineLoop(
    Loop *L, unsigned Count, bool AllowExpensiveTripCount,
    bool UnrollRemainder, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT,
    AssumptionCache *AC, TaskInfo *TI, OptimizationRemarkEmitter *ORE,
    bool PreserveLCSSA, bool ParallelEpilog);

TargetTransformInfo::StripMiningPreferences gatherStripMiningPreferences(
    Loop *L, ScalarEvolution &SE, const TargetTransformInfo &TTI,
    Optional<unsigned> UserCount);

unsigned ApproximateLoopSize(const Loop *L, unsigned &NumCalls,
                             bool &NotDuplicatable, bool &Convergent,
                             bool &IsRecursive, bool &UnknownSize,
                             const TargetTransformInfo &TTI,
                             LoopInfo *LI, ScalarEvolution &SE,
                             const SmallPtrSetImpl<const Value *> &EphValues,
                             TargetLibraryInfo *TLI = nullptr);

bool computeStripMineCount(Loop *L, const TargetTransformInfo &TTI,
                           unsigned LoopSize,
                           TargetTransformInfo::StripMiningPreferences &UP);

MDNode *GetStripMineMetadata(MDNode *LoopID, StringRef Name);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINE_H
