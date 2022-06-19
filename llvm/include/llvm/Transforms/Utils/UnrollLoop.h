//===- llvm/Transforms/Utils/UnrollLoop.h - Unrolling utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop unrolling utilities. It does not define any
// actual pass or policy, but provides a single function to perform loop
// unrolling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_UNROLLLOOP_H
#define LLVM_TRANSFORMS_UTILS_UNROLLLOOP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/InstructionCost.h"

namespace llvm {

class AssumptionCache;
class BasicBlock;
class BlockFrequencyInfo;
class DependenceInfo;
class DominatorTree;
class Loop;
class LoopInfo;
class MDNode;
class ProfileSummaryInfo;
class OptimizationRemarkEmitter;
class ScalarEvolution;
class StringRef;
class Value;

using NewLoopsMap = SmallDenseMap<const Loop *, Loop *, 4>;

/// @{
/// Metadata attribute names
const char *const LLVMLoopUnrollFollowupAll = "llvm.loop.unroll.followup_all";
const char *const LLVMLoopUnrollFollowupUnrolled =
    "llvm.loop.unroll.followup_unrolled";
const char *const LLVMLoopUnrollFollowupRemainder =
    "llvm.loop.unroll.followup_remainder";
/// @}

const Loop* addClonedBlockToLoopInfo(BasicBlock *OriginalBB,
                                     BasicBlock *ClonedBB, LoopInfo *LI,
                                     NewLoopsMap &NewLoops);

/// Represents the result of a \c UnrollLoop invocation.
enum class LoopUnrollResult {
  /// The loop was not modified.
  Unmodified,

  /// The loop was partially unrolled -- we still have a loop, but with a
  /// smaller trip count.  We may also have emitted epilogue loop if the loop
  /// had a non-constant trip count.
  PartiallyUnrolled,

  /// The loop was fully unrolled into straight-line code.  We no longer have
  /// any back-edges.
  FullyUnrolled
};

struct UnrollLoopOptions {
  unsigned Count;
  bool Force;
  bool Runtime;
  bool AllowExpensiveTripCount;
  bool UnrollRemainder;
  bool ForgetAllSCEV;
};

LoopUnrollResult UnrollLoop(Loop *L, UnrollLoopOptions ULO, LoopInfo *LI,
                            ScalarEvolution *SE, DominatorTree *DT,
                            AssumptionCache *AC,
                            const llvm::TargetTransformInfo *TTI,
                            OptimizationRemarkEmitter *ORE, bool PreserveLCSSA,
                            Loop **RemainderLoop = nullptr);

bool UnrollRuntimeLoopRemainder(
    Loop *L, unsigned Count, bool AllowExpensiveTripCount,
    bool UseEpilogRemainder, bool UnrollRemainder, bool ForgetAllSCEV,
    LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT, AssumptionCache *AC,
    const TargetTransformInfo *TTI, bool PreserveLCSSA,
    Loop **ResultLoop = nullptr);

LoopUnrollResult UnrollAndJamLoop(Loop *L, unsigned Count, unsigned TripCount,
                                  unsigned TripMultiple, bool UnrollRemainder,
                                  LoopInfo *LI, ScalarEvolution *SE,
                                  DominatorTree *DT, AssumptionCache *AC,
                                  const TargetTransformInfo *TTI,
                                  OptimizationRemarkEmitter *ORE,
                                  Loop **EpilogueLoop = nullptr);

bool isSafeToUnrollAndJam(Loop *L, ScalarEvolution &SE, DominatorTree &DT,
                          DependenceInfo &DI, LoopInfo &LI);

bool computeUnrollCount(Loop *L, const TargetTransformInfo &TTI,
                        DominatorTree &DT, LoopInfo *LI, ScalarEvolution &SE,
                        const SmallPtrSetImpl<const Value *> &EphValues,
                        OptimizationRemarkEmitter *ORE, unsigned TripCount,
                        unsigned MaxTripCount, bool MaxOrZero,
                        unsigned TripMultiple, unsigned LoopSize,
                        TargetTransformInfo::UnrollingPreferences &UP,
                        TargetTransformInfo::PeelingPreferences &PP,
                        bool &UseUpperBound);

void simplifyLoopAfterUnroll(Loop *L, bool SimplifyIVs, LoopInfo *LI,
                             ScalarEvolution *SE, DominatorTree *DT,
                             AssumptionCache *AC,
                             const TargetTransformInfo *TTI);

MDNode *GetUnrollMetadata(MDNode *LoopID, StringRef Name);

TargetTransformInfo::UnrollingPreferences gatherUnrollingPreferences(
    Loop *L, ScalarEvolution &SE, const TargetTransformInfo &TTI,
    BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
    llvm::OptimizationRemarkEmitter &ORE, int OptLevel,
    Optional<unsigned> UserThreshold, Optional<unsigned> UserCount,
    Optional<bool> UserAllowPartial, Optional<bool> UserRuntime,
    Optional<bool> UserUpperBound, Optional<unsigned> UserFullUnrollMaxCount);

InstructionCost ApproximateLoopSize(const Loop *L, unsigned &NumCalls,
    bool &NotDuplicatable, bool &Convergent, const TargetTransformInfo &TTI,
    const SmallPtrSetImpl<const Value *> &EphValues, unsigned BEInsns);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_UNROLLLOOP_H
