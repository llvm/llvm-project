//===- llvm/Transforms/Utils/LoopUtils.h - Loop utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPUTILS_H
#define LLVM_TRANSFORMS_UTILS_LOOPUTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

template <typename T> class DomTreeNodeBase;
using DomTreeNode = DomTreeNodeBase<BasicBlock>;
class AliasSet;
class AliasSetTracker;
class BasicBlock;
class IRBuilderBase;
class Loop;
class LoopInfo;
class MemoryAccess;
class MemorySSA;
class MemorySSAUpdater;
class OptimizationRemarkEmitter;
class PredIteratorCache;
class ScalarEvolution;
class SCEV;
class SCEVExpander;
class TargetLibraryInfo;
class TargetTransformInfo;
class LPPassManager;
class Instruction;
struct RuntimeCheckingPtrGroup;
typedef std::pair<const RuntimeCheckingPtrGroup *,
                  const RuntimeCheckingPtrGroup *>
    RuntimePointerCheck;

template <typename T> class Optional;
template <typename T, unsigned N> class SmallSetVector;
template <typename T, unsigned N> class SmallVector;
template <typename T> class SmallVectorImpl;
template <typename T, unsigned N> class SmallPriorityWorklist;

BasicBlock *InsertPreheaderForLoop(Loop *L, DominatorTree *DT, LoopInfo *LI,
                                   MemorySSAUpdater *MSSAU, bool PreserveLCSSA);

/// Ensure that all exit blocks of the loop are dedicated exits.
///
/// For any loop exit block with non-loop predecessors, we split the loop
/// predecessors to use a dedicated loop exit block. We update the dominator
/// tree and loop info if provided, and will preserve LCSSA if requested.
bool formDedicatedExitBlocks(Loop *L, DominatorTree *DT, LoopInfo *LI,
                             MemorySSAUpdater *MSSAU, bool PreserveLCSSA);

/// Ensures LCSSA form for every instruction from the Worklist in the scope of
/// innermost containing loop.
///
/// For the given instruction which have uses outside of the loop, an LCSSA PHI
/// node is inserted and the uses outside the loop are rewritten to use this
/// node.
///
/// LoopInfo and DominatorTree are required and, since the routine makes no
/// changes to CFG, preserved.
///
/// Returns true if any modifications are made.
bool formLCSSAForInstructions(SmallVectorImpl<Instruction *> &Worklist,
                              const DominatorTree &DT, const LoopInfo &LI,
                              ScalarEvolution *SE);

/// Put loop into LCSSA form.
///
/// Looks at all instructions in the loop which have uses outside of the
/// current loop. For each, an LCSSA PHI node is inserted and the uses outside
/// the loop are rewritten to use this node. Sub-loops must be in LCSSA form
/// already.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSA(Loop &L, const DominatorTree &DT, const LoopInfo *LI,
               ScalarEvolution *SE);

/// Put a loop nest into LCSSA form.
///
/// This recursively forms LCSSA for a loop nest.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSARecursively(Loop &L, const DominatorTree &DT, const LoopInfo *LI,
                          ScalarEvolution *SE);

struct SinkAndHoistLICMFlags {
  bool NoOfMemAccTooLarge;
  unsigned LicmMssaOptCounter;
  unsigned LicmMssaOptCap;
  unsigned LicmMssaNoAccForPromotionCap;
  bool IsSink;
};

/// Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in
/// reverse depth first order w.r.t the DominatorTree. This allows us to visit
/// uses before definitions, allowing us to sink a loop body in one pass without
/// iteration. Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree,
/// TargetLibraryInfo, Loop, AliasSet information for all
/// instructions of the loop and loop safety information as
/// arguments. Diagnostics is emitted via \p ORE. It returns changed status.
bool sinkRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                TargetLibraryInfo *, TargetTransformInfo *, Loop *,
                AliasSetTracker *, MemorySSAUpdater *, ICFLoopSafetyInfo *,
                SinkAndHoistLICMFlags &, OptimizationRemarkEmitter *);

/// Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree.  This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
/// Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree,
/// TargetLibraryInfo, Loop, AliasSet information for all instructions of the
/// loop and loop safety information as arguments. Diagnostics is emitted via \p
/// ORE. It returns changed status.
bool hoistRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                 TargetLibraryInfo *, Loop *, AliasSetTracker *,
                 MemorySSAUpdater *, ScalarEvolution *, ICFLoopSafetyInfo *,
                 SinkAndHoistLICMFlags &, OptimizationRemarkEmitter *);

/// This function deletes dead loops. The caller of this function needs to
/// guarantee that the loop is infact dead.
/// The function requires a bunch or prerequisites to be present:
///   - The loop needs to be in LCSSA form
///   - The loop needs to have a Preheader
///   - A unique dedicated exit block must exist
///
/// This also updates the relevant analysis information in \p DT, \p SE, \p LI
/// and \p MSSA if pointers to those are provided.
/// It also updates the loop PM if an updater struct is provided.

void deleteDeadLoop(Loop *L, DominatorTree *DT, ScalarEvolution *SE,
                    LoopInfo *LI, MemorySSA *MSSA = nullptr);

/// Try to promote memory values to scalars by sinking stores out of
/// the loop and moving loads to before the loop.  We do this by looping over
/// the stores in the loop, looking for stores to Must pointers which are
/// loop invariant. It takes a set of must-alias values, Loop exit blocks
/// vector, loop exit blocks insertion point vector, PredIteratorCache,
/// LoopInfo, DominatorTree, Loop, AliasSet information for all instructions
/// of the loop and loop safety information as arguments.
/// Diagnostics is emitted via \p ORE. It returns changed status.
bool promoteLoopAccessesToScalars(
    const SmallSetVector<Value *, 8> &, SmallVectorImpl<BasicBlock *> &,
    SmallVectorImpl<Instruction *> &, SmallVectorImpl<MemoryAccess *> &,
    PredIteratorCache &, LoopInfo *, DominatorTree *, const TargetLibraryInfo *,
    Loop *, AliasSetTracker *, MemorySSAUpdater *, ICFLoopSafetyInfo *,
    OptimizationRemarkEmitter *);

/// Does a BFS from a given node to all of its children inside a given loop.
/// The returned vector of nodes includes the starting point.
SmallVector<DomTreeNode *, 16> collectChildrenInLoop(DomTreeNode *N,
                                                     const Loop *CurLoop);

/// Returns the instructions that use values defined in the loop.
SmallVector<Instruction *, 8> findDefsUsedOutsideOfLoop(Loop *L);

/// Find string metadata for loop
///
/// If it has a value (e.g. {"llvm.distribute", 1} return the value as an
/// operand or null otherwise.  If the string metadata is not found return
/// Optional's not-a-value.
Optional<const MDOperand *> findStringMetadataForLoop(const Loop *TheLoop,
                                                      StringRef Name);

/// Find named metadata for a loop with an integer value.
llvm::Optional<int> getOptionalIntLoopAttribute(Loop *TheLoop, StringRef Name);

/// Create a new loop identifier for a loop created from a loop transformation.
///
/// @param OrigLoopID The loop ID of the loop before the transformation.
/// @param FollowupAttrs List of attribute names that contain attributes to be
///                      added to the new loop ID.
/// @param InheritOptionsAttrsPrefix Selects which attributes should be inherited
///                                  from the original loop. The following values
///                                  are considered:
///        nullptr   : Inherit all attributes from @p OrigLoopID.
///        ""        : Do not inherit any attribute from @p OrigLoopID; only use
///                    those specified by a followup attribute.
///        "<prefix>": Inherit all attributes except those which start with
///                    <prefix>; commonly used to remove metadata for the
///                    applied transformation.
/// @param AlwaysNew If true, do not try to reuse OrigLoopID and never return
///                  None.
///
/// @return The loop ID for the after-transformation loop. The following values
///         can be returned:
///         None         : No followup attribute was found; it is up to the
///                        transformation to choose attributes that make sense.
///         @p OrigLoopID: The original identifier can be reused.
///         nullptr      : The new loop has no attributes.
///         MDNode*      : A new unique loop identifier.
Optional<MDNode *>
makeFollowupLoopID(MDNode *OrigLoopID, ArrayRef<StringRef> FollowupAttrs,
                   const char *InheritOptionsAttrsPrefix = "",
                   bool AlwaysNew = false);

/// Look for the loop attribute that disables all transformation heuristic.
bool hasDisableAllTransformsHint(const Loop *L);

/// Look for the loop attribute that disables the LICM transformation heuristics.
bool hasDisableLICMTransformsHint(const Loop *L);

/// The mode sets how eager a transformation should be applied.
enum TransformationMode {
  /// The pass can use heuristics to determine whether a transformation should
  /// be applied.
  TM_Unspecified,

  /// The transformation should be applied without considering a cost model.
  TM_Enable,

  /// The transformation should not be applied.
  TM_Disable,

  /// Force is a flag and should not be used alone.
  TM_Force = 0x04,

  /// The transformation was directed by the user, e.g. by a #pragma in
  /// the source code. If the transformation could not be applied, a
  /// warning should be emitted.
  TM_ForcedByUser = TM_Enable | TM_Force,

  /// The transformation must not be applied. For instance, `#pragma clang loop
  /// unroll(disable)` explicitly forbids any unrolling to take place. Unlike
  /// general loop metadata, it must not be dropped. Most passes should not
  /// behave differently under TM_Disable and TM_SuppressedByUser.
  TM_SuppressedByUser = TM_Disable | TM_Force
};

/// @{
/// Get the mode for LLVM's supported loop transformations.
TransformationMode hasUnrollTransformation(Loop *L);
TransformationMode hasUnrollAndJamTransformation(Loop *L);
TransformationMode hasVectorizeTransformation(Loop *L);
TransformationMode hasDistributeTransformation(Loop *L);
TransformationMode hasLICMVersioningTransformation(Loop *L);
/// @}

/// Set input string into loop metadata by keeping other values intact.
/// If the string is already in loop metadata update value if it is
/// different.
void addStringMetadataToLoop(Loop *TheLoop, const char *MDString,
                             unsigned V = 0);

/// Returns a loop's estimated trip count based on branch weight metadata.
/// In addition if \p EstimatedLoopInvocationWeight is not null it is
/// initialized with weight of loop's latch leading to the exit.
/// Returns 0 when the count is estimated to be 0, or None when a meaningful
/// estimate can not be made.
Optional<unsigned>
getLoopEstimatedTripCount(Loop *L,
                          unsigned *EstimatedLoopInvocationWeight = nullptr);

/// Set a loop's branch weight metadata to reflect that loop has \p
/// EstimatedTripCount iterations and \p EstimatedLoopInvocationWeight exits
/// through latch. Returns true if metadata is successfully updated, false
/// otherwise. Note that loop must have a latch block which controls loop exit
/// in order to succeed.
bool setLoopEstimatedTripCount(Loop *L, unsigned EstimatedTripCount,
                               unsigned EstimatedLoopInvocationWeight);

/// Check inner loop (L) backedge count is known to be invariant on all
/// iterations of its outer loop. If the loop has no parent, this is trivially
/// true.
bool hasIterationCountInvariantInParent(Loop *L, ScalarEvolution &SE);

/// Helper to consistently add the set of standard passes to a loop pass's \c
/// AnalysisUsage.
///
/// All loop passes should call this as part of implementing their \c
/// getAnalysisUsage.
void getLoopAnalysisUsage(AnalysisUsage &AU);

/// Returns true if is legal to hoist or sink this instruction disregarding the
/// possible introduction of faults.  Reasoning about potential faulting
/// instructions is the responsibility of the caller since it is challenging to
/// do efficiently from within this routine.
/// \p TargetExecutesOncePerLoop is true only when it is guaranteed that the
/// target executes at most once per execution of the loop body.  This is used
/// to assess the legality of duplicating atomic loads.  Generally, this is
/// true when moving out of loop and not true when moving into loops.
/// If \p ORE is set use it to emit optimization remarks.
bool canSinkOrHoistInst(Instruction &I, AAResults *AA, DominatorTree *DT,
                        Loop *CurLoop, AliasSetTracker *CurAST,
                        MemorySSAUpdater *MSSAU, bool TargetExecutesOncePerLoop,
                        SinkAndHoistLICMFlags *LICMFlags = nullptr,
                        OptimizationRemarkEmitter *ORE = nullptr);

/// Returns a Min/Max operation corresponding to MinMaxRecurrenceKind.
Value *createMinMaxOp(IRBuilderBase &Builder,
                      RecurrenceDescriptor::MinMaxRecurrenceKind RK,
                      Value *Left, Value *Right);

/// Generates an ordered vector reduction using extracts to reduce the value.
Value *
getOrderedReduction(IRBuilderBase &Builder, Value *Acc, Value *Src, unsigned Op,
                    RecurrenceDescriptor::MinMaxRecurrenceKind MinMaxKind =
                        RecurrenceDescriptor::MRK_Invalid,
                    ArrayRef<Value *> RedOps = None);

/// Generates a vector reduction using shufflevectors to reduce the value.
/// Fast-math-flags are propagated using the IRBuilder's setting.
Value *getShuffleReduction(IRBuilderBase &Builder, Value *Src, unsigned Op,
                           RecurrenceDescriptor::MinMaxRecurrenceKind
                               MinMaxKind = RecurrenceDescriptor::MRK_Invalid,
                           ArrayRef<Value *> RedOps = None);

/// Create a target reduction of the given vector. The reduction operation
/// is described by the \p Opcode parameter. min/max reductions require
/// additional information supplied in \p Flags.
/// The target is queried to determine if intrinsics or shuffle sequences are
/// required to implement the reduction.
/// Fast-math-flags are propagated using the IRBuilder's setting.
Value *createSimpleTargetReduction(IRBuilderBase &B,
                                   const TargetTransformInfo *TTI,
                                   unsigned Opcode, Value *Src,
                                   TargetTransformInfo::ReductionFlags Flags =
                                       TargetTransformInfo::ReductionFlags(),
                                   ArrayRef<Value *> RedOps = None);

/// Create a generic target reduction using a recurrence descriptor \p Desc
/// The target is queried to determine if intrinsics or shuffle sequences are
/// required to implement the reduction.
/// Fast-math-flags are propagated using the RecurrenceDescriptor.
Value *createTargetReduction(IRBuilderBase &B, const TargetTransformInfo *TTI,
                             RecurrenceDescriptor &Desc, Value *Src,
                             bool NoNaN = false);

/// Get the intersection (logical and) of all of the potential IR flags
/// of each scalar operation (VL) that will be converted into a vector (I).
/// If OpValue is non-null, we only consider operations similar to OpValue
/// when intersecting.
/// Flag set: NSW, NUW, exact, and all of fast-math.
void propagateIRFlags(Value *I, ArrayRef<Value *> VL, Value *OpValue = nullptr);

/// Returns true if we can prove that \p S is defined and always negative in
/// loop \p L.
bool isKnownNegativeInLoop(const SCEV *S, const Loop *L, ScalarEvolution &SE);

/// Returns true if we can prove that \p S is defined and always non-negative in
/// loop \p L.
bool isKnownNonNegativeInLoop(const SCEV *S, const Loop *L,
                              ScalarEvolution &SE);

/// Returns true if \p S is defined and never is equal to signed/unsigned max.
bool cannotBeMaxInLoop(const SCEV *S, const Loop *L, ScalarEvolution &SE,
                       bool Signed);

/// Returns true if \p S is defined and never is equal to signed/unsigned min.
bool cannotBeMinInLoop(const SCEV *S, const Loop *L, ScalarEvolution &SE,
                       bool Signed);

enum ReplaceExitVal { NeverRepl, OnlyCheapRepl, NoHardUse, AlwaysRepl };

/// If the final value of any expressions that are recurrent in the loop can
/// be computed, substitute the exit values from the loop into any instructions
/// outside of the loop that use the final values of the current expressions.
/// Return the number of loop exit values that have been replaced, and the
/// corresponding phi node will be added to DeadInsts.
int rewriteLoopExitValues(Loop *L, LoopInfo *LI, TargetLibraryInfo *TLI,
                          ScalarEvolution *SE, const TargetTransformInfo *TTI,
                          SCEVExpander &Rewriter, DominatorTree *DT,
                          ReplaceExitVal ReplaceExitValue,
                          SmallVector<WeakTrackingVH, 16> &DeadInsts);

/// Set weights for \p UnrolledLoop and \p RemainderLoop based on weights for
/// \p OrigLoop and the following distribution of \p OrigLoop iteration among \p
/// UnrolledLoop and \p RemainderLoop. \p UnrolledLoop receives weights that
/// reflect TC/UF iterations, and \p RemainderLoop receives weights that reflect
/// the remaining TC%UF iterations.
///
/// Note that \p OrigLoop may be equal to either \p UnrolledLoop or \p
/// RemainderLoop in which case weights for \p OrigLoop are updated accordingly.
/// Note also behavior is undefined if \p UnrolledLoop and \p RemainderLoop are
/// equal. \p UF must be greater than zero.
/// If \p OrigLoop has no profile info associated nothing happens.
///
/// This utility may be useful for such optimizations as unroller and
/// vectorizer as it's typical transformation for them.
void setProfileInfoAfterUnrolling(Loop *OrigLoop, Loop *UnrolledLoop,
                                  Loop *RemainderLoop, uint64_t UF);

/// Utility that implements appending of loops onto a worklist given a range.
/// We want to process loops in postorder, but the worklist is a LIFO data
/// structure, so we append to it in *reverse* postorder.
/// For trees, a preorder traversal is a viable reverse postorder, so we
/// actually append using a preorder walk algorithm.
template <typename RangeT>
void appendLoopsToWorklist(RangeT &&, SmallPriorityWorklist<Loop *, 4> &);
/// Utility that implements appending of loops onto a worklist given a range.
/// It has the same behavior as appendLoopsToWorklist, but assumes the range of
/// loops has already been reversed, so it processes loops in the given order.
template <typename RangeT>
void appendReversedLoopsToWorklist(RangeT &&,
                                   SmallPriorityWorklist<Loop *, 4> &);

/// Utility that implements appending of loops onto a worklist given LoopInfo.
/// Calls the templated utility taking a Range of loops, handing it the Loops
/// in LoopInfo, iterated in reverse. This is because the loops are stored in
/// RPO w.r.t. the control flow graph in LoopInfo. For the purpose of unrolling,
/// loop deletion, and LICM, we largely want to work forward across the CFG so
/// that we visit defs before uses and can propagate simplifications from one
/// loop nest into the next. Calls appendReversedLoopsToWorklist with the
/// already reversed loops in LI.
/// FIXME: Consider changing the order in LoopInfo.
void appendLoopsToWorklist(LoopInfo &, SmallPriorityWorklist<Loop *, 4> &);

/// Recursively clone the specified loop and all of its children,
/// mapping the blocks with the specified map.
Loop *cloneLoop(Loop *L, Loop *PL, ValueToValueMapTy &VM,
                LoopInfo *LI, LPPassManager *LPM);

/// Add code that checks at runtime if the accessed arrays in \p PointerChecks
/// overlap.
///
/// Returns a pair of instructions where the first element is the first
/// instruction generated in possibly a sequence of instructions and the
/// second value is the final comparator value or NULL if no check is needed.
std::pair<Instruction *, Instruction *>
addRuntimeChecks(Instruction *Loc, Loop *TheLoop,
                 const SmallVectorImpl<RuntimePointerCheck> &PointerChecks,
                 ScalarEvolution *SE);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPUTILS_H
