//===- SLPTree.h - SLP vectorization graph (BoUpSLP) ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header declaring the SLP vectorization graph BoUpSLP and its
// nested types. Method definitions live in SLPVectorizer.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTREE_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTREE_H

#include "SLPCompatibilityAnalysis.h"
#include "SLPCostAnalysis.h"
#include "SLPMemoryUtils.h"
#include "SLPReductionUtils.h"
#include "SLPShuffleAnalysis.h"
#include "SLPTypeUtils.h"
#include "SLPUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace llvm::slpvectorizer {

/// If the ScheduleRegionSizeBudget is exhausted, we allow small scheduling
/// regions to be handled.
static const int MinScheduleRegionSize = 16;

/// A vectorized part of a split reduction, combined into the final reduction
/// result by the horizontal reduction emitter.
struct ReductionVectorPart {
  /// The vectorized value, tracked in case it is replaced while other parts
  /// are vectorized.
  WeakTrackingVH Vec;
  /// The number of times each lane is repeated in the reduction (emitted as a
  /// multiplication by the scale for add/fadd reductions).
  unsigned Scale = 1;
  /// Signedness of \p Vec for reductions, operating on truncated types.
  bool IsSigned = false;
  /// True if the value was already reduced in-tree.
  bool ReducedInTree = false;
  /// True if the part contribution is subtracted from (rather than added to)
  /// the final reduction result. Used for reassociated fadd reductions,
  /// flattened through fsub/fneg operations.
  bool Negated = false;
};

} // namespace llvm::slpvectorizer

namespace llvm {

/// Bottom Up SLP Vectorizer.
class slpvectorizer::BoUpSLP {
  class TreeEntry;
  class ScheduleEntity;
  class ScheduleData;
  class ScheduleCopyableData;
  class ScheduleBundle;
  class ShuffleCostEstimator;
  class ShuffleInstructionBuilder;

public:
  /// If we decide to generate strided load / store, this struct contains all
  /// the necessary info. It's fields are calculated by analyzeRtStrideCandidate
  /// and analyzeConstantStrideCandidate. Note that Stride can be given either
  /// as a SCEV or as a Value if it already exists. To get the stride in bytes,
  /// StrideVal (or value obtained from StrideSCEV) has to by multiplied by the
  /// size of element of FixedVectorType.
  struct StridedPtrInfo {
    Value *StrideVal = nullptr;
    const SCEV *StrideSCEV = nullptr;
    FixedVectorType *Ty = nullptr;
  };

  /// Tracks the state we can represent the loads in the given sequence.
  enum class LoadsState {
    Gather,
    Vectorize,
    ScatterVectorize,
    StridedVectorize,
    CompressVectorize,
    BlendedLoadVectorize
  };

  using ValueList = SmallVector<Value *, 8>;
  using InstrList = SmallVector<Instruction *, 16>;
  using ValueSet = SmallPtrSet<Value *, 16>;
  using StoreList = SmallVector<StoreInst *, 8>;
  using ExtraValueToDebugLocsMap = SmallDenseSet<Value *, 4>;
  using OrdersType = SmallVector<unsigned, 4>;

  BoUpSLP(Function *Func, ScalarEvolution *Se, TargetTransformInfo *Tti,
          TargetLibraryInfo *TLi, AAResults *Aa, LoopInfo *Li,
          DominatorTree *Dt, AssumptionCache *AC, DemandedBits *DB,
          const DataLayout *DL, OptimizationRemarkEmitter *ORE);

  /// Vectorize the tree that starts with the elements in \p VL.
  /// Returns the vectorized root.
  Value *vectorizeTree();

  /// Vectorize the tree but with the list of externally used values \p
  /// ExternallyUsedValues. Values in this MapVector can be replaced but the
  /// generated extractvalue instructions.
  Value *
  vectorizeTree(const ExtraValueToDebugLocsMap &ExternallyUsedValues,
                Instruction *ReductionRoot = nullptr,
                ArrayRef<ReductionVectorPart> VectorValuesAndScales = {});

  /// \returns the cost incurred by unwanted spills and fills, caused by
  /// holding live values over call sites.
  InstructionCost getSpillCost();

  TargetTransformInfo::TargetCostKind getCostKind() const { return CostKind; }

  /// Calculates the cost of the subtrees, trims non-profitable ones and returns
  /// final cost.
  InstructionCost
  calculateTreeCostAndTrimNonProfitable(ArrayRef<Value *> VectorizedVals = {},
                                        Instruction *RdxRoot = nullptr);

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  InstructionCost getTreeCost(InstructionCost TreeCost,
                              ArrayRef<Value *> VectorizedVals = {},
                              InstructionCost ReductionCost = TTI::TCC_Free,
                              Instruction *RdxRoot = nullptr);

  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst.
  void buildTree(ArrayRef<Value *> Roots,
                 const SmallDenseSet<Value *> &UserIgnoreLst);

  /// Construct a vectorizable tree that starts at \p Roots.
  void buildTree(ArrayRef<Value *> Roots);

  /// Sets the narrowed reduction chain instructions, dropped together with
  /// the reduction.
  void setNarrowedChainInsts(ArrayRef<Instruction *> Insts) {
    NarrowedChainInsts.insert(Insts.begin(), Insts.end());
  }

  /// Returns true if the last buildTree() observed a may-alias memory
  /// dependency between two distinct, range-checkable base objects, i.e. a
  /// dependency that could be turned into a runtime alias check.
  bool hasRuntimeCheckableBlockers() const {
    return HasRuntimeCheckableBlockers;
  }

  /// Records whether a may-alias dependency between distinct, range-checkable
  /// base objects has been observed, so the caller can decide to retry with
  /// runtime alias checks enabled.
  void setHasRuntimeCheckableBlockers(bool V) {
    HasRuntimeCheckableBlockers = V;
  }

  /// Returns true if the last buildTree() kept a may-alias memory dependency
  /// that is not runtime-checkable (call or a non-simple mem access). Such a
  /// dependency cannot be dropped, so a runtime-checks retry cannot unblock the
  /// region and would be pure overhead.
  bool hasNonCheckableMemBlocker() const { return HasNonCheckableMemBlocker; }

  /// Records that a non-runtime-checkable may-alias dependency was kept.
  void setHasNonCheckableMemBlocker(bool V) { HasNonCheckableMemBlocker = V; }

  /// Returns true if the current vectorization attempt may drop
  /// runtime-checkable may-alias dependencies and guard the region with
  /// runtime alias checks.
  bool isTryingRuntimeAliasChecks() const { return TryRuntimeAliasChecks; }

  /// Enables or disables dropping runtime-checkable may-alias dependencies in
  /// favor of runtime alias checks for the current vectorization attempt.
  void setTryRuntimeAliasChecks(bool V) { TryRuntimeAliasChecks = V; }

  /// Resets the runtime alias check data.
  void resetRuntimeAliasCheckState() {
    HasRuntimeCheckableBlockers = false;
    HasNonCheckableMemBlocker = false;
    RTChecksFinalized = false;
    RTChecks.clear();
    RTOrigBodyOrder.clear();
  }

  /// Snapshots RTChecks.BB's body (non-PHI, non-terminator) into
  /// RTOrigBodyOrder in program order, for the scalar fallback.
  void captureRuntimeCheckBodySnapshot();

  /// Returns true if \p BB satisfies the block-level preconditions for runtime
  /// alias check versioning (straight-line, outside any loop, duplicable, not a
  /// scalar fallback, function not optimized for size). These checks do not
  /// depend on the collected checks, so they can gate the (expensive)
  /// optimistic retry before any tree is rebuilt.
  bool canVersionBlockForRuntimeChecks(BasicBlock *BB) const;

  /// Returns true if the runtime alias checks can be safely emitted to guard
  /// the vectorized region.
  bool canVersionForRuntimeChecks();

  /// Returns true if \p BB is a scalar fallback block created by runtime alias
  /// check versioning.
  bool isScalarFallbackBlock(BasicBlock *BB) const {
    return ScalarFallbackBlocks.contains(BB);
  }

  /// Returns true if an optimistic runtime-checks versioning attempt already
  /// failed for \p BB, so further retries in the same block can be skipped.
  bool runtimeChecksFailedForBlock(BasicBlock *BB) const {
    return FailedRuntimeChecksBlocks.contains(BB);
  }

  /// Records that an optimistic runtime-checks versioning attempt failed for
  /// \p BB.
  void markRuntimeChecksFailedForBlock(BasicBlock *BB) {
    FailedRuntimeChecksBlocks.insert(BB);
  }

  /// Returns the modeled cost of the runtime alias checks collected during the
  /// last (optimistic) buildTree().
  InstructionCost getRuntimeChecksCost() const;

  /// Returns true if the last (optimistic) buildTree() collected any runtime
  /// alias checks that must guard the vectorized region.
  bool hasRuntimeAliasChecks() const { return !RTChecks.BasePairs.empty(); }

  /// Returns true if vectorization changed the CFG (i.e. a block was versioned
  /// with runtime alias checks). When true, CFG analyses must not be preserved.
  bool isCFGChanged() const { return CFGChanged; }

  TreeEntry &getRootNode() {
    assert(!VectorizableTree.empty() && "No graph to get the first node from");
    return *VectorizableTree.front();
  }

  const TreeEntry &getRootNode() const {
    assert(!VectorizableTree.empty() && "No graph to get the first node from");
    return *VectorizableTree.front();
  }

  /// Returns the scalars of the root node.
  ArrayRef<Value *> getRootNodeScalars() const { return getRootNode().Scalars; }

  /// Returns the lane the given value is vectorized to in the root node.
  unsigned findRootLaneForValue(Value *V) const {
    return getRootNode().findLaneForValue(V);
  }

  /// Returns the type/is-signed info for the root node in the graph without
  /// casting.
  std::optional<std::pair<Type *, bool>> getRootNodeTypeWithNoCast() const {
    const TreeEntry &Root = getRootNode();
    if (Root.State != TreeEntry::Vectorize || Root.isAltShuffle() ||
        !Root.Scalars.front()->getType()->isIntegerTy())
      return std::nullopt;
    auto It = MinBWs.find(&Root);
    if (It != MinBWs.end())
      return std::make_pair(IntegerType::get(Root.Scalars.front()->getContext(),
                                             It->second.first),
                            It->second.second);
    if (Root.getOpcode() == Instruction::ZExt ||
        Root.getOpcode() == Instruction::SExt)
      return std::make_pair(cast<CastInst>(Root.getMainOp())->getSrcTy(),
                            Root.getOpcode() == Instruction::SExt);
    return std::nullopt;
  }

  /// Checks if the root graph node can be emitted with narrower bitwidth at
  /// codegen and returns it signedness, if so.
  bool isSignedMinBitwidthRootNode() const {
    return MinBWs.at(&getRootNode()).second;
  }

  /// Returns reduction type after minbitdth analysis.
  FixedVectorType *getReductionType() const {
    if (ReductionBitWidth == 0 ||
        !getRootNodeScalars().front()->getType()->isIntegerTy() ||
        ReductionBitWidth >=
            DL->getTypeSizeInBits(getRootNodeScalars().front()->getType()))
      return cast<FixedVectorType>(
          getWidenedType(getRootNodeScalars().front()->getType(),
                         getRootNode().getVectorFactor()));
    return cast<FixedVectorType>(getWidenedType(
        IntegerType::get(getRootNodeScalars().front()->getContext(),
                         ReductionBitWidth),
        getRootNode().getVectorFactor()));
  }

  /// Returns true if the tree results in one of the reduced bitcasts variants.
  bool isReducedBitcastRoot() const {
    return getRootNode().hasState() &&
           (getRootNode().CombinedOp == TreeEntry::ReducedBitcast ||
            getRootNode().CombinedOp == TreeEntry::ReducedBitcastBSwap ||
            getRootNode().CombinedOp == TreeEntry::ReducedBitcastLoads ||
            getRootNode().CombinedOp == TreeEntry::ReducedBitcastBSwapLoads) &&
           getRootNode().State == TreeEntry::Vectorize;
  }

  /// Returns true if the tree results in the reduced cmp bitcast root.
  bool isReducedCmpBitcastRoot() const {
    return getRootNode().hasState() &&
           getRootNode().CombinedOp == TreeEntry::ReducedCmpBitcast &&
           getRootNode().State == TreeEntry::Vectorize;
  }

  /// Returns true if the tree is a reduction tree.
  bool isReductionTree() const { return UserIgnoreList != nullptr; }

  /// Builds external uses of the vectorized scalars, i.e. the list of
  /// vectorized scalars to be extracted, their lanes and their scalar users. \p
  /// ExternallyUsedValues contains additional list of external uses to handle
  /// vectorization of reductions.
  void
  buildExternalUses(const ExtraValueToDebugLocsMap &ExternallyUsedValues = {});

  /// Transforms graph nodes to target specific representations, if profitable.
  void transformNodes();

  /// Clear the internal data structures that are created by 'buildTree'.
  void deleteTree() {
    VectorizableTree.clear();
    ScalarToTreeEntries.clear();
    DeletedNodes.clear();
    TransformedToGatherNodes.clear();
    OperandsToTreeEntry.clear();
    ScalarsInSplitNodes.clear();
    MustGather.clear();
    ReassocScalarToTreeEntries.clear();
    KeptReassocScalars.clear();
    NonScheduledFirst.clear();
    EntryToLastInstruction.clear();
    LastInstructionToPos.clear();
    LoadEntriesToVectorize.clear();
    IsGraphTransformMode = false;
    GatheredLoadsEntriesFirst.reset();
    SplatGatheredScalarsRoots.clear();
    NumCanonicalSplatSubtreeEntries = 0;
    CompressEntryToData.clear();
    ExternalUses.clear();
    ExternalUsesAsOriginalScalar.clear();
    ExternalUsesWithNonUsers.clear();
    ExternalUseReplacements.clear();
    RTChecks.clear();
    HasRuntimeCheckableBlockers = false;
    HasNonCheckableMemBlocker = false;
    RTChecksFinalized = false;
    for (auto &Iter : BlocksSchedules) {
      BlockScheduling *BS = Iter.second.get();
      BS->clear();
    }
    MinBWs.clear();
    ReductionBitWidth = 0;
    BaseGraphSize = 1;
    CastMaxMinBWSizes.reset();
    ExtraBitWidthNodes.clear();
    InstrElementSize.clear();
    UserIgnoreList = nullptr;
    NarrowedChainInsts.clear();
    PostponedGathers.clear();
    ValueToGatherNodes.clear();
    TreeEntryToStridedPtrInfoMap.clear();
    CurrentLoopNest.clear();
    MergedLoopBTCs.clear();
  }

  unsigned getTreeSize() const { return VectorizableTree.size(); }

  /// Returns the base graph size, before any transformations.
  unsigned getCanonicalGraphSize() const { return BaseGraphSize; }

  /// Number of tree entries that form the splat gather subtrees.
  unsigned getNumSplatSubtreeEntries() const {
    return NumCanonicalSplatSubtreeEntries;
  }

  /// Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();

  /// Does this non-empty order represent an identity order?  Identity
  /// should be represented as an empty order, so this is used to
  /// decide if we can canonicalize a computed order.  Undef elements
  /// (represented as size) are ignored.
  static bool isIdentityOrder(ArrayRef<unsigned> Order) {
    assert(!Order.empty() && "expected non-empty order");
    const unsigned Sz = Order.size();
    return all_of(enumerate(Order), [&](const auto &P) {
      return P.value() == P.index() || P.value() == Sz;
    });
  }

  /// Checks if the specified gather tree entry \p TE can be represented as a
  /// shuffled vector entry + (possibly) permutation with other gathers. It
  /// implements the checks only for possibly ordered scalars (Loads,
  /// ExtractElement, ExtractValue), which can be part of the graph.
  /// \param TopToBottom If true, used for the whole tree rotation, false - for
  /// sub-tree rotations. \param IgnoreReorder true, if the order of the root
  /// node might be ignored.
  std::optional<OrdersType> findReusedOrderedScalars(const TreeEntry &TE,
                                                     bool TopToBottom,
                                                     bool IgnoreReorder);

  /// Sort loads into increasing pointers offsets to allow greater clustering.
  std::optional<OrdersType> findPartiallyOrderedLoads(const TreeEntry &TE);

  /// Gets reordering data for the given tree entry. If the entry is vectorized
  /// - just return ReorderIndices, otherwise check if the scalars can be
  /// reordered and return the most optimal order.
  /// \return std::nullopt if ordering is not important, empty order, if
  /// identity order is important, or the actual order.
  /// \param TopToBottom If true, include the order of vectorized stores and
  /// insertelement nodes, otherwise skip them.
  /// \param IgnoreReorder true, if the root node order can be ignored.
  std::optional<OrdersType>
  getReorderingData(const TreeEntry &TE, bool TopToBottom, bool IgnoreReorder);

  /// Checks if it is profitable to reorder the current tree.
  /// If the tree does not contain many profitable reordable nodes, better to
  /// skip it to save compile time.
  bool isProfitableToReorder() const;

  /// Reorders the current graph to the most profitable order starting from the
  /// root node to the leaf nodes. The best order is chosen only from the nodes
  /// of the same size (vectorization factor). Smaller nodes are considered
  /// parts of subgraph with smaller VF and they are reordered independently. We
  /// can make it because we still need to extend smaller nodes to the wider VF
  /// and we can merge reordering shuffles with the widening shuffles.
  void reorderTopToBottom();

  /// Reorders the current graph to the most profitable order starting from
  /// leaves to the root. It allows to rotate small subgraphs and reduce the
  /// number of reshuffles if the leaf nodes use the same order. In this case we
  /// can merge the orders and just shuffle user node instead of shuffling its
  /// operands. Plus, even the leaf nodes have different orders, it allows to
  /// sink reordering in the graph closer to the root node and merge it later
  /// during analysis.
  void reorderBottomToTop(bool IgnoreReorder = false);

  /// Marks the schedule data of the copyable-modeled operands of \p TE for
  /// dependency recalculation at the next bundle scheduling.
  void markCopyableDepsForRecalc(TreeEntry &TE);

  /// \return The vector element size in bits to use when vectorizing the
  /// expression tree ending at \p V. If V is a store, the size is the width of
  /// the stored value. Otherwise, the size is the width of the largest loaded
  /// value reaching V. This method is used by the vectorizer to calculate
  /// vectorization factors.
  unsigned getVectorElementSize(Value *V);

  /// Compute the minimum type sizes required to represent the entries in a
  /// vectorizable tree.
  void computeMinimumValueSizes();

  // \returns maximum vector register size as set by TTI or overridden by
  // cl::opt.
  unsigned getMaxVecRegSize() const { return MaxVecRegSize; }

  // \returns minimum vector register size as set by cl::opt.
  unsigned getMinVecRegSize() const { return MinVecRegSize; }

  /// \returns the number of parts, the type \p VecTy is split at the codegen
  /// phase. The type legalization queries are repeated for the very same types
  /// during the analysis, so the results are cached for the function.
  unsigned
  getNumberOfParts(Type *VecTy, Type *ScalarTy,
                   unsigned Limit = std::numeric_limits<unsigned>::max()) const;

  unsigned getMinVF(unsigned Sz) const {
    return std::max(2U, getMinVecRegSize() / Sz);
  }

  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const;

  /// Check if homogeneous aggregate is isomorphic to some VectorType.
  /// Accepts homogeneous multidimensional aggregate of scalars/vectors like
  /// {[4 x i16], [4 x i16]}, { <2 x float>, <2 x float> },
  /// {{{i16, i16}, {i16, i16}}, {{i16, i16}, {i16, i16}}} and so on.
  ///
  /// \returns number of elements in vector if isomorphism exists, 0 otherwise.
  unsigned canMapToVector(Type *T) const;

  /// \returns true if the vectorized insertvalue result can be stored directly
  /// as a vector, i.e. every insertvalue with an external user is consumed by a
  /// single store only.
  bool canVectorStoreInsertValue(const TreeEntry *E) const;

  /// \returns the source vector type for an InsertElement/InsertValue
  /// buildvector node \p E: the inserted vector type for insertelement, or a
  /// vector of the inserted scalar type wide enough to cover the highest
  /// inserted index for insertvalue.
  FixedVectorType *getInsertBuildVectorSrcTy(const TreeEntry *E) const;

  /// \returns True if the VectorizableTree is both tiny and not fully
  /// vectorizable. We do not vectorize such trees.
  bool isTreeTinyAndNotFullyVectorizable(bool ForReduction = false) const;

  /// Checks if the graph and all its subgraphs cannot be better vectorized.
  /// It may happen, if all gather nodes are loads and they cannot be
  /// "clusterized". In this case even subgraphs cannot be vectorized more
  /// effectively than the base graph.
  bool isTreeNotExtendable() const;

  bool isStridedLoad(ArrayRef<Value *> PointerOps, Type *ScalarTy,
                     Align Alignment, const int64_t Diff,
                     const size_t Sz) const;

  /// Return true if an array of scalar loads can be replaced with a strided
  ///  load (with constant stride).
  ///
  ///  It is possible that the load gets "widened". Suppose that originally each
  ///  load loads `k` bytes and `PointerOps` can be arranged as follows (`%s` is
  ///  constant): %b + 0 * %s + 0 %b + 0 * %s + 1 %b + 0 * %s + 2
  ///  ...
  ///  %b + 0 * %s + (w - 1)
  ///
  ///  %b + 1 * %s + 0
  ///  %b + 1 * %s + 1
  ///  %b + 1 * %s + 2
  ///  ...
  ///  %b + 1 * %s + (w - 1)
  ///  ...
  ///
  ///  %b + (n - 1) * %s + 0
  ///  %b + (n - 1) * %s + 1
  ///  %b + (n - 1) * %s + 2
  ///  ...
  ///  %b + (n - 1) * %s + (w - 1)
  ///
  /// In this case we will generate a strided load of type `<n x (k * w)>`.
  ///
  /// \param PointerOps list of pointer arguments of loads.
  /// \param ElemTy original scalar type of loads.
  /// \param Alignment alignment of the first load.
  /// \param SortedIndices is the order of PointerOps as returned by
  /// `sortPtrAccesses`
  /// \param Diff Pointer difference between the lowest and the highes pointer
  /// in `PointerOps` as returned by `getPointersDiff`.
  /// \param Ptr0 first pointer in `PointersOps`.
  /// \param PtrN last pointer in `PointersOps`.
  /// \param SPtrInfo If the function return `true`, it also sets all the fields
  /// of `SPtrInfo` necessary to generate the strided load later.
  bool analyzeConstantStrideCandidate(
      const ArrayRef<Value *> PointerOps, Type *ElemTy, Align Alignment,
      const SmallVectorImpl<unsigned> &SortedIndices, const int64_t Diff,
      Value *Ptr0, StridedPtrInfo &SPtrInfo) const;

  /// Return true if an array of scalar loads can be replaced with a strided
  /// load (with run-time stride).
  /// \param PointerOps list of pointer arguments of loads.
  /// \param ScalarTy type of loads.
  /// \param CommonAlignment common alignement of loads as computed by
  /// `computeCommonAlignment<LoadInst>`.
  /// \param SortedIndicies is a list of indicies computed by this function such
  /// that the sequence `PointerOps[SortedIndices[0]],
  /// PointerOps[SortedIndicies[1]], ..., PointerOps[SortedIndices[n]]` is
  /// ordered by the coefficient of the stride. For example, if PointerOps is
  /// `%base + %stride, %base, %base + 2 * stride` the `SortedIndices` will be
  /// `[1, 0, 2]`. We follow the convention that if `SortedIndices` has to be
  /// `0, 1, 2, 3, ...` we return empty vector for `SortedIndicies`.
  /// \param SPtrInfo If the function return `true`, it also sets all the fields
  /// of `SPtrInfo` necessary to generate the strided load later.
  /// \param IsLoad Is this a strided load (true) or strided store (false)
  bool analyzeRtStrideCandidate(ArrayRef<Value *> PointerOps, Type *ScalarTy,
                                Align CommonAlignment,
                                SmallVectorImpl<unsigned> &SortedIndices,
                                StridedPtrInfo &SPtrInfo, bool IsLoad) const;

  /// Checks if the given array of loads can be represented as a vectorized,
  /// scatter or just simple gather.
  /// \param VL list of loads.
  /// \param VL0 main load value.
  /// \param Order returned order of load instructions.
  /// \param PointerOps returned list of pointer operands.
  /// \param BestVF return best vector factor, if recursive check found better
  /// vectorization sequences rather than masked gather.
  /// \param TryRecursiveCheck used to check if long masked gather can be
  /// represented as a serie of loads/insert subvector, if profitable.
  LoadsState canVectorizeLoads(ArrayRef<Value *> VL, const Value *VL0,
                               SmallVectorImpl<unsigned> &Order,
                               SmallVectorImpl<Value *> &PointerOps,
                               StridedPtrInfo &SPtrInfo,
                               unsigned *BestVF = nullptr,
                               bool TryRecursiveCheck = true) const;

  /// Checks whether some existing tree entry has scalars equal to \p VL.
  /// \p S is the common opcode of \p VL when one exists; an empty \p S means
  /// the values have no common opcode (mixed buildvector/gather candidates).
  bool hasSameNode(const InstructionsState &S, ArrayRef<Value *> VL) const {
    auto IsSame = [&](const TreeEntry *TE) { return TE->isSame(VL); };
    if (S) {
      // Any vectorized or gather entry equal to VL must contain S.getMainOp()
      // (the representative instruction, which is also the recorded scalar
      // for copyable-elements bundles), so probing the MainOp-indexed maps
      // is sufficient and avoids scanning the whole tree.
      return any_of(getTreeEntries(S.getMainOp()), IsSame) ||
             any_of(ValueToGatherNodes.lookup(S.getMainOp()), IsSame);
    }
    // No common opcode: only gather entries can match. Each non-constant
    // value in VL has to be in the gather entry's scalar list and is
    // therefore present in ValueToGatherNodes. Probe by VL members instead
    // of scanning the whole tree (O(tree) -> O(|VL|)).
    SmallPtrSet<const TreeEntry *, 4> Visited;
    for (Value *V : VL) {
      // Constants/poisons are not tracked in ValueToGatherNodes.
      if (isConstant(V))
        continue;
      for (const TreeEntry *TE : ValueToGatherNodes.lookup(V)) {
        if (!Visited.insert(TE).second)
          continue;
        if (IsSame(TE))
          return true;
      }
    }
    return false;
  }

  /// Registers non-vectorizable sequence of loads
  template <typename T> void registerNonVectorizableLoads(ArrayRef<T *> VL) {
    ListOfKnonwnNonVectorizableLoads.insert(hash_value(VL));
  }

  /// Checks if the given loads sequence is known as not vectorizable
  template <typename T>
  bool areKnownNonVectorizableLoads(ArrayRef<T *> VL) const {
    return ListOfKnonwnNonVectorizableLoads.contains(hash_value(VL));
  }

  OptimizationRemarkEmitter *getORE() { return ORE; }

  /// This structure holds any data we need about the edges being traversed
  /// during buildTreeRec(). We keep track of:
  /// (i) the user TreeEntry index, and
  /// (ii) the index of the edge.
  struct EdgeInfo {
    EdgeInfo() = default;
    EdgeInfo(TreeEntry *UserTE, unsigned EdgeIdx)
        : UserTE(UserTE), EdgeIdx(EdgeIdx) {}
    /// The user TreeEntry.
    TreeEntry *UserTE = nullptr;
    /// The operand index of the use.
    unsigned EdgeIdx = UINT_MAX;
#ifndef NDEBUG
    friend inline raw_ostream &operator<<(raw_ostream &OS,
                                          const BoUpSLP::EdgeInfo &EI) {
      EI.dump(OS);
      return OS;
    }
    /// Debug print.
    void dump(raw_ostream &OS) const {
      OS << "{User:" << (UserTE ? std::to_string(UserTE->Idx) : "null")
         << " EdgeIdx:" << EdgeIdx << "}";
    }
    LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
#endif
    bool operator==(const EdgeInfo &Other) const {
      return UserTE == Other.UserTE && EdgeIdx == Other.EdgeIdx;
    }

    operator bool() const { return UserTE != nullptr; }
  };
  friend struct DenseMapInfo<EdgeInfo>;

  /// A helper class used for scoring candidates for two consecutive lanes.
  class LookAheadHeuristics {
    const TargetLibraryInfo &TLI;
    const DataLayout &DL;
    ScalarEvolution &SE;
    const BoUpSLP &R;
    int NumLanes; // Total number of lanes (aka vectorization factor).
    int MaxLevel; // The maximum recursion depth for accumulating score.

  public:
    LookAheadHeuristics(const TargetLibraryInfo &TLI, const DataLayout &DL,
                        ScalarEvolution &SE, const BoUpSLP &R, int NumLanes,
                        int MaxLevel)
        : TLI(TLI), DL(DL), SE(SE), R(R), NumLanes(NumLanes),
          MaxLevel(MaxLevel) {}

    // The hard-coded scores listed here are not very important, though it shall
    // be higher for better matches to improve the resulting cost. When
    // computing the scores of matching one sub-tree with another, we are
    // basically counting the number of values that are matching. So even if all
    // scores are set to 1, we would still get a decent matching result.
    // However, sometimes we have to break ties. For example we may have to
    // choose between matching loads vs matching opcodes. This is what these
    // scores are helping us with: they provide the order of preference. Also,
    // this is important if the scalar is externally used or used in another
    // tree entry node in the different lane.

    /// Loads from consecutive memory addresses, e.g. load(A[i]), load(A[i+1]).
    static constexpr int ScoreConsecutiveLoads = 40;
    /// The same load multiple times. This should have a better score than
    /// `ScoreSplat` because it in x86 for a 2-lane vector we can represent it
    /// with `movddup (%reg), xmm0` which has a throughput of 0.5 versus 0.5 for
    /// a vector load and 1.0 for a broadcast.
    static constexpr int ScoreSplatLoads = 30;
    /// Loads from reversed memory addresses, e.g. load(A[i+1]), load(A[i]).
    static constexpr int ScoreReversedLoads = 30;
    /// A load candidate for masked gather.
    static constexpr int ScoreMaskedGatherCandidate = 10;
    /// ExtractElementInst from same vector and consecutive indexes.
    static constexpr int ScoreConsecutiveExtracts = 40;
    /// ExtractElementInst from same vector and reversed indices.
    static constexpr int ScoreReversedExtracts = 30;
    /// Constants.
    static constexpr int ScoreConstants = 15;
    /// Same constants.
    static constexpr int ScoreSameConstants = 17;
    /// Instructions with the same opcode.
    static constexpr int ScoreSameOpcode = 20;
    /// Instructions with alt opcodes (e.g, add + sub).
    static constexpr int ScoreAltOpcodes = 10;
    /// Identical instructions (a.k.a. splat or broadcast).
    static constexpr int ScoreSplat = 10;
    /// Matching with an undef is preferable to failing.
    static constexpr int ScoreUndef = 10;
    /// Score for failing to find a decent match.
    static constexpr int ScoreFail = 0;
    /// Score if all users are vectorized.
    static constexpr int ScoreAllUserVectorized = 10;

    /// \returns the score of placing \p V1 and \p V2 in consecutive lanes.
    /// \p U1 and \p U2 are the users of \p V1 and \p V2.
    /// Also, checks if \p V1 and \p V2 are compatible with instructions in \p
    /// MainAltOps.
    int getShallowScore(Value *V1, Value *V2, Instruction *U1, Instruction *U2,
                        ArrayRef<Value *> MainAltOps) const;

    /// Go through the operands of \p LHS and \p RHS recursively until
    /// MaxLevel, and return the cummulative score. \p U1 and \p U2 are
    /// the users of \p LHS and \p RHS (that is \p LHS and \p RHS are operands
    /// of \p U1 and \p U2), except at the beginning of the recursion where
    /// these are set to nullptr.
    ///
    /// For example:
    /// \verbatim
    ///  A[0]  B[0]  A[1]  B[1]  C[0] D[0]  B[1] A[1]
    ///     \ /         \ /         \ /        \ /
    ///      +           +           +          +
    ///     G1          G2          G3         G4
    /// \endverbatim
    /// The getScoreAtLevelRec(G1, G2) function will try to match the nodes at
    /// each level recursively, accumulating the score. It starts from matching
    /// the additions at level 0, then moves on to the loads (level 1). The
    /// score of G1 and G2 is higher than G1 and G3, because {A[0],A[1]} and
    /// {B[0],B[1]} match with LookAheadHeuristics::ScoreConsecutiveLoads, while
    /// {A[0],C[0]} has a score of LookAheadHeuristics::ScoreFail.
    /// Please note that the order of the operands does not matter, as we
    /// evaluate the score of all profitable combinations of operands. In
    /// other words the score of G1 and G4 is the same as G1 and G2. This
    /// heuristic is based on ideas described in:
    ///   Look-ahead SLP: Auto-vectorization in the presence of commutative
    ///   operations, CGO 2018 by Vasileios Porpodas, Rodrigo C. O. Rocha,
    ///   Luís F. W. Góes
    int getScoreAtLevelRec(Value *LHS, Value *RHS, Instruction *U1,
                           Instruction *U2, int CurrLevel,
                           ArrayRef<Value *> MainAltOps) const {

      // Get the shallow score of V1 and V2.
      int ShallowScoreAtThisLevel =
          getShallowScore(LHS, RHS, U1, U2, MainAltOps);

      // If reached MaxLevel,
      //  or if V1 and V2 are not instructions,
      //  or if they are SPLAT,
      //  or if they are not consecutive,
      //  or if profitable to vectorize loads or extractelements, early return
      //  the current cost.
      auto *I1 = dyn_cast<Instruction>(LHS);
      auto *I2 = dyn_cast<Instruction>(RHS);
      if (CurrLevel == MaxLevel || !(I1 && I2) || I1 == I2 ||
          ShallowScoreAtThisLevel == LookAheadHeuristics::ScoreFail ||
          (((isa<LoadInst>(I1) && isa<LoadInst>(I2)) ||
            (I1->getNumOperands() > 2 && I2->getNumOperands() > 2) ||
            (isa<ExtractElementInst>(I1) && isa<ExtractElementInst>(I2))) &&
           ShallowScoreAtThisLevel))
        return ShallowScoreAtThisLevel;
      assert(I1 && I2 && "Should have early exited.");

      // Contains the I2 operand indexes that got matched with I1 operands.
      SmallSet<unsigned, 4> Op2Used;

      // Recursion towards the operands of I1 and I2. We are trying all possible
      // operand pairs, and keeping track of the best score.
      if (I1->getNumOperands() != I2->getNumOperands())
        return LookAheadHeuristics::ScoreSameOpcode;
      for (unsigned OpIdx1 = 0, NumOperands1 = I1->getNumOperands();
           OpIdx1 != NumOperands1; ++OpIdx1) {
        // Try to pair op1I with the best operand of I2.
        int MaxTmpScore = 0;
        unsigned MaxOpIdx2 = 0;
        bool FoundBest = false;
        // If I2 is commutative try all combinations.
        unsigned FromIdx = isCommutative(I2) ? 0 : OpIdx1;
        unsigned ToIdx = isCommutative(I2)
                             ? I2->getNumOperands()
                             : std::min(I2->getNumOperands(), OpIdx1 + 1);
        assert(FromIdx <= ToIdx && "Bad index");
        for (unsigned OpIdx2 = FromIdx; OpIdx2 != ToIdx; ++OpIdx2) {
          // Skip operands already paired with OpIdx1.
          if (Op2Used.count(OpIdx2))
            continue;
          // Recursively calculate the cost at each level
          int TmpScore =
              getScoreAtLevelRec(I1->getOperand(OpIdx1), I2->getOperand(OpIdx2),
                                 I1, I2, CurrLevel + 1, {});
          // Look for the best score.
          if (TmpScore > LookAheadHeuristics::ScoreFail &&
              TmpScore > MaxTmpScore) {
            MaxTmpScore = TmpScore;
            MaxOpIdx2 = OpIdx2;
            FoundBest = true;
          }
        }
        if (FoundBest) {
          // Pair {OpIdx1, MaxOpIdx2} was found to be best. Never revisit it.
          Op2Used.insert(MaxOpIdx2);
          ShallowScoreAtThisLevel += MaxTmpScore;
        }
      }
      return ShallowScoreAtThisLevel;
    }
  };
  /// A helper data structure to hold the operands of a vector of instructions.
  /// This supports a fixed vector length for all operand vectors.
  class VLOperands {
    /// For each operand we need (i) the value, and (ii) the opcode that it
    /// would be attached to if the expression was in a left-linearized form.
    /// This is required to avoid illegal operand reordering.
    /// For example:
    /// \verbatim
    ///                         0 Op1
    ///                         |/
    /// Op1 Op2   Linearized    + Op2
    ///   \ /     ---------->   |/
    ///    -                    -
    ///
    /// Op1 - Op2            (0 + Op1) - Op2
    /// \endverbatim
    ///
    /// Value Op1 is attached to a '+' operation, and Op2 to a '-'.
    ///
    /// Another way to think of this is to track all the operations across the
    /// path from the operand all the way to the root of the tree and to
    /// calculate the operation that corresponds to this path. For example, the
    /// path from Op2 to the root crosses the RHS of the '-', therefore the
    /// corresponding operation is a '-' (which matches the one in the
    /// linearized tree, as shown above).
    ///
    /// For lack of a better term, we refer to this operation as Accumulated
    /// Path Operation (APO).
    struct OperandData {
      OperandData() = default;
      OperandData(Value *V, bool APO, bool IsUsed)
          : V(V), APO(APO), IsUsed(IsUsed) {}
      /// The operand value.
      Value *V = nullptr;
      /// TreeEntries only allow a single opcode, or an alternate sequence of
      /// them (e.g, +, -). Therefore, we can safely use a boolean value for the
      /// APO. It is set to 'true' if 'V' is attached to an inverse operation
      /// in the left-linearized form (e.g., Sub/Div), and 'false' otherwise
      /// (e.g., Add/Mul)
      bool APO = false;
      /// Helper data for the reordering function.
      bool IsUsed = false;
    };

    /// During operand reordering, we are trying to select the operand at lane
    /// that matches best with the operand at the neighboring lane. Our
    /// selection is based on the type of value we are looking for. For example,
    /// if the neighboring lane has a load, we need to look for a load that is
    /// accessing a consecutive address. These strategies are summarized in the
    /// 'ReorderingMode' enumerator.
    enum class ReorderingMode {
      Load,     ///< Matching loads to consecutive memory addresses
      Opcode,   ///< Matching instructions based on opcode (same or alternate)
      Constant, ///< Matching constants
      Splat,    ///< Matching the same instruction multiple times (broadcast)
      Failed,   ///< We failed to create a vectorizable group
    };

    using OperandDataVec = SmallVector<OperandData, 2>;

    /// A vector of operand vectors.
    SmallVector<OperandDataVec, 4> OpsVec;
    /// When VL[0] is IntrinsicInst, ArgSize is CallBase::arg_size. When VL[0]
    /// is not IntrinsicInst, ArgSize is User::getNumOperands.
    unsigned ArgSize = 0;

    const TargetLibraryInfo &TLI;
    const DataLayout &DL;
    ScalarEvolution &SE;
    const BoUpSLP &R;
    const Loop *L = nullptr;

    /// \returns the operand data at \p OpIdx and \p Lane.
    OperandData &getData(unsigned OpIdx, unsigned Lane) {
      return OpsVec[OpIdx][Lane];
    }

    /// \returns the operand data at \p OpIdx and \p Lane. Const version.
    const OperandData &getData(unsigned OpIdx, unsigned Lane) const {
      return OpsVec[OpIdx][Lane];
    }

    /// Clears the used flag for all entries.
    void clearUsed() {
      for (unsigned OpIdx = 0, NumOperands = getNumOperands();
           OpIdx != NumOperands; ++OpIdx)
        for (unsigned Lane = 0, NumLanes = getNumLanes(); Lane != NumLanes;
             ++Lane)
          OpsVec[OpIdx][Lane].IsUsed = false;
    }

    /// Swap the operand at \p OpIdx1 with that one at \p OpIdx2.
    void swap(unsigned OpIdx1, unsigned OpIdx2, unsigned Lane) {
      std::swap(OpsVec[OpIdx1][Lane], OpsVec[OpIdx2][Lane]);
    }

    /// \param Lane lane of the operands under analysis.
    /// \param OpIdx operand index in \p Lane lane we're looking the best
    /// candidate for.
    /// \param Idx operand index of the current candidate value.
    /// \returns The additional score due to possible broadcasting of the
    /// elements in the lane. It is more profitable to have power-of-2 unique
    /// elements in the lane, it will be vectorized with higher probability
    /// after removing duplicates. Currently the SLP vectorizer supports only
    /// vectorization of the power-of-2 number of unique scalars.
    int getSplatScore(unsigned Lane, unsigned OpIdx, unsigned Idx,
                      const SmallBitVector &UsedLanes) const {
      Value *IdxLaneV = getData(Idx, Lane).V;
      if (!isa<Instruction>(IdxLaneV) || IdxLaneV == getData(OpIdx, Lane).V ||
          isa<ExtractElementInst>(IdxLaneV))
        return 0;
      SmallDenseMap<Value *, unsigned, 4> Uniques;
      for (unsigned Ln : seq<unsigned>(getNumLanes())) {
        if (Ln == Lane)
          continue;
        Value *OpIdxLnV = getData(OpIdx, Ln).V;
        if (!isa<Instruction>(OpIdxLnV))
          return 0;
        Uniques.try_emplace(OpIdxLnV, Ln);
      }
      unsigned UniquesCount = Uniques.size();
      auto IdxIt = Uniques.find(IdxLaneV);
      unsigned UniquesCntWithIdxLaneV =
          IdxIt != Uniques.end() ? UniquesCount : UniquesCount + 1;
      Value *OpIdxLaneV = getData(OpIdx, Lane).V;
      auto OpIdxIt = Uniques.find(OpIdxLaneV);
      unsigned UniquesCntWithOpIdxLaneV =
          OpIdxIt != Uniques.end() ? UniquesCount : UniquesCount + 1;
      if (UniquesCntWithIdxLaneV == UniquesCntWithOpIdxLaneV)
        return 0;
      return std::min(bit_ceil(UniquesCntWithOpIdxLaneV) -
                          UniquesCntWithOpIdxLaneV,
                      UniquesCntWithOpIdxLaneV -
                          bit_floor(UniquesCntWithOpIdxLaneV)) -
             ((IdxIt != Uniques.end() && UsedLanes.test(IdxIt->second))
                  ? UniquesCntWithIdxLaneV - bit_floor(UniquesCntWithIdxLaneV)
                  : bit_ceil(UniquesCntWithIdxLaneV) - UniquesCntWithIdxLaneV);
    }

    /// \param Lane lane of the operands under analysis.
    /// \param OpIdx operand index in \p Lane lane we're looking the best
    /// candidate for.
    /// \param Idx operand index of the current candidate value.
    /// \returns The additional score for the scalar which users are all
    /// vectorized.
    int getExternalUseScore(unsigned Lane, unsigned OpIdx, unsigned Idx) const {
      Value *IdxLaneV = getData(Idx, Lane).V;
      Value *OpIdxLaneV = getData(OpIdx, Lane).V;
      // Do not care about number of uses for vector-like instructions
      // (extractelement/extractvalue with constant indices), they are extracts
      // themselves and already externally used. Vectorization of such
      // instructions does not add extra extractelement instruction, just may
      // remove it.
      if (isVectorLikeInstWithConstOps(IdxLaneV) &&
          isVectorLikeInstWithConstOps(OpIdxLaneV))
        return LookAheadHeuristics::ScoreAllUserVectorized;
      auto *IdxLaneI = dyn_cast<Instruction>(IdxLaneV);
      if (!IdxLaneI || !isa<Instruction>(OpIdxLaneV))
        return 0;
      return R.areAllUsersVectorized(IdxLaneI)
                 ? LookAheadHeuristics::ScoreAllUserVectorized
                 : 0;
    }

    /// Score scaling factor for fully compatible instructions but with
    /// different number of external uses. Allows better selection of the
    /// instructions with less external uses.
    static constexpr int ScoreScaleFactor = 10;
    /// Scale factor for constants only.
    static constexpr int ScoreConstantScaleFactor = 6;

    /// \Returns the look-ahead score, which tells us how much the sub-trees
    /// rooted at \p LHS and \p RHS match, the more they match the higher the
    /// score. This helps break ties in an informed way when we cannot decide on
    /// the order of the operands by just considering the immediate
    /// predecessors.
    int getLookAheadScore(Value *LHS, Value *RHS, ArrayRef<Value *> MainAltOps,
                          int Lane, unsigned OpIdx, unsigned Idx, bool &IsUsed,
                          const SmallBitVector &UsedLanes);

    /// Best defined scores per lanes between the passes. Used to choose the
    /// best operand (with the highest score) between the passes.
    /// The key - {Operand Index, Lane}.
    /// The value - the best score between the passes for the lane and the
    /// operand.
    SmallDenseMap<std::pair<unsigned, unsigned>, unsigned, 8>
        BestScoresPerLanes;

    // Search all operands in Ops[*][Lane] for the one that matches best
    // Ops[OpIdx][LastLane] and return its opreand index.
    // If no good match can be found, return std::nullopt.
    std::optional<unsigned>
    getBestOperand(unsigned OpIdx, int Lane, int LastLane,
                   ArrayRef<ReorderingMode> ReorderingModes,
                   ArrayRef<Value *> MainAltOps,
                   const SmallBitVector &UsedLanes) {
      unsigned NumOperands = getNumOperands();

      // The operand of the previous lane at OpIdx.
      Value *OpLastLane = getData(OpIdx, LastLane).V;

      // Our strategy mode for OpIdx.
      ReorderingMode RMode = ReorderingModes[OpIdx];
      if (RMode == ReorderingMode::Failed)
        return std::nullopt;

      // The linearized opcode of the operand at OpIdx, Lane.
      bool OpIdxAPO = getData(OpIdx, Lane).APO;

      // The best operand index and its score.
      // Sometimes we have more than one option (e.g., Opcode and Undefs), so we
      // are using the score to differentiate between the two.
      struct BestOpData {
        std::optional<unsigned> Idx;
        unsigned Score = 0;
      } BestOp;
      BestOp.Score =
          BestScoresPerLanes.try_emplace(std::make_pair(OpIdx, Lane), 0)
              .first->second;

      // Track if the operand must be marked as used. If the operand is set to
      // Score 1 explicitly (because of non power-of-2 unique scalars, we may
      // want to reestimate the operands again on the following iterations).
      bool IsUsed = RMode == ReorderingMode::Splat ||
                    RMode == ReorderingMode::Constant ||
                    RMode == ReorderingMode::Load;
      // Iterate through all unused operands and look for the best.
      for (unsigned Idx = 0; Idx != NumOperands; ++Idx) {
        // Get the operand at Idx and Lane.
        OperandData &OpData = getData(Idx, Lane);
        Value *Op = OpData.V;
        bool OpAPO = OpData.APO;

        // Skip already selected operands.
        if (OpData.IsUsed)
          continue;

        // Skip if we are trying to move the operand to a position with a
        // different opcode in the linearized tree form. This would break the
        // semantics.
        if (OpAPO != OpIdxAPO)
          continue;

        // Look for an operand that matches the current mode.
        switch (RMode) {
        case ReorderingMode::Load:
        case ReorderingMode::Opcode: {
          bool LeftToRight = Lane > LastLane;
          Value *OpLeft = (LeftToRight) ? OpLastLane : Op;
          Value *OpRight = (LeftToRight) ? Op : OpLastLane;
          int Score = getLookAheadScore(OpLeft, OpRight, MainAltOps, Lane,
                                        OpIdx, Idx, IsUsed, UsedLanes);
          if (Score > static_cast<int>(BestOp.Score) ||
              (Score > 0 && Score == static_cast<int>(BestOp.Score) &&
               Idx == OpIdx)) {
            BestOp.Idx = Idx;
            BestOp.Score = Score;
            BestScoresPerLanes[std::make_pair(OpIdx, Lane)] = Score;
          }
          break;
        }
        case ReorderingMode::Constant:
          if (isa<Constant>(Op) ||
              (!BestOp.Score && L && L->isLoopInvariant(Op))) {
            BestOp.Idx = Idx;
            if (isa<Constant>(Op)) {
              BestOp.Score = LookAheadHeuristics::ScoreConstants;
              BestScoresPerLanes[std::make_pair(OpIdx, Lane)] =
                  LookAheadHeuristics::ScoreConstants;
            }
            if (isa<UndefValue>(Op) || !isa<Constant>(Op))
              IsUsed = false;
          }
          break;
        case ReorderingMode::Splat:
          if (Op == OpLastLane || (!BestOp.Score && isa<Constant>(Op))) {
            IsUsed = Op == OpLastLane;
            if (Op == OpLastLane) {
              BestOp.Score = LookAheadHeuristics::ScoreSplat;
              BestScoresPerLanes[std::make_pair(OpIdx, Lane)] =
                  LookAheadHeuristics::ScoreSplat;
            }
            BestOp.Idx = Idx;
          }
          break;
        case ReorderingMode::Failed:
          llvm_unreachable("Not expected Failed reordering mode.");
        }
      }

      if (BestOp.Idx) {
        getData(*BestOp.Idx, Lane).IsUsed = IsUsed;
        return BestOp.Idx;
      }
      // If we could not find a good match return std::nullopt.
      return std::nullopt;
    }

    /// Helper for reorderOperandVecs.
    /// \returns the lane that we should start reordering from. This is the one
    /// which has the least number of operands that can freely move about or
    /// less profitable because it already has the most optimal set of operands.
    unsigned getBestLaneToStartReordering() const {
      unsigned Min = UINT_MAX;
      unsigned SameOpNumber = 0;
      // std::pair<unsigned, unsigned> is used to implement a simple voting
      // algorithm and choose the lane with the least number of operands that
      // can freely move about or less profitable because it already has the
      // most optimal set of operands. The first unsigned is a counter for
      // voting, the second unsigned is the counter of lanes with instructions
      // with same/alternate opcodes and same parent basic block.
      MapVector<unsigned, std::pair<unsigned, unsigned>> HashMap;
      // Try to be closer to the original results, if we have multiple lanes
      // with same cost. If 2 lanes have the same cost, use the one with the
      // highest index.
      for (int I = getNumLanes(); I > 0; --I) {
        unsigned Lane = I - 1;
        OperandsOrderData NumFreeOpsHash =
            getMaxNumOperandsThatCanBeReordered(Lane);
        // Compare the number of operands that can move and choose the one with
        // the least number.
        if (NumFreeOpsHash.NumOfAPOs < Min) {
          Min = NumFreeOpsHash.NumOfAPOs;
          SameOpNumber = NumFreeOpsHash.NumOpsWithSameOpcodeParent;
          HashMap.clear();
          HashMap[NumFreeOpsHash.Hash] = std::make_pair(1, Lane);
        } else if (NumFreeOpsHash.NumOfAPOs == Min &&
                   NumFreeOpsHash.NumOpsWithSameOpcodeParent < SameOpNumber) {
          // Select the most optimal lane in terms of number of operands that
          // should be moved around.
          SameOpNumber = NumFreeOpsHash.NumOpsWithSameOpcodeParent;
          HashMap[NumFreeOpsHash.Hash] = std::make_pair(1, Lane);
        } else if (NumFreeOpsHash.NumOfAPOs == Min &&
                   NumFreeOpsHash.NumOpsWithSameOpcodeParent == SameOpNumber) {
          auto [It, Inserted] =
              HashMap.try_emplace(NumFreeOpsHash.Hash, 1, Lane);
          if (!Inserted)
            ++It->second.first;
        }
      }
      // Select the lane with the minimum counter.
      unsigned BestLane = 0;
      unsigned CntMin = UINT_MAX;
      for (const auto &Data : reverse(HashMap)) {
        if (Data.second.first < CntMin) {
          CntMin = Data.second.first;
          BestLane = Data.second.second;
        }
      }
      return BestLane;
    }

    /// Data structure that helps to reorder operands.
    struct OperandsOrderData {
      /// The best number of operands with the same APOs, which can be
      /// reordered.
      unsigned NumOfAPOs = UINT_MAX;
      /// Number of operands with the same/alternate instruction opcode and
      /// parent.
      unsigned NumOpsWithSameOpcodeParent = 0;
      /// Hash for the actual operands ordering.
      /// Used to count operands, actually their position id and opcode
      /// value. It is used in the voting mechanism to find the lane with the
      /// least number of operands that can freely move about or less profitable
      /// because it already has the most optimal set of operands. Can be
      /// replaced with SmallVector<unsigned> instead but hash code is faster
      /// and requires less memory.
      unsigned Hash = 0;
    };
    /// \returns the maximum number of operands that are allowed to be reordered
    /// for \p Lane and the number of compatible instructions(with the same
    /// parent/opcode). This is used as a heuristic for selecting the first lane
    /// to start operand reordering.
    OperandsOrderData getMaxNumOperandsThatCanBeReordered(unsigned Lane) const {
      unsigned CntTrue = 0;
      unsigned NumOperands = getNumOperands();
      // Operands with the same APO can be reordered. We therefore need to count
      // how many of them we have for each APO, like this: Cnt[APO] = x.
      // Since we only have two APOs, namely true and false, we can avoid using
      // a map. Instead we can simply count the number of operands that
      // correspond to one of them (in this case the 'true' APO), and calculate
      // the other by subtracting it from the total number of operands.
      // Operands with the same instruction opcode and parent are more
      // profitable since we don't need to move them in many cases, with a high
      // probability such lane already can be vectorized effectively.
      bool AllUndefs = true;
      unsigned NumOpsWithSameOpcodeParent = 0;
      Instruction *OpcodeI = nullptr;
      BasicBlock *Parent = nullptr;
      unsigned Hash = 0;
      for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
        const OperandData &OpData = getData(OpIdx, Lane);
        if (OpData.APO)
          ++CntTrue;
        // Use Boyer-Moore majority voting for finding the majority opcode and
        // the number of times it occurs.
        if (auto *I = dyn_cast<Instruction>(OpData.V)) {
          if (!OpcodeI || !getSameOpcode({OpcodeI, I}, TLI) ||
              I->getParent() != Parent) {
            if (NumOpsWithSameOpcodeParent == 0) {
              NumOpsWithSameOpcodeParent = 1;
              OpcodeI = I;
              Parent = I->getParent();
            } else {
              --NumOpsWithSameOpcodeParent;
            }
          } else {
            ++NumOpsWithSameOpcodeParent;
          }
        }
        Hash = hash_combine(
            Hash, hash_value((OpIdx + 1) * (OpData.V->getValueID() + 1)));
        AllUndefs = AllUndefs && isa<UndefValue>(OpData.V);
      }
      if (AllUndefs)
        return {};
      OperandsOrderData Data;
      Data.NumOfAPOs = std::max(CntTrue, NumOperands - CntTrue);
      Data.NumOpsWithSameOpcodeParent = NumOpsWithSameOpcodeParent;
      Data.Hash = Hash;
      return Data;
    }

    /// Go through the instructions in VL and append their operands.
    void appendOperands(ArrayRef<Value *> VL, ArrayRef<ValueList> Operands,
                        const InstructionsState &S) {
      assert(!Operands.empty() && !VL.empty() && "Bad list of operands");
      assert((empty() || all_of(Operands,
                                [this](const ValueList &VL) {
                                  return VL.size() == getNumLanes();
                                })) &&
             "Expected same number of lanes");
      assert(S.valid() && "InstructionsState is invalid.");
      // IntrinsicInst::isCommutative returns true if swapping the first "two"
      // arguments to the intrinsic produces the same result.
      Instruction *MainOp = S.getMainOp();
      ArgSize = getNumberOfPotentiallyCommutativeOps(MainOp);
      OpsVec.resize(ArgSize);
      unsigned NumLanes = VL.size();
      for (OperandDataVec &Ops : OpsVec)
        Ops.resize(NumLanes);
      for (unsigned Lane : seq<unsigned>(NumLanes)) {
        // Our tree has just 3 nodes: the root and two operands.
        // It is therefore trivial to get the APO. We only need to check the
        // opcode of V and whether the operand at OpIdx is the LHS or RHS
        // operand. The LHS operand of both add and sub is never attached to an
        // inversese operation in the linearized form, therefore its APO is
        // false. The RHS is true only if V is an inverse operation.

        // Since operand reordering is performed on groups of commutative
        // operations or alternating sequences (e.g., +, -), we can safely tell
        // the inverse operations by checking commutativity.
        auto *I = dyn_cast<Instruction>(VL[Lane]);
        if (!I && isa<PoisonValue>(VL[Lane])) {
          for (unsigned OpIdx : seq<unsigned>(ArgSize))
            OpsVec[OpIdx][Lane] = {Operands[OpIdx][Lane], true, false};
          continue;
        }
        bool IsInverseOperation = false;
        if (S.isCopyableElement(VL[Lane])) {
          // The value is a copyable element.
          IsInverseOperation =
              !isCommutative(MainOp, VL[Lane], /*IsCopyable=*/true);
        } else {
          assert(I && "Expected instruction");
          auto [SelectedOp, Ops] = convertTo(I, S);
          // We cannot check commutativity by the converted instruction
          // (SelectedOp) because isCommutative also examines def-use
          // relationships.
          IsInverseOperation = !isCommutative(SelectedOp, I);
        }
        for (unsigned OpIdx : seq<unsigned>(ArgSize)) {
          bool APO = (OpIdx == 0) ? false : IsInverseOperation;
          OpsVec[OpIdx][Lane] = {Operands[OpIdx][Lane], APO, false};
        }
      }
    }

    /// \returns the number of operands.
    unsigned getNumOperands() const { return ArgSize; }

    /// \returns the number of lanes.
    unsigned getNumLanes() const { return OpsVec[0].size(); }

    /// \returns the operand value at \p OpIdx and \p Lane.
    Value *getValue(unsigned OpIdx, unsigned Lane) const {
      return getData(OpIdx, Lane).V;
    }

    /// \returns true if the data structure is empty.
    bool empty() const { return OpsVec.empty(); }

    /// Clears the data.
    void clear() { OpsVec.clear(); }

    /// \Returns true if there are enough operands identical to \p Op to fill
    /// the whole vector (it is mixed with constants or loop invariant values).
    /// Note: This modifies the 'IsUsed' flag, so a cleanUsed() must follow.
    bool shouldBroadcast(Value *Op, unsigned OpIdx, unsigned Lane) {
      assert(Op == getValue(OpIdx, Lane) &&
             "Op is expected to be getValue(OpIdx, Lane).");
      // Small number of loads - try load matching.
      if (isa<LoadInst>(Op) && getNumLanes() == 2 && getNumOperands() == 2)
        return false;
      bool OpAPO = getData(OpIdx, Lane).APO;
      bool IsInvariant = L && L->isLoopInvariant(Op);
      unsigned Cnt = 0;
      for (unsigned Ln = 0, Lns = getNumLanes(); Ln != Lns; ++Ln) {
        if (Ln == Lane)
          continue;
        // This is set to true if we found a candidate for broadcast at Lane.
        bool FoundCandidate = false;
        for (unsigned OpI = 0, OpE = getNumOperands(); OpI != OpE; ++OpI) {
          OperandData &Data = getData(OpI, Ln);
          if (Data.APO != OpAPO || Data.IsUsed)
            continue;
          Value *OpILane = getValue(OpI, Lane);
          bool IsConstantOp = isa<Constant>(OpILane);
          // Consider the broadcast candidate if:
          // 1. Same value is found in one of the operands.
          if (Data.V == Op ||
              // 2. The operand in the given lane is not constant but there is a
              // constant operand in another lane (which can be moved to the
              // given lane). In this case we can represent it as a simple
              // permutation of constant and broadcast.
              (!IsConstantOp &&
               ((Lns > 2 && isa<Constant>(Data.V)) ||
                // 2.1. If we have only 2 lanes, need to check that value in the
                // next lane does not build same opcode sequence.
                (Lns == 2 &&
                 !getSameOpcode({Op, getValue((OpI + 1) % OpE, Ln)}, TLI) &&
                 isa<Constant>(Data.V)))) ||
              // 3. The operand in the current lane is loop invariant (can be
              // hoisted out) and another operand is also a loop invariant
              // (though not a constant). In this case the whole vector can be
              // hoisted out.
              // FIXME: need to teach the cost model about this case for better
              // estimation.
              (IsInvariant && !isa<Constant>(Data.V) &&
               !getSameOpcode({Op, Data.V}, TLI) &&
               L->isLoopInvariant(Data.V))) {
            FoundCandidate = true;
            Data.IsUsed = Data.V == Op;
            if (Data.V == Op)
              ++Cnt;
            break;
          }
        }
        if (!FoundCandidate)
          return false;
      }
      return getNumLanes() == 2 || Cnt > 1;
    }

    /// Checks if there is at least single compatible operand in lanes other
    /// than \p Lane, compatible with the operand \p Op.
    bool canBeVectorized(Instruction *Op, unsigned OpIdx, unsigned Lane) const {
      assert(Op == getValue(OpIdx, Lane) &&
             "Op is expected to be getValue(OpIdx, Lane).");
      bool OpAPO = getData(OpIdx, Lane).APO;
      for (unsigned Ln = 0, Lns = getNumLanes(); Ln != Lns; ++Ln) {
        if (Ln == Lane)
          continue;
        if (any_of(seq<unsigned>(getNumOperands()), [&](unsigned OpI) {
              const OperandData &Data = getData(OpI, Ln);
              if (Data.APO != OpAPO || Data.IsUsed)
                return true;
              Value *OpILn = getValue(OpI, Ln);
              return (L && L->isLoopInvariant(OpILn)) ||
                     (getSameOpcode({Op, OpILn}, TLI) &&
                      allSameBlock({Op, OpILn}));
            }))
          return true;
      }
      return false;
    }

  public:
    /// Initialize with all the operands of the instruction vector \p RootVL.
    VLOperands(ArrayRef<Value *> RootVL, ArrayRef<ValueList> Operands,
               const InstructionsState &S, const BoUpSLP &R)
        : TLI(*R.TLI), DL(*R.DL), SE(*R.SE), R(R),
          L(R.LI->getLoopFor(S.getMainOp()->getParent())) {
      // Append all the operands of RootVL.
      appendOperands(RootVL, Operands, S);
    }

    /// Initialize with flattened operand columns of an associative node.
    /// ArgSize is taken from \p Operands, APO is always false.
    VLOperands(ArrayRef<ValueList> Operands, const BasicBlock *BB,
               const BoUpSLP &R)
        : TLI(*R.TLI), DL(*R.DL), SE(*R.SE), R(R), L(R.LI->getLoopFor(BB)) {
      assert(!Operands.empty() && "Expected at least one operand column");
      ArgSize = Operands.size();
      OpsVec.resize(ArgSize);
      unsigned NumLanes = Operands.front().size();
      for (auto [OpIdx, Ops] : enumerate(OpsVec)) {
        Ops.resize(NumLanes);
        for (unsigned Lane : seq<unsigned>(NumLanes))
          Ops[Lane] = OperandData(Operands[OpIdx][Lane], /*APO=*/false,
                                  /*IsUsed=*/false);
      }
    }

    /// \Returns a value vector with the operands across all lanes for the
    /// opearnd at \p OpIdx.
    ValueList getVL(unsigned OpIdx) const {
      ValueList OpVL(OpsVec[OpIdx].size());
      assert(OpsVec[OpIdx].size() == getNumLanes() &&
             "Expected same num of lanes across all operands");
      for (unsigned Lane = 0, Lanes = getNumLanes(); Lane != Lanes; ++Lane)
        OpVL[Lane] = OpsVec[OpIdx][Lane].V;
      return OpVL;
    }

    // Performs operand reordering for 2 or more operands.
    // The original operands are in OrigOps[OpIdx][Lane].
    // The reordered operands are returned in 'SortedOps[OpIdx][Lane]'.
    void reorder() {
      unsigned NumOperands = getNumOperands();
      unsigned NumLanes = getNumLanes();
      // Each operand has its own mode. We are using this mode to help us select
      // the instructions for each lane, so that they match best with the ones
      // we have selected so far.
      SmallVector<ReorderingMode, 2> ReorderingModes(NumOperands);

      // This is a greedy single-pass algorithm. We are going over each lane
      // once and deciding on the best order right away with no back-tracking.
      // However, in order to increase its effectiveness, we start with the lane
      // that has operands that can move the least. For example, given the
      // following lanes:
      //  Lane 0 : A[0] = B[0] + C[0]   // Visited 3rd
      //  Lane 1 : A[1] = C[1] - B[1]   // Visited 1st
      //  Lane 2 : A[2] = B[2] + C[2]   // Visited 2nd
      //  Lane 3 : A[3] = C[3] - B[3]   // Visited 4th
      // we will start at Lane 1, since the operands of the subtraction cannot
      // be reordered. Then we will visit the rest of the lanes in a circular
      // fashion. That is, Lanes 2, then Lane 0, and finally Lane 3.

      // Find the first lane that we will start our search from.
      unsigned FirstLane = getBestLaneToStartReordering();

      // Initialize the modes.
      for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
        Value *OpLane0 = getValue(OpIdx, FirstLane);
        // Keep track if we have instructions with all the same opcode on one
        // side.
        if (auto *OpILane0 = dyn_cast<Instruction>(OpLane0)) {
          // Check if OpLane0 should be broadcast.
          if (shouldBroadcast(OpLane0, OpIdx, FirstLane) ||
              !canBeVectorized(OpILane0, OpIdx, FirstLane))
            ReorderingModes[OpIdx] = ReorderingMode::Splat;
          else if (isa<LoadInst>(OpILane0))
            ReorderingModes[OpIdx] = ReorderingMode::Load;
          else
            ReorderingModes[OpIdx] = ReorderingMode::Opcode;
        } else if (isa<Constant>(OpLane0)) {
          ReorderingModes[OpIdx] = ReorderingMode::Constant;
        } else if (isa<Argument>(OpLane0)) {
          // Our best hope is a Splat. It may save some cost in some cases.
          ReorderingModes[OpIdx] = ReorderingMode::Splat;
        } else {
          llvm_unreachable("Unexpected value kind.");
        }
      }

      // Check that we don't have same operands. No need to reorder if operands
      // are just perfect diamond or shuffled diamond match. Do not do it only
      // for possible broadcasts.
      auto &&SkipReordering = [this]() {
        SmallPtrSet<Value *, 4> UniqueValues;
        ArrayRef<OperandData> Op0 = OpsVec.front();
        for (const OperandData &Data : Op0)
          UniqueValues.insert(Data.V);
        for (ArrayRef<OperandData> Op :
             ArrayRef(OpsVec).slice(1, getNumOperands() - 1)) {
          if (any_of(Op, [&UniqueValues](const OperandData &Data) {
                return !UniqueValues.contains(Data.V);
              }))
            return false;
        }
        return UniqueValues.size() != 2;
      };

      // If the initial strategy fails for any of the operand indexes, then we
      // perform reordering again in a second pass. This helps avoid assigning
      // high priority to the failed strategy, and should improve reordering for
      // the non-failed operand indexes.
      for (int Pass = 0; Pass != 2; ++Pass) {
        // Check if no need to reorder operands since they're are perfect or
        // shuffled diamond match.
        // Need to do it to avoid extra external use cost counting for
        // shuffled matches, which may cause regressions.
        if (SkipReordering())
          break;
        // Skip the second pass if the first pass did not fail.
        bool StrategyFailed = false;
        // Mark all operand data as free to use.
        clearUsed();
        // We keep the original operand order for the FirstLane, so reorder the
        // rest of the lanes. We are visiting the nodes in a circular fashion,
        // using FirstLane as the center point and increasing the radius
        // distance.
        SmallVector<SmallVector<Value *, 2>> MainAltOps(NumOperands);
        for (unsigned I = 0; I < NumOperands; ++I)
          MainAltOps[I].push_back(getData(I, FirstLane).V);

        SmallBitVector UsedLanes(NumLanes);
        UsedLanes.set(FirstLane);
        for (unsigned Distance = 1; Distance != NumLanes; ++Distance) {
          // Visit the lane on the right and then the lane on the left.
          for (int Direction : {+1, -1}) {
            int Lane = FirstLane + Direction * Distance;
            if (Lane < 0 || Lane >= (int)NumLanes)
              continue;
            UsedLanes.set(Lane);
            int LastLane = Lane - Direction;
            assert(LastLane >= 0 && LastLane < (int)NumLanes &&
                   "Out of bounds");
            // Look for a good match for each operand.
            for (unsigned OpIdx = 0; OpIdx != NumOperands; ++OpIdx) {
              // Search for the operand that matches SortedOps[OpIdx][Lane-1].
              std::optional<unsigned> BestIdx =
                  getBestOperand(OpIdx, Lane, LastLane, ReorderingModes,
                                 MainAltOps[OpIdx], UsedLanes);
              // By not selecting a value, we allow the operands that follow to
              // select a better matching value. We will get a non-null value in
              // the next run of getBestOperand().
              if (BestIdx) {
                // Swap the current operand with the one returned by
                // getBestOperand().
                swap(OpIdx, *BestIdx, Lane);
              } else {
                // Enable the second pass.
                StrategyFailed = true;
              }
              // Try to get the alternate opcode and follow it during analysis.
              if (MainAltOps[OpIdx].size() != 2) {
                OperandData &AltOp = getData(OpIdx, Lane);
                InstructionsState OpS =
                    getSameOpcode({MainAltOps[OpIdx].front(), AltOp.V}, TLI);
                if (OpS && OpS.isAltShuffle())
                  MainAltOps[OpIdx].push_back(AltOp.V);
              }
            }
          }
        }
        // Skip second pass if the strategy did not fail.
        if (!StrategyFailed)
          break;
      }
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    LLVM_DUMP_METHOD static StringRef getModeStr(ReorderingMode RMode) {
      switch (RMode) {
      case ReorderingMode::Load:
        return "Load";
      case ReorderingMode::Opcode:
        return "Opcode";
      case ReorderingMode::Constant:
        return "Constant";
      case ReorderingMode::Splat:
        return "Splat";
      case ReorderingMode::Failed:
        return "Failed";
      }
      llvm_unreachable("Unimplemented Reordering Type");
    }

    LLVM_DUMP_METHOD static raw_ostream &printMode(ReorderingMode RMode,
                                                   raw_ostream &OS) {
      return OS << getModeStr(RMode);
    }

    /// Debug print.
    LLVM_DUMP_METHOD static void dumpMode(ReorderingMode RMode) {
      printMode(RMode, dbgs());
    }

    friend raw_ostream &operator<<(raw_ostream &OS, ReorderingMode RMode) {
      return printMode(RMode, OS);
    }

    LLVM_DUMP_METHOD raw_ostream &print(raw_ostream &OS) const {
      const unsigned Indent = 2;
      unsigned Cnt = 0;
      for (const OperandDataVec &OpDataVec : OpsVec) {
        OS << "Operand " << Cnt++ << "\n";
        for (const OperandData &OpData : OpDataVec) {
          OS.indent(Indent) << "{";
          if (Value *V = OpData.V)
            OS << *V;
          else
            OS << "null";
          OS << ", APO:" << OpData.APO << "}\n";
        }
        OS << "\n";
      }
      return OS;
    }

    /// Debug print.
    LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif
  };

  /// Evaluate each pair in \p Candidates and return index into \p Candidates
  /// for a pair which have highest score deemed to have best chance to form
  /// root of profitable tree to vectorize. Return std::nullopt if no candidate
  /// scored above the LookAheadHeuristics::ScoreFail. \param Limit Lower limit
  /// of the cost, considered to be good enough score.
  std::pair<std::optional<int>, int>
  findBestRootPair(ArrayRef<std::pair<Value *, Value *>> Candidates,
                   int Limit = LookAheadHeuristics::ScoreFail) const;

  /// Checks if the instruction is marked for deletion.
  bool isDeleted(Instruction *I) const { return DeletedInstructions.count(I); }

  /// Checks if the value is used only by the assume-like intrinsics.
  bool isEphemeralValue(const Value *V) const { return EphValues.contains(V); }

  /// Removes an instruction from its block and eventually deletes it.
  /// It's like Instruction::eraseFromParent() except that the actual deletion
  /// is delayed until BoUpSLP is destructed.
  void eraseInstruction(Instruction *I) { DeletedInstructions.insert(I); }

  /// Remove instructions from the parent function and clear the operands of \p
  /// DeadVals instructions, marking for deletion trivially dead operands.
  template <typename T>
  void removeInstructionsAndOperands(
      ArrayRef<T *> DeadVals,
      ArrayRef<ReductionVectorPart> VectorValuesAndScales) {
    SmallVector<WeakTrackingVH> DeadInsts;
    for (T *V : DeadVals) {
      auto *I = cast<Instruction>(V);
      eraseInstruction(I);
    }
    DenseSet<Value *> Processed;
    for (T *V : DeadVals) {
      if (!V || !Processed.insert(V).second)
        continue;
      auto *I = cast<Instruction>(V);
      salvageDebugInfo(*I);
      ArrayRef<TreeEntry *> Entries = getTreeEntries(I);
      for (Use &U : I->operands()) {
        if (auto *OpI = dyn_cast_if_present<Instruction>(U.get());
            OpI && !DeletedInstructions.contains(OpI) && OpI->hasOneUser() &&
            wouldInstructionBeTriviallyDead(OpI, TLI) &&
            !ExternalUseReplacements.contains(OpI) &&
            (Entries.empty() || none_of(Entries, [&](const TreeEntry *Entry) {
               return Entry->VectorizedValue == OpI;
             })))
          DeadInsts.push_back(OpI);
      }
      I->dropAllReferences();
    }
    for (T *V : DeadVals) {
      auto *I = cast<Instruction>(V);
      if (!I->getParent())
        continue;
      assert((I->use_empty() || all_of(I->uses(),
                                       [&](Use &U) {
                                         return isDeleted(
                                             cast<Instruction>(U.getUser()));
                                       })) &&
             "trying to erase instruction with users.");
      I->removeFromParent();
      SE->forgetValue(I);
    }
    // Process the dead instruction list until empty.
    while (!DeadInsts.empty()) {
      Value *V = DeadInsts.pop_back_val();
      Instruction *VI = cast_or_null<Instruction>(V);
      if (!VI || !VI->getParent())
        continue;
      assert(isInstructionTriviallyDead(VI, TLI) &&
             "Live instruction found in dead worklist!");
      assert(VI->use_empty() && "Instructions with uses are not dead.");

      // Don't lose the debug info while deleting the instructions.
      salvageDebugInfo(*VI);

      // Null out all of the instruction's operands to see if any operand
      // becomes dead as we go.
      for (Use &OpU : VI->operands()) {
        Value *OpV = OpU.get();
        if (!OpV)
          continue;
        OpU.set(nullptr);

        if (!OpV->use_empty())
          continue;

        // If the operand is an instruction that became dead as we nulled out
        // the operand, and if it is 'trivially' dead, delete it in a future
        // loop iteration.
        if (auto *OpI = dyn_cast<Instruction>(OpV))
          if (!DeletedInstructions.contains(OpI) &&
              !ExternalUseReplacements.contains(OpI) &&
              (!OpI->getType()->isVectorTy() ||
               none_of(VectorValuesAndScales,
                       [&](const ReductionVectorPart &V) {
                         return V.Vec == OpI;
                       })) &&
              isInstructionTriviallyDead(OpI, TLI))
            DeadInsts.push_back(OpI);
      }

      VI->removeFromParent();
      eraseInstruction(VI);
      SE->forgetValue(VI);
    }
  }

  /// Checks if the instruction was already analyzed for being possible
  /// reduction root.
  bool isAnalyzedReductionRoot(Instruction *I) const {
    return AnalyzedReductionsRoots.count(I);
  }
  /// Register given instruction as already analyzed for being possible
  /// reduction root.
  void analyzedReductionRoot(Instruction *I) {
    AnalyzedReductionsRoots.insert(I);
  }
  /// Checks if the provided list of reduced values was checked already for
  /// vectorization.
  bool areAnalyzedReductionVals(ArrayRef<Value *> VL) const {
    return AnalyzedReductionVals.contains(hash_value(VL));
  }
  /// Adds the list of reduced values to list of already checked values for the
  /// vectorization.
  void analyzedReductionVals(ArrayRef<Value *> VL) {
    AnalyzedReductionVals.insert(hash_value(VL));
  }
  /// Checks if the value was already a part of the analyzed vector node.
  bool isAnalyzedScalar(const Value *V) const {
    return AnalyzedScalars.contains(V);
  }
  /// Checks if the given bundle was already rejected as non-vectorizable.
  bool isAnalyzedBundle(ArrayRef<Value *> VL) const {
    return AnalyzedBundles.contains(hash_value(VL));
  }
  /// Registers the bundle as rejected for the vectorization.
  void analyzedBundle(ArrayRef<Value *> VL) {
    AnalyzedBundles.insert(hash_value(VL));
  }
  /// Clear the list of the analyzed reduction root instructions.
  void clearReductionData() {
    AnalyzedReductionsRoots.clear();
    AnalyzedReductionVals.clear();
    AnalyzedBundles.clear();
    AnalyzedMinBWVals.clear();
  }
  /// Checks if the given value is gathered in one of the nodes.
  bool isAnyGathered(const SmallDenseSet<Value *> &Vals) const {
    return any_of(MustGather, [&](Value *V) { return Vals.contains(V); });
  }
  /// Checks if the given value is gathered in one of the nodes.
  bool isGathered(const Value *V) const { return MustGather.contains(V); }
  /// Checks if the specified value was not schedule.
  bool isNotScheduled(const Value *V) const {
    return NonScheduledFirst.contains(V);
  }

  /// Check if \p V is a peeled reassociated scalar still owned by a live
  /// (non-deleted, non-gathered) tree entry.
  bool isReassocScalarVectorized(const Value *V) const {
    auto It = ReassocScalarToTreeEntries.find(V);
    return It != ReassocScalarToTreeEntries.end() &&
           any_of(It->second, [&](const TreeEntry *E) {
             return !DeletedNodes.contains(E) &&
                    !TransformedToGatherNodes.contains(E);
           });
  }

  /// Check if the value is vectorized in the tree.
  bool isVectorized(const Value *V) const {
    assert(V && "V cannot be nullptr.");
    if (isReassocScalarVectorized(V))
      return true;
    return any_of(getTreeEntries(V), [&](const TreeEntry *E) {
      return !DeletedNodes.contains(E) && !TransformedToGatherNodes.contains(E);
    });
  }

  /// Returns true if the role of \p I is already decided by its user: a deleted
  /// user was folded into some other vector by an earlier attempt.
  bool hasResolvedUser(Instruction *I) const {
    return any_of(I->users(), [&](User *U) {
      auto *UI = dyn_cast<Instruction>(U);
      return UI && isDeleted(UI);
    });
  }

  /// Checks if it is legal and profitable to build SplitVectorize node for the
  /// given \p VL.
  /// \param Op1 first homogeneous scalars.
  /// \param Op2 second homogeneous scalars.
  /// \param ReorderIndices indices to reorder the scalars.
  /// \returns true if the node was successfully built.
  bool canBuildSplitNode(ArrayRef<Value *> VL,
                         const InstructionsState &LocalState,
                         SmallVectorImpl<Value *> &Op1,
                         SmallVectorImpl<Value *> &Op2,
                         OrdersType &ReorderIndices) const;

  ~BoUpSLP();

private:
  /// Determine if a node \p E in can be demoted to a smaller type with a
  /// truncation. We collect the entries that will be demoted in ToDemote.
  /// \param E Node for analysis
  /// \param ToDemote indices of the nodes to be demoted.
  bool collectValuesToDemote(
      const TreeEntry &E, bool IsProfitableToDemoteRoot, unsigned &BitWidth,
      SmallVectorImpl<unsigned> &ToDemote, DenseSet<const TreeEntry *> &Visited,
      const SmallDenseSet<unsigned, 8> &NodesToKeepBWs, unsigned &MaxDepthLevel,
      bool &IsProfitableToDemote, bool IsTruncRoot) const;

  /// Builds the list of reorderable operands on the edges \p Edges of the \p
  /// UserTE, which allow reordering (i.e. the operands can be reordered because
  /// they have only one user and reordarable).
  /// \param ReorderableGathers List of all gather nodes that require reordering
  /// (e.g., gather of extractlements or partially vectorizable loads).
  /// \param GatherOps List of gather operand nodes for \p UserTE that require
  /// reordering, subset of \p NonVectorized.
  void buildReorderableOperands(
      TreeEntry *UserTE,
      SmallVectorImpl<std::pair<unsigned, TreeEntry *>> &Edges,
      const SmallPtrSetImpl<const TreeEntry *> &ReorderableGathers,
      SmallVectorImpl<TreeEntry *> &GatherOps);

  /// Checks if the given \p TE is a gather node with clustered reused scalars
  /// and reorders it per given \p Mask.
  void reorderNodeWithReuses(TreeEntry &TE, ArrayRef<int> Mask) const;

  /// Checks if all users of \p I are the part of the vectorization tree.
  bool areAllUsersVectorized(
      Instruction *I,
      const SmallDenseSet<Value *> *VectorizedVals = nullptr) const;

  /// Estimates the number of scalar instructions in the tree, each weighted by
  /// its loop-nest trip count (nest-invariant entries are dropped when
  /// \p TreeLoop is non-null).
  uint64_t getNumScalarInsts(bool HasTreeLoop);

  /// Estimates the number of vector instructions (including buildvectors,
  /// shuffles, and extracts) the tree produces, weighted like
  /// getNumScalarInsts().
  uint64_t getNumVectorInsts(bool HasTreeLoop);

  /// Return information about the vector formed for the specified index
  /// of a vector of (the same) instruction.
  TargetTransformInfo::OperandValueInfo
  getOperandInfo(ArrayRef<Value *> Ops) const;

  /// \returns the graph entry for the \p Idx operand of the \p E entry.
  const TreeEntry *getOperandEntry(const TreeEntry *E, unsigned Idx) const;
  TreeEntry *getOperandEntry(TreeEntry *E, unsigned Idx) {
    return const_cast<TreeEntry *>(
        getOperandEntry(const_cast<const TreeEntry *>(E), Idx));
  }

  /// Gets the root instruction for the given node. If the node is a strided
  /// load/store node with the reverse order, the root instruction is the last
  /// one.
  Instruction *getRootEntryInstruction(const TreeEntry &Entry) const;

  /// \returns Cast context for the given graph node.
  TargetTransformInfo::CastContextHint
  getCastContextHint(const TreeEntry &TE) const;

  /// \returns the scale of the given tree entry to the loop iteration.
  /// \p Scalar is the scalar value from the entry, if using the parent for the
  /// external use.
  /// \p U is the user of the vectorized value from the entry, if using the
  /// parent for the external use.
  uint64_t getScaleToLoopIterations(const TreeEntry &TE,
                                    Value *Scalar = nullptr,
                                    Instruction *U = nullptr);

  /// \returns the product of trip counts of the loop \p L and all of its
  /// enclosing loops. Unlike the state kept by getScaleToLoopIterations(),
  /// this helper depends only on the loop structure and is independent of
  /// per-entry operand invariance. Returns 1 when loop-aware cost modeling
  /// is disabled or \p L is null.
  uint64_t getLoopNestScale(const Loop *L);

  /// \returns a refined execution scale for a gather/buildvector tree entry
  /// \p TE. The scale is computed as the average of per-lane execution
  /// scales: each lane's scale is the loop-nest scale of the loop that
  /// contains the lane's defining instruction (or 1 if the lane is a
  /// constant / loop-invariant non-instruction value). This models the
  /// LICM hoisting that optimizeGatherSequence() performs after vectorization
  /// for inserts with loop-invariant operands. Falls back to the whole-entry
  /// scale when per-lane information is unavailable or the feature is off.
  uint64_t getGatherNodeEffectiveScale(const TreeEntry &TE,
                                       Instruction *U = nullptr);

  /// \returns the loop-nest execution scale of \p TE.
  uint64_t getEntryEffectiveScale(const TreeEntry &TE,
                                  Instruction *U = nullptr);

  /// Get the loop nest for the given loop \p L.
  ArrayRef<const Loop *> getLoopNest(const Loop *L);

  /// \returns the cost of the vectorizable entry.
  InstructionCost getEntryCost(const TreeEntry *E,
                               ArrayRef<Value *> VectorizedVals,
                               SmallPtrSetImpl<Value *> &CheckedExtracts);

  /// Estimates spill/reload cost from vector register pressure for \p E at the
  /// point of emitting its vector result type \p FinalVecTy. \p ScalarTy is the
  /// scalar/slot type used to widen into \p VecTy/\p FinalVecTy and may itself
  /// be a FixedVectorType in ReVec mode or an adjusted type due to MinBWs.
  InstructionCost
  getVectorSpillReloadCost(const TreeEntry *E, Type *ScalarTy, Type *VecTy,
                           Type *FinalVecTy,
                           const TTI::TargetCostKind CostKind) const;

  /// This is the recursive part of buildTree.
  void buildTreeRec(ArrayRef<Value *> Roots, unsigned Depth, const EdgeInfo &EI,
                    unsigned InterleaveFactor = 0);

  /// \returns true if the ExtractElement/ExtractValue instructions in \p VL can
  /// be vectorized to use the original vector (or aggregate "bitcast" to a
  /// vector) and sets \p CurrentOrder to the identity permutation; otherwise
  /// returns false, setting \p CurrentOrder to either an empty vector or a
  /// non-identity permutation that allows to reuse extract instructions.
  /// \param ResizeAllowed indicates whether it is allowed to handle subvector
  /// extract order.
  bool canReuseExtract(ArrayRef<Value *> VL,
                       SmallVectorImpl<unsigned> &CurrentOrder,
                       bool ResizeAllowed = false) const;

  /// Vectorize a single entry in the tree.
  Value *vectorizeTree(TreeEntry *E);

  /// Vectorize a single entry in the tree, the \p Idx-th operand of the entry
  /// \p E.
  Value *vectorizeOperand(TreeEntry *E, unsigned NodeIdx);

  /// Create a new vector from a list of scalar values.  Produces a sequence
  /// which exploits values reused across lanes, and arranges the inserts
  /// for ease of later optimization.
  template <typename BVTy, typename ResTy, typename... Args>
  ResTy processBuildVector(const TreeEntry *E, Type *ScalarTy, Args &...Params);

  /// Create a new vector from a list of scalar values.  Produces a sequence
  /// which exploits values reused across lanes, and arranges the inserts
  /// for ease of later optimization.
  Value *createBuildVector(const TreeEntry *E, Type *ScalarTy);

  /// Returns the instruction in the bundle, which can be used as a base point
  /// for scheduling. Usually it is the last instruction in the bundle, except
  /// for the case when all operands are external (in this case, it is the first
  /// instruction in the list).
  Instruction &getLastInstructionInBundle(const TreeEntry *E);

  /// Tries to find extractelement instructions with constant indices from fixed
  /// vector type and gather such instructions into a bunch, which highly likely
  /// might be detected as a shuffle of 1 or 2 input vectors. If this attempt
  /// was successful, the matched scalars are replaced by poison values in \p VL
  /// for future analysis.
  std::optional<TargetTransformInfo::ShuffleKind>
  tryToGatherSingleRegisterExtractElements(MutableArrayRef<Value *> VL,
                                           SmallVectorImpl<int> &Mask) const;

  /// Tries to find extractelement instructions with constant indices from fixed
  /// vector type and gather such instructions into a bunch, which highly likely
  /// might be detected as a shuffle of 1 or 2 input vectors. If this attempt
  /// was successful, the matched scalars are replaced by poison values in \p VL
  /// for future analysis.
  SmallVector<std::optional<TargetTransformInfo::ShuffleKind>>
  tryToGatherExtractElements(SmallVectorImpl<Value *> &VL,
                             SmallVectorImpl<int> &Mask,
                             unsigned NumParts) const;

  /// Checks if the gathered \p VL can be represented as a single register
  /// shuffle(s) of previous tree entries.
  /// \param TE Tree entry checked for permutation.
  /// \param VL List of scalars (a subset of the TE scalar), checked for
  /// permutations. Must form single-register vector.
  /// \param ForOrder Tries to fetch the best candidates for ordering info. Also
  /// commands to build the mask using the original vector value, without
  /// relying on the potential reordering.
  /// \returns ShuffleKind, if gathered values can be represented as shuffles of
  /// previous tree entries. \p Part of \p Mask is filled with the shuffle mask.
  std::optional<TargetTransformInfo::ShuffleKind>
  isGatherShuffledSingleRegisterEntry(
      const TreeEntry *TE, ArrayRef<Value *> VL, MutableArrayRef<int> Mask,
      SmallVectorImpl<const TreeEntry *> &Entries, unsigned Part, bool ForOrder,
      unsigned SliceSize);

  /// Checks if the gathered \p VL can be represented as multi-register
  /// shuffle(s) of previous tree entries.
  /// \param TE Tree entry checked for permutation.
  /// \param VL List of scalars (a subset of the TE scalar), checked for
  /// permutations.
  /// \param ForOrder Tries to fetch the best candidates for ordering info. Also
  /// commands to build the mask using the original vector value, without
  /// relying on the potential reordering.
  /// \returns per-register series of ShuffleKind, if gathered values can be
  /// represented as shuffles of previous tree entries. \p Mask is filled with
  /// the shuffle mask (also on per-register base).
  SmallVector<std::optional<TargetTransformInfo::ShuffleKind>>
  isGatherShuffledEntry(
      const TreeEntry *TE, ArrayRef<Value *> VL, SmallVectorImpl<int> &Mask,
      SmallVectorImpl<SmallVector<const TreeEntry *>> &Entries,
      unsigned NumParts, bool ForOrder = false);

  /// \returns the cost of gathering (inserting) the values in \p VL into a
  /// vector.
  /// \param ForPoisonSrc true if initial vector is poison, false otherwise.
  InstructionCost getGatherCost(ArrayRef<Value *> VL, bool ForPoisonSrc,
                                Type *ScalarTy) const;

  /// Set the Builder insert point to one after the last instruction in
  /// the bundle
  void setInsertPointAfterBundle(const TreeEntry *E);

  /// \returns a vector from a collection of scalars in \p VL. if \p Root is not
  /// specified, the starting vector value is poison.
  Value *
  gather(ArrayRef<Value *> VL, Value *Root, Type *ScalarTy,
         function_ref<Value *(Value *, Value *, ArrayRef<int>)> CreateShuffle);

  /// \returns whether the VectorizableTree is fully vectorizable and will
  /// be beneficial even the tree height is tiny.
  bool isFullyVectorizableTinyTree(bool ForReduction) const;

  /// Run through the list of all gathered loads in the graph and try to find
  /// vector loads/masked gathers instead of regular gathers. Later these loads
  /// are reshufled to build final gathered nodes.
  void tryToVectorizeGatheredLoads(
      const SmallMapVector<
          std::tuple<BasicBlock *, Value *, Type *>,
          SmallVector<SmallVector<std::pair<LoadInst *, int64_t>>>, 8>
          &GatheredLoads);

  /// Run through the gather nodes that are splats of the same instruction and
  /// try to vectorize the unique splatted values together as a separate
  /// subtree. The splat gathers are then emitted as broadcasts of the
  /// vectorized subtree instead of insertion sequences.
  void tryToVectorizeSplatGatheredScalars();

  /// Helper for `findExternalStoreUsersReorderIndices()`. It iterates over the
  /// users of \p TE and collects the stores. It returns the map from the store
  /// pointers to the collected stores.
  SmallVector<SmallVector<StoreInst *>>
  collectUserStores(const BoUpSLP::TreeEntry *TE) const;

  /// Helper for `findExternalStoreUsersReorderIndices()`. It checks if the
  /// stores in \p StoresVec can form a vector instruction. If so it returns
  /// true and populates \p ReorderIndices with the shuffle indices of the
  /// stores when compared to the sorted vector.
  bool canFormVector(ArrayRef<StoreInst *> StoresVec,
                     OrdersType &ReorderIndices) const;

  /// Iterates through the users of \p TE, looking for scalar stores that can be
  /// potentially vectorized in a future SLP-tree. If found, it keeps track of
  /// their order and builds an order index vector for each store bundle. It
  /// returns all these order vectors found.
  /// We run this after the tree has formed, otherwise we may come across user
  /// instructions that are not yet in the tree.
  SmallVector<OrdersType, 1>
  findExternalStoreUsersReorderIndices(TreeEntry *TE) const;

  /// Tries to reorder the gathering node for better vectorization
  /// opportunities.
  void reorderGatherNode(TreeEntry &TE);

  /// Checks if the tree represents disjoint or reduction of shl(zext, (0, 8,
  /// .., 56))-like pattern.
  /// If the int shifts unique, also strided, but not ordered, sets \p Order.
  /// If the node can be represented as a bitcast + bswap, sets \p IsBSwap.
  /// If the root nodes are loads, sets \p ForLoads to true.
  bool matchesShlZExt(const TreeEntry &TE, OrdersType &Order, bool &IsBSwap,
                      bool &ForLoads) const;

  /// Checks if the \p SelectTE matches zext+selects, which can be inversed for
  /// better codegen in case like zext (icmp ne), select (icmp eq), ....
  bool matchesInversedZExtSelect(
      const TreeEntry &SelectTE,
      SmallVectorImpl<unsigned> &InversedCmpsIndices) const;

  /// Checks if the tree is reduction or of bit selects, like select %cmp, <1,
  /// 2, 4, 8, ..>, zeroinitializer, which can be reduced just to a bitcast %cmp
  /// to in.
  bool matchesSelectOfBits(const TreeEntry &SelectTE) const;

  class TreeEntry {
  public:
    using VecTreeTy = SmallVector<std::unique_ptr<TreeEntry>, 8>;
    TreeEntry(VecTreeTy &Container) : Container(Container) {}

    /// \returns Common mask for reorder indices and reused scalars.
    SmallVector<int> getCommonMask() const {
      if (State == TreeEntry::SplitVectorize)
        return {};
      SmallVector<int> Mask;
      inversePermutation(ReorderIndices, Mask);
      addMask(Mask, ReuseShuffleIndices);
      return Mask;
    }

    /// \returns The mask for split nodes.
    SmallVector<int> getSplitMask() const {
      assert(State == TreeEntry::SplitVectorize && !ReorderIndices.empty() &&
             "Expected only split vectorize node.");
      unsigned CommonVF = std::max<unsigned>(
          CombinedEntriesWithIndices.back().second,
          Scalars.size() - CombinedEntriesWithIndices.back().second);
      const unsigned Scale = getNumElements(Scalars.front()->getType());
      CommonVF *= Scale;
      SmallVector<int> Mask(getVectorFactor() * Scale, PoisonMaskElem);
      for (auto [Idx, I] : enumerate(ReorderIndices)) {
        for (unsigned K : seq<unsigned>(Scale)) {
          Mask[Scale * I + K] =
              Scale * Idx + K +
              (Idx >= CombinedEntriesWithIndices.back().second
                   ? CommonVF - CombinedEntriesWithIndices.back().second * Scale
                   : 0);
        }
      }
      return Mask;
    }

    /// Updates (reorders) SplitVectorize node according to the given mask \p
    /// Mask and order \p MaskOrder.
    void reorderSplitNode(unsigned Idx, ArrayRef<int> Mask,
                          ArrayRef<int> MaskOrder);

    /// \returns true if the scalars in VL are equal to this entry.
    bool isSame(ArrayRef<Value *> VL) const {
      auto &&IsSame = [VL](ArrayRef<Value *> Scalars, ArrayRef<int> Mask) {
        if (Mask.size() != VL.size() && VL.size() == Scalars.size())
          return std::equal(VL.begin(), VL.end(), Scalars.begin());
        return VL.size() == Mask.size() &&
               std::equal(VL.begin(), VL.end(), Mask.begin(),
                          [Scalars](Value *V, int Idx) {
                            return isa<PoisonValue>(V) ||
                                   (Idx != PoisonMaskElem && V == Scalars[Idx]);
                          });
      };
      if (!ReorderIndices.empty()) {
        // TODO: implement matching if the nodes are just reordered, still can
        // treat the vector as the same if the list of scalars matches VL
        // directly, without reordering.
        SmallVector<int> Mask;
        inversePermutation(ReorderIndices, Mask);
        if (VL.size() == Scalars.size())
          return IsSame(Scalars, Mask);
        if (VL.size() == ReuseShuffleIndices.size()) {
          addMask(Mask, ReuseShuffleIndices);
          return IsSame(Scalars, Mask);
        }
        return false;
      }
      return IsSame(Scalars, ReuseShuffleIndices);
    }

    /// \returns true if current entry has same operands as \p TE.
    bool hasEqualOperands(const TreeEntry &TE) const {
      if (TE.getNumOperands() != getNumOperands())
        return false;
      SmallBitVector Used(getNumOperands());
      for (unsigned I = 0, E = getNumOperands(); I < E; ++I) {
        unsigned PrevCount = Used.count();
        for (unsigned K = 0; K < E; ++K) {
          if (Used.test(K))
            continue;
          if (getOperand(K) == TE.getOperand(I)) {
            Used.set(K);
            break;
          }
        }
        // Check if we actually found the matching operand.
        if (PrevCount == Used.count())
          return false;
      }
      return true;
    }

    /// \return Final vectorization factor for the node. Defined by the total
    /// number of vectorized scalars, including those, used several times in the
    /// entry and counted in the \a ReuseShuffleIndices, if any.
    unsigned getVectorFactor() const {
      if (!ReuseShuffleIndices.empty())
        return ReuseShuffleIndices.size();
      return Scalars.size();
    };

    /// Checks if the current node is a gather node.
    bool isGather() const { return State == NeedToGather; }

    /// A vector of scalars.
    ValueList Scalars;

    /// The Scalars are vectorized into this value. It is initialized to Null.
    WeakTrackingVH VectorizedValue = nullptr;

    /// Do we need to gather this sequence or vectorize it
    /// (either with vector instruction or with scatter/gather
    /// intrinsics for store/load)?
    enum EntryState {
      Vectorize,            ///< The node is regularly vectorized.
      ScatterVectorize,     ///< Masked scatter/gather node.
      StridedVectorize,     ///< Strided loads (and stores)
      ExpandVectorize,      ///< Masked stores, the values are expanded into
                            ///< a wider vector and vectorized with a mask.
      CompressVectorize,    ///< (Masked) load with compress.
      BlendedLoadVectorize, ///< (Masked) loads blended via `select` from two
                            ///< candidate base pointers.
      NeedToGather,         ///< Gather/buildvector node.
      CombinedVectorize, ///< Vectorized node, combined with its user into more
                         ///< complex node like select/cmp to minmax, mul/add to
                         ///< fma, etc. Must be used for the following nodes in
                         ///< the pattern, not the very first one.
      SplitVectorize,    ///< Splits the node into 2 subnodes, vectorizes them
                         ///< independently and then combines back.
    };
    EntryState State;

    /// List of combined opcodes supported by the vectorizer.
    enum CombinedOpcode {
      NotCombinedOp = -1,
      MinMax = Instruction::OtherOpsEnd + 1,
      FMulAdd,
      ReducedBitcast,
      ReducedBitcastBSwap,
      ReducedBitcastLoads,
      ReducedBitcastBSwapLoads,
      ReducedCmpBitcast,
    };
    CombinedOpcode CombinedOp = NotCombinedOp;

    /// Does this sequence require some shuffling?
    SmallVector<int, 4> ReuseShuffleIndices;

    /// Does this entry require reordering?
    SmallVector<unsigned, 4> ReorderIndices;

    /// Points back to the VectorizableTree.
    ///
    /// Only used for Graphviz right now.  Unfortunately GraphTrait::NodeRef has
    /// to be a pointer and needs to be able to initialize the child iterator.
    /// Thus we need a reference back to the container to translate the indices
    /// to entries.
    VecTreeTy &Container;

    /// The TreeEntry index containing the user of this entry.
    EdgeInfo UserTreeIndex;

    /// The index of this treeEntry in VectorizableTree.
    unsigned Idx = 0;

    /// For gather/buildvector/alt opcode nodes, which are combined from
    /// other nodes as a series of insertvector instructions.
    SmallVector<std::pair<unsigned, unsigned>, 2> CombinedEntriesWithIndices;

    /// For ExtractValue entries that are vectorized via the struct-call path
    /// (checkEVsForVecCalls succeeded during tree building), stores the common
    /// field-index path shared by all scalars in the bundle. Empty for all
    /// other entry kinds.
    SmallVector<unsigned, 1> StructEVIndices;

  private:
    /// The operands of each instruction in each lane Operands[op_index][lane].
    /// Note: This helps avoid the replication of the code that performs the
    /// reordering of operands during buildTreeRec() and vectorizeTree().
    SmallVector<ValueList, 2> Operands;

    /// Copyable elements of the entry node.
    SmallPtrSet<const Value *, 4> CopyableElements;

    /// Intermediate instructions peeled from an associative chain (e.g. the
    /// inner add in add(add(v0,x),v1)). Not part of Scalars.
    SmallVector<Value *, 4> ReassocScalars;

    /// Sign of each flattened operand column of a reassociated add/sub
    /// chain, parallel to the operand columns: a negated column is
    /// subtracted from the positive total. Empty when no column is negated.
    SmallBitVector ReassocNegatedOps;

    /// MainOp and AltOp are recorded inside. S should be obtained from
    /// newTreeEntry.
    InstructionsState S = InstructionsState::invalid();

    /// Interleaving factor for interleaved loads Vectorize nodes.
    unsigned InterleaveFactor = 0;

    /// True if the node does not require scheduling.
    bool DoesNotNeedToSchedule = false;

    /// Set this bundle's \p OpIdx'th operand to \p OpVL.
    void setOperand(unsigned OpIdx, ArrayRef<Value *> OpVL) {
      if (Operands.size() < OpIdx + 1)
        Operands.resize(OpIdx + 1);
      assert(Operands[OpIdx].empty() && "Already resized?");
      assert(OpVL.size() <= Scalars.size() &&
             "Number of operands is greater than the number of scalars.");
      Operands[OpIdx].resize(OpVL.size());
      copy(OpVL, Operands[OpIdx].begin());
    }

    /// Maps values to their lanes in the node.
    mutable SmallDenseMap<Value *, unsigned> ValueToLane;

  public:
    /// Returns interleave factor for interleave nodes.
    unsigned getInterleaveFactor() const { return InterleaveFactor; }
    /// Sets interleaving factor for the interleaving nodes.
    void setInterleave(unsigned Factor) { InterleaveFactor = Factor; }

    /// Marks the node as one that does not require scheduling.
    void setDoesNotNeedToSchedule() { DoesNotNeedToSchedule = true; }
    /// Returns true if the node is marked as one that does not require
    /// scheduling.
    bool doesNotNeedToSchedule() const { return DoesNotNeedToSchedule; }

    /// Set this bundle's operands from \p Operands.
    void setOperands(ArrayRef<ValueList> Operands) {
      for (unsigned I : seq<unsigned>(Operands.size()))
        setOperand(I, Operands[I]);
    }

    /// Reorders operands of the node to the given mask \p Mask.
    void reorderOperands(ArrayRef<int> Mask) {
      for (ValueList &Operand : Operands)
        reorderScalars(Operand, Mask);
    }

    /// \returns the \p OpIdx operand of this TreeEntry.
    ValueList &getOperand(unsigned OpIdx) {
      assert(OpIdx < Operands.size() && "Off bounds");
      return Operands[OpIdx];
    }

    /// \returns the \p OpIdx operand of this TreeEntry.
    ArrayRef<Value *> getOperand(unsigned OpIdx) const {
      assert(OpIdx < Operands.size() && "Off bounds");
      return Operands[OpIdx];
    }

    /// \returns the number of operands.
    unsigned getNumOperands() const { return Operands.size(); }

    /// \return the single \p OpIdx operand.
    Value *getSingleOperand(unsigned OpIdx) const {
      assert(OpIdx < Operands.size() && "Off bounds");
      assert(!Operands[OpIdx].empty() && "No operand available");
      return Operands[OpIdx][0];
    }

    /// Some of the instructions in the list have alternate opcodes.
    bool isAltShuffle() const { return S.isAltShuffle(); }

    Instruction *getMatchingMainOpOrAltOp(Instruction *I) const {
      return S.getMatchingMainOpOrAltOp(I);
    }

    /// Chooses the correct key for scheduling data. If \p Op has the same (or
    /// alternate) opcode as \p OpValue, the key is \p Op. Otherwise the key is
    /// \p OpValue.
    Value *isOneOf(Value *Op) const {
      auto *I = dyn_cast<Instruction>(Op);
      if (I && getMatchingMainOpOrAltOp(I))
        return Op;
      return S.getMainOp();
    }

    void setOperations(const InstructionsState &S) {
      assert(S && "InstructionsState is invalid.");
      this->S = S;
    }

    Instruction *getMainOp() const { return S.getMainOp(); }

    Instruction *getAltOp() const { return S.getAltOp(); }

    /// The main/alternate opcodes for the list of instructions.
    unsigned getOpcode() const { return S.getOpcode(); }

    unsigned getAltOpcode() const { return S.getAltOpcode(); }

    bool hasState() const { return S.valid(); }

    /// Add \p V to the list of copyable elements.
    void addCopyableElement(Value *V) {
      assert(S.isCopyableElement(V) && "Not a copyable element.");
      CopyableElements.insert(V);
    }

    /// Returns true if \p V is a copyable element.
    bool isCopyableElement(Value *V) const {
      return CopyableElements.contains(V);
    }

    /// Checks if the value \p V is a transformed instruction, compatible either
    /// with main or alternate ops.
    bool isExpandedBinOp(Value *V) const {
      assert(hasState() && "InstructionsState is invalid.");
      if (isCopyableElement(V))
        return false;
      return S.isExpandedBinOp(V);
    }

    /// Checks if the operand at index \p Idx of instruction \p I is an expanded
    /// operand.
    bool isExpandedOperand(Instruction *I, unsigned Idx) const {
      assert(hasState() && "InstructionsState is invalid.");
      if (isCopyableElement(I))
        return false;
      if (!isExpandedBinOp(I))
        return false;
      return S.isExpandedOperand(I, Idx);
    }

    /// Returns true if any scalar in the list is a copyable element.
    bool hasCopyableElements() const { return !CopyableElements.empty(); }

    /// Adds \p V to the peeled reassociated scalars.
    void addReassocScalar(Value *V) { ReassocScalars.push_back(V); }

    /// True if operands were gathered from an associative chain.
    bool hasReassocScalars() const { return !ReassocScalars.empty(); }

    /// Returns peeled reassociated scalars.
    ArrayRef<Value *> getReassocScalars() const { return ReassocScalars; }

    /// Records the signs of the flattened operand columns.
    void setReassocNegatedOps(const SmallBitVector &NegatedOps) {
      assert(NegatedOps.size() == getNumOperands() &&
             "Signs must cover all operand columns.");
      ReassocNegatedOps = NegatedOps;
    }

    /// True if operand column \p Idx is subtracted rather than added.
    bool isReassocNegatedOp(unsigned Idx) const {
      return Idx < ReassocNegatedOps.size() && ReassocNegatedOps[Idx];
    }

    /// Returns the state of the operations.
    const InstructionsState &getOperations() const { return S; }

    /// When ReuseReorderShuffleIndices is empty it just returns position of \p
    /// V within vector of Scalars. Otherwise, try to remap on its reuse index.
    unsigned findLaneForValue(Value *V) const {
      auto Res = ValueToLane.try_emplace(V, getVectorFactor());
      if (!Res.second)
        return Res.first->second;
      unsigned &FoundLane = Res.first->getSecond();
      // Poison can take any lane, match it to the lane of the first non-poison
      // scalar.
      auto IsMatch = [V](Value *S) {
        return isa<PoisonValue>(V) ? !isa<PoisonValue>(S) : S == V;
      };
      for (auto *It = find_if(Scalars, IsMatch), *End = Scalars.end();
           It != End; std::advance(It, 1)) {
        if (!IsMatch(*It))
          continue;
        FoundLane = std::distance(Scalars.begin(), It);
        assert(FoundLane < Scalars.size() && "Couldn't find extract lane");
        if (!ReorderIndices.empty())
          FoundLane = ReorderIndices[FoundLane];
        assert(FoundLane < Scalars.size() && "Couldn't find extract lane");
        if (ReuseShuffleIndices.empty())
          break;
        if (auto *RIt = find(ReuseShuffleIndices, FoundLane);
            RIt != ReuseShuffleIndices.end()) {
          FoundLane = std::distance(ReuseShuffleIndices.begin(), RIt);
          break;
        }
      }
      assert(FoundLane < getVectorFactor() && "Unable to find given value.");
      return FoundLane;
    }

    /// Build a shuffle mask for graph entry which represents a merge of main
    /// and alternate operations.
    void
    buildAltOpShuffleMask(const function_ref<bool(Instruction *)> IsAltOp,
                          SmallVectorImpl<int> &Mask,
                          SmallVectorImpl<Value *> *OpScalars = nullptr,
                          SmallVectorImpl<Value *> *AltScalars = nullptr) const;

    /// Return true if this is a non-power-of-2 node.
    bool isNonPowOf2Vec() const {
      bool IsNonPowerOf2 = !has_single_bit(Scalars.size());
      return IsNonPowerOf2;
    }

    Value *getOrdered(unsigned Idx) const {
      if (ReorderIndices.empty())
        return Scalars[Idx];
      SmallVector<int> Mask;
      inversePermutation(ReorderIndices, Mask);
      return Scalars[Mask[Idx]];
    }

#ifndef NDEBUG
    /// Debug printer.
    LLVM_DUMP_METHOD void dump() const {
      dbgs() << Idx << ".\n";
      for (unsigned OpI = 0, OpE = Operands.size(); OpI != OpE; ++OpI) {
        dbgs() << "Operand " << OpI << ":\n";
        for (const Value *V : Operands[OpI])
          dbgs().indent(2) << *V << "\n";
      }
      dbgs() << "Scalars: \n";
      for (Value *V : Scalars) {
        dbgs().indent(2) << *V
                         << ((S && S.isExpandedBinOp(V)) ? " [[Expanded]]\n"
                                                         : "\n");
      }
      dbgs() << "State: ";
      if (S && hasCopyableElements())
        dbgs() << "[[Copyable]] ";
      switch (State) {
      case Vectorize:
        if (InterleaveFactor > 0) {
          dbgs() << "Vectorize with interleave factor " << InterleaveFactor
                 << "\n";
        } else {
          dbgs() << "Vectorize\n";
        }
        break;
      case ScatterVectorize:
        dbgs() << "ScatterVectorize\n";
        break;
      case StridedVectorize:
        dbgs() << "StridedVectorize\n";
        break;
      case ExpandVectorize:
        dbgs() << "ExpandVectorize\n";
        break;
      case CompressVectorize:
        dbgs() << "CompressVectorize\n";
        break;
      case BlendedLoadVectorize:
        dbgs() << "BlendedLoadVectorize\n";
        break;
      case NeedToGather:
        dbgs() << "NeedToGather\n";
        break;
      case CombinedVectorize:
        dbgs() << "CombinedVectorize\n";
        break;
      case SplitVectorize:
        dbgs() << "SplitVectorize\n";
        break;
      }
      if (S) {
        dbgs() << "MainOp: " << *S.getMainOp() << "\n";
        dbgs() << "AltOp: " << *S.getAltOp() << "\n";
      } else {
        dbgs() << "MainOp: NULL\n";
        dbgs() << "AltOp: NULL\n";
      }
      dbgs() << "VectorizedValue: ";
      if (VectorizedValue)
        dbgs() << *VectorizedValue << "\n";
      else
        dbgs() << "NULL\n";
      dbgs() << "ReuseShuffleIndices: ";
      if (ReuseShuffleIndices.empty())
        dbgs() << "Empty";
      else
        for (int ReuseIdx : ReuseShuffleIndices)
          dbgs() << ReuseIdx << ", ";
      dbgs() << "\n";
      dbgs() << "ReorderIndices: ";
      for (unsigned ReorderIdx : ReorderIndices)
        dbgs() << ReorderIdx << ", ";
      dbgs() << "\n";
      dbgs() << "UserTreeIndex: ";
      if (UserTreeIndex)
        dbgs() << UserTreeIndex;
      else
        dbgs() << "<invalid>";
      dbgs() << "\n";
      if (!StructEVIndices.empty()) {
        dbgs() << "StructEVIndices: ";
        interleaveComma(StructEVIndices, dbgs());
        dbgs() << "\n";
      }
      if (!CombinedEntriesWithIndices.empty()) {
        dbgs() << "Combined entries: ";
        interleaveComma(CombinedEntriesWithIndices, dbgs(), [&](const auto &P) {
          dbgs() << "Entry index " << P.first << " with offset " << P.second;
        });
        dbgs() << "\n";
      }
    }
#endif
  };

#ifndef NDEBUG
  void dumpTreeCosts(const TreeEntry *E, InstructionCost ReuseShuffleCost,
                     InstructionCost VecCost, InstructionCost ScalarCost,
                     StringRef Banner) const {
    dbgs() << "SLP: " << Banner << ":\n";
    E->dump();
    dbgs() << "SLP: Costs:\n";
    dbgs() << "SLP:     ReuseShuffleCost = " << ReuseShuffleCost << "\n";
    dbgs() << "SLP:     VectorCost = " << VecCost << "\n";
    dbgs() << "SLP:     ScalarCost = " << ScalarCost << "\n";
    dbgs() << "SLP:     ReuseShuffleCost + VecCost - ScalarCost = "
           << ReuseShuffleCost + VecCost - ScalarCost << "\n";
  }
#endif

  /// Create a new gather TreeEntry
  TreeEntry *newGatherTreeEntry(ArrayRef<Value *> VL,
                                const InstructionsState &S,
                                const EdgeInfo &UserTreeIdx,
                                ArrayRef<int> ReuseShuffleIndices = {}) {
    auto Invalid = ScheduleBundle::invalid();
    return newTreeEntry(VL, Invalid, S, UserTreeIdx, ReuseShuffleIndices);
  }

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL, ScheduleBundle &Bundle,
                          const InstructionsState &S,
                          const EdgeInfo &UserTreeIdx,
                          ArrayRef<int> ReuseShuffleIndices = {},
                          ArrayRef<unsigned> ReorderIndices = {},
                          unsigned InterleaveFactor = 0) {
    TreeEntry::EntryState EntryState =
        Bundle ? TreeEntry::Vectorize : TreeEntry::NeedToGather;
    TreeEntry *E = newTreeEntry(VL, EntryState, Bundle, S, UserTreeIdx,
                                ReuseShuffleIndices, ReorderIndices);
    if (E && InterleaveFactor > 0)
      E->setInterleave(InterleaveFactor);
    return E;
  }

  TreeEntry *newTreeEntry(ArrayRef<Value *> VL,
                          TreeEntry::EntryState EntryState,
                          ScheduleBundle &Bundle, const InstructionsState &S,
                          const EdgeInfo &UserTreeIdx,
                          ArrayRef<int> ReuseShuffleIndices = {},
                          ArrayRef<unsigned> ReorderIndices = {}) {
    assert(((!Bundle && (EntryState == TreeEntry::NeedToGather ||
                         EntryState == TreeEntry::SplitVectorize)) ||
            (Bundle && EntryState != TreeEntry::NeedToGather &&
             EntryState != TreeEntry::SplitVectorize)) &&
           "Need to vectorize gather entry?");
    // Gathered loads still gathered? Do not create entry, use the original one.
    if (GatheredLoadsEntriesFirst.has_value() &&
        EntryState == TreeEntry::NeedToGather && S &&
        S.getOpcode() == Instruction::Load && UserTreeIdx.EdgeIdx == UINT_MAX &&
        !UserTreeIdx.UserTE)
      return nullptr;
    VectorizableTree.push_back(std::make_unique<TreeEntry>(VectorizableTree));
    TreeEntry *Last = VectorizableTree.back().get();
    Last->Idx = VectorizableTree.size() - 1;
    Last->State = EntryState;
    if (UserTreeIdx.UserTE)
      OperandsToTreeEntry.try_emplace(
          std::make_pair(UserTreeIdx.UserTE, UserTreeIdx.EdgeIdx), Last);
    Last->ReuseShuffleIndices.append(ReuseShuffleIndices.begin(),
                                     ReuseShuffleIndices.end());
    if (ReorderIndices.empty()) {
      Last->Scalars.assign(VL.begin(), VL.end());
      if (S)
        Last->setOperations(S);
    } else {
      // Reorder scalars and build final mask.
      Last->Scalars.assign(VL.size(), nullptr);
      transform(ReorderIndices, Last->Scalars.begin(),
                [VL](unsigned Idx) -> Value * {
                  if (Idx >= VL.size())
                    return UndefValue::get(VL.front()->getType());
                  return VL[Idx];
                });
      InstructionsState S = getSameOpcode(Last->Scalars, *TLI);
      if (S)
        Last->setOperations(S);
      Last->ReorderIndices.append(ReorderIndices.begin(), ReorderIndices.end());
    }
    if (EntryState == TreeEntry::SplitVectorize) {
      assert(S && "Split nodes must have operations.");
      Last->setOperations(S);
      SmallPtrSet<Value *, 4> Processed;
      for (Value *V : VL) {
        auto *I = dyn_cast<Instruction>(V);
        if (!I)
          continue;
        auto It = ScalarsInSplitNodes.find(V);
        if (It == ScalarsInSplitNodes.end()) {
          ScalarsInSplitNodes.try_emplace(V).first->getSecond().push_back(Last);
          (void)Processed.insert(V);
        } else if (Processed.insert(V).second) {
          assert(!is_contained(It->getSecond(), Last) &&
                 "Value already associated with the node.");
          It->getSecond().push_back(Last);
        }
      }
    } else if (!Last->isGather()) {
      if (isa<PHINode>(S.getMainOp()) ||
          isVectorLikeInstWithConstOps(S.getMainOp()) ||
          (!S.areInstructionsWithCopyableElements() &&
           doesNotNeedToSchedule(VL)) ||
          all_of(VL, [&](Value *V) { return S.isNonSchedulable(V); }))
        Last->setDoesNotNeedToSchedule();
      SmallPtrSet<Value *, 4> Processed;
      for (Value *V : VL) {
        if (isa<PoisonValue>(V))
          continue;
        if (S.isCopyableElement(V)) {
          Last->addCopyableElement(V);
          continue;
        }
        auto It = ScalarToTreeEntries.find(V);
        if (It == ScalarToTreeEntries.end()) {
          ScalarToTreeEntries.try_emplace(V).first->getSecond().push_back(Last);
          (void)Processed.insert(V);
        } else if (Processed.insert(V).second) {
          assert(!is_contained(It->getSecond(), Last) &&
                 "Value already associated with the node.");
          It->getSecond().push_back(Last);
        }
      }
      // Update the scheduler bundle to point to this TreeEntry.
      assert((!Bundle.getBundle().empty() || Last->doesNotNeedToSchedule()) &&
             "Bundle and VL out of sync");
      if (!Bundle.getBundle().empty()) {
#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
        auto *BundleMember = Bundle.getBundle().begin();
        SmallPtrSet<Value *, 4> Processed;
        for (Value *V : VL) {
          if (S.isNonSchedulable(V) || !Processed.insert(V).second)
            continue;
          ++BundleMember;
        }
        assert(BundleMember == Bundle.getBundle().end() &&
               "Bundle and VL out of sync");
#endif
        Bundle.setTreeEntry(Last);
      }
    } else {
      // Build a map for gathered scalars to the nodes where they are used.
      bool AllConstsOrCasts = true;
      for (Value *V : VL) {
        if (S && S.areInstructionsWithCopyableElements() &&
            S.isCopyableElement(V))
          Last->addCopyableElement(V);
        if (!isConstant(V)) {
          auto *I = dyn_cast<CastInst>(V);
          AllConstsOrCasts &= I && I->getType()->isIntegerTy();
          if (UserTreeIdx.EdgeIdx != UINT_MAX || !UserTreeIdx.UserTE ||
              !UserTreeIdx.UserTE->isGather())
            ValueToGatherNodes.try_emplace(V).first->getSecond().insert(Last);
        }
      }
      if (AllConstsOrCasts)
        CastMaxMinBWSizes =
            std::make_pair(std::numeric_limits<unsigned>::max(), 1);
      MustGather.insert_range(VL);
    }

    if (UserTreeIdx.UserTE)
      Last->UserTreeIndex = UserTreeIdx;
    return Last;
  }

  /// -- Vectorization State --
  /// Holds all of the tree entries.
  TreeEntry::VecTreeTy VectorizableTree;

#ifndef NDEBUG
  /// Debug printer.
  LLVM_DUMP_METHOD void dumpVectorizableTree() const {
    for (unsigned Id = 0, IdE = VectorizableTree.size(); Id != IdE; ++Id) {
      VectorizableTree[Id]->dump();
      if (TransformedToGatherNodes.contains(VectorizableTree[Id].get()))
        dbgs() << "[[TRANSFORMED TO GATHER]]";
      else if (DeletedNodes.contains(VectorizableTree[Id].get()))
        dbgs() << "[[DELETED NODE]]";
      dbgs() << "\n";
    }
  }
#endif

  /// Get list of vector entries, associated with the value \p V.
  ArrayRef<TreeEntry *> getTreeEntries(const Value *V) const {
    assert(V && "V cannot be nullptr.");
    auto It = ScalarToTreeEntries.find(V);
    if (It == ScalarToTreeEntries.end())
      return {};
    return It->getSecond();
  }

  /// Get list of split vector entries, associated with the value \p V.
  ArrayRef<TreeEntry *> getSplitTreeEntries(Value *V) const {
    assert(V && "V cannot be nullptr.");
    auto It = ScalarsInSplitNodes.find(V);
    if (It == ScalarsInSplitNodes.end())
      return {};
    return It->getSecond();
  }

  /// Returns first vector node for value \p V, matching values \p VL.
  TreeEntry *getSameValuesTreeEntry(Value *V, ArrayRef<Value *> VL,
                                    bool SameVF = false) const {
    assert(V && "V cannot be nullptr.");
    for (TreeEntry *TE : ScalarToTreeEntries.lookup(V))
      if ((!SameVF || TE->getVectorFactor() == VL.size()) && TE->isSame(VL))
        return TE;
    return nullptr;
  }

  /// Contains all the outputs of legality analysis for a list of values to
  /// vectorize.
  class ScalarsVectorizationLegality {
    InstructionsState S;
    bool IsLegal;
    bool TryToFindDuplicates;
    bool TrySplitVectorize;

  public:
    ScalarsVectorizationLegality(InstructionsState S, bool IsLegal,
                                 bool TryToFindDuplicates = true,
                                 bool TrySplitVectorize = false)
        : S(S), IsLegal(IsLegal), TryToFindDuplicates(TryToFindDuplicates),
          TrySplitVectorize(TrySplitVectorize) {
      assert((!IsLegal || (S.valid() && TryToFindDuplicates)) &&
             "Inconsistent state");
    }
    const InstructionsState &getInstructionsState() const { return S; };
    bool isLegal() const { return IsLegal; }
    bool tryToFindDuplicates() const { return TryToFindDuplicates; }
    bool trySplitVectorize() const { return TrySplitVectorize; }
  };

  /// Checks if the specified list of the instructions/values can be vectorized
  /// in general.
  ScalarsVectorizationLegality
  getScalarsVectorizationLegality(ArrayRef<Value *> VL, unsigned Depth,
                                  const EdgeInfo &UserTreeIdx) const;

  /// Checks if the specified list of the instructions/values can be vectorized
  /// and fills required data before actual scheduling of the instructions.
  TreeEntry::EntryState getScalarsVectorizationState(
      const InstructionsState &S, ArrayRef<Value *> VL,
      bool IsScatterVectorizeUserTE, OrdersType &CurrentOrder,
      SmallVectorImpl<Value *> &PointerOps, StridedPtrInfo &SPtrInfo,
      SmallVectorImpl<int> &ReuseShuffleIndices);

  /// Maps a specific scalar to its tree entry(ies).
  SmallDenseMap<Value *, SmallVector<TreeEntry *>> ScalarToTreeEntries;

  /// List of deleted non-profitable nodes.
  SmallPtrSet<const TreeEntry *, 8> DeletedNodes;

  /// List of nodes, transformed to gathered, with their conservative
  /// gather/buildvector cost estimation.
  SmallDenseMap<const TreeEntry *, InstructionCost> TransformedToGatherNodes;

  /// Maps the operand index and entry to the corresponding tree entry.
  SmallDenseMap<std::pair<const TreeEntry *, unsigned>, TreeEntry *>
      OperandsToTreeEntry;

  /// Scalars, used in split vectorize nodes.
  SmallDenseMap<Value *, SmallVector<TreeEntry *>> ScalarsInSplitNodes;

  /// Maps a value to the proposed vectorizable size.
  SmallDenseMap<Value *, unsigned> InstrElementSize;

  /// A list of scalars that we found that we need to keep as scalars.
  ValueSet MustGather;

  /// Maps each peeled reassociated scalar to owning entries. Keeps them
  /// treated as vectorized while an owner is live.
  SmallDenseMap<const Value *, SmallVector<const TreeEntry *>>
      ReassocScalarToTreeEntries;

  /// Peeled reassociated scalars that must survive erasure: claimed by a
  /// gather node, listed in some tree entry's scalars, or feeding another
  /// kept scalar.
  SmallPtrSet<const Value *, 8> KeptReassocScalars;

  /// A set of first non-schedulable values.
  ValueSet NonScheduledFirst;

  /// A map between the vectorized entries and the last instructions in the
  /// bundles. The bundles are built in use order, not in the def order of the
  /// instructions. So, we cannot rely directly on the last instruction in the
  /// bundle being the last instruction in the program order during
  /// vectorization process since the basic blocks are affected, need to
  /// pre-gather them before.
  SmallDenseMap<const TreeEntry *, WeakTrackingVH> EntryToLastInstruction;

  /// Keeps the mapping between the last instructions and their insertion
  /// points, which is an instruction-after-the-last-instruction.
  SmallDenseMap<const Instruction *, Instruction *> LastInstructionToPos;

  /// List of gather nodes, depending on other gather/vector nodes, which should
  /// be emitted after the vector instruction emission process to correctly
  /// handle order of the vector instructions and shuffles.
  SetVector<const TreeEntry *> PostponedGathers;

  using ValueToGatherNodesMap =
      DenseMap<Value *, SmallSetVector<const TreeEntry *, 4>>;
  ValueToGatherNodesMap ValueToGatherNodes;

  SmallDenseMap<TreeEntry *, StridedPtrInfo> TreeEntryToStridedPtrInfoMap;

  /// A list of the load entries (node indices), which can be vectorized using
  /// strided or masked gather approach, but attempted to be represented as
  /// contiguous loads.
  SetVector<unsigned> LoadEntriesToVectorize;

  /// true if graph nodes transforming mode is on.
  bool IsGraphTransformMode = false;

  /// The index of the first gathered load entry in the VectorizeTree.
  std::optional<unsigned> GatheredLoadsEntriesFirst;

  /// Root entries of the subtrees built for the splat gather nodes' unique
  /// scalars. They have no users in the tree and must be emitted explicitly
  /// before the root node.
  SmallVector<TreeEntry *> SplatGatheredScalarsRoots;

  /// Number of tree entries added while building the splat gather subtrees.
  /// The subtrees are auxiliary and must not inflate the tree size recorded
  /// for failed store chain attempts.
  unsigned NumCanonicalSplatSubtreeEntries = 0;

  /// Maps compress entries to their mask data for the final codegen.
  SmallDenseMap<const TreeEntry *,
                std::tuple<SmallVector<int>, VectorType *, unsigned, bool>>
      CompressEntryToData;

  /// The loop nest, used to check if only a single loop nest is vectorized, not
  /// multiple, to avoid side-effects from the loop-aware cost model.
  SmallVector<const Loop *> CurrentLoopNest;

  /// Per-depth SCEVs trip counts at every loop level where the tree builder has
  /// joined diverging sibling loops.
  SmallVector<const SCEV *> MergedLoopBTCs;

  /// Maps the loops to their loop nests.
  SmallDenseMap<const Loop *, SmallVector<const Loop *>> LoopToLoopNest;

  /// Per-loop cache of nest scale factors: the product of trip counts of the
  /// loop and all of its ancestors. Shared by getLoopNestScale() and (via it)
  /// by getScaleToLoopIterations() and getGatherNodeEffectiveScale().
  SmallDenseMap<const Loop *, uint64_t> LoopNestScaleCache;

  /// This POD struct describes one external user in the vectorized tree.
  struct ExternalUser {
    ExternalUser(Value *S, llvm::User *U, const TreeEntry &E, unsigned L)
        : Scalar(S), User(U), E(E), Lane(L) {}

    /// Which scalar in our function.
    Value *Scalar = nullptr;

    /// Which user that uses the scalar.
    llvm::User *User = nullptr;

    /// Vector node, the value is part of.
    const TreeEntry &E;

    /// Which lane does the scalar belong to.
    unsigned Lane;
  };
  using UserList = SmallVector<ExternalUser, 16>;

  /// Checks if two instructions may access the same memory.
  ///
  /// \p Loc1 is the location of \p Inst1. It is passed explicitly because it
  /// is invariant in the calling loop.
  bool isAliased(const MemoryLocation &Loc1, Instruction *Inst1,
                 Instruction *Inst2) {
    assert(Loc1.Ptr && isSimple(Inst1) && "Expected simple first instruction.");
    // First check if the result is already in the cache.
    AliasCacheKey Key = std::make_pair(Inst1, Inst2);
    auto Res = AliasCache.try_emplace(Key);
    if (!Res.second)
      return Res.first->second;
    bool Aliased = isModOrRefSet(BatchAA.getModRefInfo(Inst2, Loc1));
    // Store the result in the cache.
    Res.first->getSecond() = Aliased;
    return Aliased;
  }

  /// Returns true if the may-alias dependency between simple load/store
  /// instructions \p Inst1 and \p Inst2 could be disambiguated by a runtime
  /// alias check.
  bool isRuntimeCheckableAliasPair(Instruction *Inst1, Instruction *Inst2);

  /// Records the (distinct base object) pair behind the may-alias dependency
  /// of \p Inst1 and \p Inst2 as a runtime alias check guarding the region in
  /// block \p BB. Returns true if the pair was recorded.
  bool recordRuntimeAliasCheck(BasicBlock *BB, Instruction *Inst1,
                               Instruction *Inst2);

  /// Emits the collected runtime alias checks and versions the affected block,
  /// duplicating its body into a scalar fallback guarded by the checks.
  void versionBlocksForRuntimeChecks();

  /// Builds the i1 value that is true when any pair of checked base objects
  /// overlaps at runtime. The base address bounds are materialized from their
  /// SCEVs with \p Exp.
  Value *emitRuntimeAliasCheck(IRBuilderBase &Builder, SCEVExpander &Exp);

  /// Data to model and emit the runtime alias checks.
  struct RuntimeAliasCheckInfo {
    /// The block whose body is guarded by the checks. Exactly one block is
    /// supported per attempt.
    BasicBlock *BB = nullptr;
    /// Pairs of base objects that must be proven disjoint.
    SmallSetVector<std::pair<const Value *, const Value *>, 4> BasePairs;
    /// Accessed address range [Low, High) for each involved base object.
    SmallMapVector<const Value *, std::pair<const SCEV *, const SCEV *>, 4>
        Bounds;

    void clear() {
      BB = nullptr;
      BasePairs.clear();
      Bounds.clear();
    }
  };

  /// When true, scheduling drops may-alias memory dependencies between
  /// distinct, range-checkable base objects and records them as runtime alias
  /// checks instead.
  bool TryRuntimeAliasChecks = false;

  /// Runtime alias checks collected during the last optimistic buildTree().
  RuntimeAliasCheckInfo RTChecks;

  /// Base-object pairs already proven disjoint by the block's runtime alias
  /// check.
  SmallDenseMap<BasicBlock *,
                SmallDenseSet<std::pair<const Value *, const Value *>, 4>, 2>
      VersionedBlockCheckedPairs;

  /// Scalar fallback blocks.
  SmallPtrSet<BasicBlock *, 4> ScalarFallbackBlocks;

  /// Blocks for which a runtime-checks versioning attempt was made
  /// and did not produce a profitable versioning.
  SmallPtrSet<BasicBlock *, 8> FailedRuntimeChecksBlocks;

  /// Returns true if a may-alias dependency between the simple load/store
  /// instructions \p Inst1 and \p Inst2 in block \p BB is already covered by a
  /// runtime alias check emitted for \p BB by a previous versioning.
  bool isCoveredByExistingVersionCheck(BasicBlock *BB, Instruction *Inst1,
                                       Instruction *Inst2) const;

  /// True, if a may-alias dependency between distinct, range-checkable base
  /// objects is observed (whether or not it was dropped).
  bool HasRuntimeCheckableBlockers = false;

  /// True, if a kept may-alias dependency is not runtime-checkable (call or a
  /// non-simple memaccess).
  bool HasNonCheckableMemBlocker = false;

  /// Runtime checks are validated and bounded the collected checks.
  bool RTChecksFinalized = false;

  /// Set when a block was versioned with runtime alias checks, which changes
  /// the CFG. Used to drop CFG-analysis preservation for the run.
  bool CFGChanged = false;

  /// Guarded block body (non-PHI, non-terminator) in original source order.
  SmallVector<Instruction *> RTOrigBodyOrder;

  using AliasCacheKey = std::pair<Instruction *, Instruction *>;

  /// Cache for alias results.
  /// TODO: consider moving this to the AliasAnalysis itself.
  SmallDenseMap<AliasCacheKey, bool> AliasCache;

  // Cache for pointerMayBeCaptured calls inside AA.  This is preserved
  // globally through SLP because we don't perform any action which
  // invalidates capture results.
  BatchAAResults BatchAA;

  /// Temporary store for deleted instructions. Instructions will be deleted
  /// eventually when the BoUpSLP is destructed.  The deferral is required to
  /// ensure that there are no incorrect collisions in the AliasCache, which
  /// can happen if a new instruction is allocated at the same address as a
  /// previously deleted instruction.
  DenseSet<Instruction *> DeletedInstructions;

  /// Set of the instruction, being analyzed already for reductions.
  SmallPtrSet<Instruction *, 16> AnalyzedReductionsRoots;

  /// Set of hashes for the list of reduction values already being analyzed.
  DenseSet<size_t> AnalyzedReductionVals;

  /// Set of hashes for the bundles, rejected as non-vectorizable.
  SmallDenseSet<size_t, 8> AnalyzedBundles;

  /// Set of the values, which were a part of the analyzed vector nodes.
  SmallPtrSet<const Value *, 32> AnalyzedScalars;

  /// Cache of the number of parts for the types and the parts limit.
  mutable SmallDenseMap<std::tuple<Type *, Type *, unsigned>, unsigned>
      NumberOfPartsCache;

  /// Values, already been analyzed for mininmal bitwidth and found to be
  /// non-profitable.
  DenseSet<Value *> AnalyzedMinBWVals;

  /// A list of values that need to extracted out of the tree.
  /// This list holds pairs of (Internal Scalar : External User). External User
  /// can be nullptr, it means that this Internal Scalar will be used later,
  /// after vectorization.
  UserList ExternalUses;

  /// A list of GEPs which can be reaplced by scalar GEPs instead of
  /// extractelement instructions.
  SmallPtrSet<Value *, 4> ExternalUsesAsOriginalScalar;

  /// A list of scalar to be extracted without specific user necause of too many
  /// uses.
  SmallPtrSet<Value *, 4> ExternalUsesWithNonUsers;

  /// Replacements emitted for the external uses without users, consumed after
  /// the tree vectorization; must not be collected as dead operands of the
  /// erased scalars.
  SmallPtrSet<Value *, 4> ExternalUseReplacements;

  /// Values used only by @llvm.assume calls.
  SmallPtrSet<const Value *, 32> EphValues;

  /// Holds all of the instructions that we gathered, shuffle instructions and
  /// extractelements.
  SetVector<Instruction *> GatherShuffleExtractSeq;

  /// A list of blocks that we are going to CSE.
  DenseSet<BasicBlock *> CSEBlocks;

  /// List of hashes of vector of loads, which are known to be non vectorizable.
  DenseSet<size_t> ListOfKnonwnNonVectorizableLoads;

  /// Represents a scheduling entity, either ScheduleData, ScheduleCopyableData
  /// or ScheduleBundle. ScheduleData used to gather dependecies for a single
  /// instructions, while ScheduleBundle represents a batch of instructions,
  /// going to be groupped together. ScheduleCopyableData models extra user for
  /// "copyable" instructions.
  class ScheduleEntity {
    friend class ScheduleBundle;
    friend class ScheduleData;
    friend class ScheduleCopyableData;

  protected:
    enum class Kind { ScheduleData, ScheduleBundle, ScheduleCopyableData };
    Kind getKind() const { return K; }
    ScheduleEntity(Kind K) : K(K) {}

  private:
    /// Used for getting a "good" final ordering of instructions.
    int SchedulingPriority = 0;
    /// True if this instruction (or bundle) is scheduled (or considered as
    /// scheduled in the dry-run).
    bool IsScheduled = false;
    /// The kind of the ScheduleEntity.
    const Kind K = Kind::ScheduleData;

  public:
    ScheduleEntity() = delete;
    /// Gets/sets the scheduling priority.
    void setSchedulingPriority(int Priority) { SchedulingPriority = Priority; }
    int getSchedulingPriority() const { return SchedulingPriority; }
    bool isReady() const {
      if (const auto *SD = dyn_cast<ScheduleData>(this))
        return SD->isReady();
      if (const auto *CD = dyn_cast<ScheduleCopyableData>(this))
        return CD->isReady();
      return cast<ScheduleBundle>(this)->isReady();
    }
    /// Returns true if the dependency information has been calculated.
    /// Note that depenendency validity can vary between instructions within
    /// a single bundle.
    bool hasValidDependencies() const {
      if (const auto *SD = dyn_cast<ScheduleData>(this))
        return SD->hasValidDependencies();
      if (const auto *CD = dyn_cast<ScheduleCopyableData>(this))
        return CD->hasValidDependencies();
      return cast<ScheduleBundle>(this)->hasValidDependencies();
    }
    /// Gets the number of unscheduled dependencies.
    int getUnscheduledDeps() const {
      if (const auto *SD = dyn_cast<ScheduleData>(this))
        return SD->getUnscheduledDeps();
      if (const auto *CD = dyn_cast<ScheduleCopyableData>(this))
        return CD->getUnscheduledDeps();
      return cast<ScheduleBundle>(this)->unscheduledDepsInBundle();
    }
    /// Increments the number of unscheduled dependencies.
    int incrementUnscheduledDeps(int Incr) {
      if (auto *SD = dyn_cast<ScheduleData>(this))
        return SD->incrementUnscheduledDeps(Incr);
      return cast<ScheduleCopyableData>(this)->incrementUnscheduledDeps(Incr);
    }
    /// Gets the number of dependencies.
    int getDependencies() const {
      if (const auto *SD = dyn_cast<ScheduleData>(this))
        return SD->getDependencies();
      return cast<ScheduleCopyableData>(this)->getDependencies();
    }
    /// Gets the instruction.
    Instruction *getInst() const {
      if (const auto *SD = dyn_cast<ScheduleData>(this))
        return SD->getInst();
      return cast<ScheduleCopyableData>(this)->getInst();
    }

    /// Gets/sets if the bundle is scheduled.
    bool isScheduled() const { return IsScheduled; }
    void setScheduled(bool Scheduled) { IsScheduled = Scheduled; }

    static bool classof(const ScheduleEntity *) { return true; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dump(raw_ostream &OS) const {
      if (const auto *SD = dyn_cast<ScheduleData>(this))
        return SD->dump(OS);
      if (const auto *CD = dyn_cast<ScheduleCopyableData>(this))
        return CD->dump(OS);
      return cast<ScheduleBundle>(this)->dump(OS);
    }

    LLVM_DUMP_METHOD void dump() const {
      dump(dbgs());
      dbgs() << '\n';
    }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  };

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  friend inline raw_ostream &operator<<(raw_ostream &OS,
                                        const BoUpSLP::ScheduleEntity &SE) {
    SE.dump(OS);
    return OS;
  }
#endif

  /// Contains all scheduling relevant data for an instruction.
  /// A ScheduleData either represents a single instruction or a member of an
  /// instruction bundle (= a group of instructions which is combined into a
  /// vector instruction).
  class ScheduleData final : public ScheduleEntity {
  public:
    // The initial value for the dependency counters. It means that the
    // dependencies are not calculated yet.
    enum { InvalidDeps = -1 };

    ScheduleData() : ScheduleEntity(Kind::ScheduleData) {}
    static bool classof(const ScheduleEntity *Entity) {
      return Entity->getKind() == Kind::ScheduleData;
    }

    void init(int BlockSchedulingRegionID, Instruction *I) {
      NextLoadStore = nullptr;
      IsScheduled = false;
      SchedulingRegionID = BlockSchedulingRegionID;
      clearDependencies();
      Inst = I;
    }

    /// Verify basic self consistency properties
    void verify() {
      if (hasValidDependencies()) {
        assert(UnscheduledDeps <= Dependencies && "invariant");
      } else {
        assert(UnscheduledDeps == Dependencies && "invariant");
      }

      if (IsScheduled) {
        assert(hasValidDependencies() && UnscheduledDeps == 0 &&
               "unexpected scheduled state");
      }
    }

    /// Returns true if the dependency information has been calculated.
    /// Note that depenendency validity can vary between instructions within
    /// a single bundle.
    bool hasValidDependencies() const { return Dependencies != InvalidDeps; }

    /// Returns true if it is ready for scheduling, i.e. it has no more
    /// unscheduled depending instructions/bundles.
    bool isReady() const { return UnscheduledDeps == 0 && !IsScheduled; }

    /// Modifies the number of unscheduled dependencies for this instruction,
    /// and returns the number of remaining dependencies for the containing
    /// bundle.
    int incrementUnscheduledDeps(int Incr) {
      assert(hasValidDependencies() &&
             "increment of unscheduled deps would be meaningless");
      UnscheduledDeps += Incr;
      assert(UnscheduledDeps >= 0 &&
             "Expected valid number of unscheduled deps");
      return UnscheduledDeps;
    }

    /// Sets the number of unscheduled dependencies to the number of
    /// dependencies.
    void resetUnscheduledDeps() { UnscheduledDeps = Dependencies; }

    /// Clears all dependency information.
    void clearDependencies() {
      clearDirectDependencies();
      MemoryDependencies.clear();
      ControlDependencies.clear();
    }

    /// Clears all direct dependencies only, except for control and memory
    /// dependencies.
    /// Required for copyable elements to correctly handle control/memory deps
    /// and avoid extra reclaculation of such deps.
    void clearDirectDependencies() {
      Dependencies = InvalidDeps;
      resetUnscheduledDeps();
      IsScheduled = false;
    }

    /// Gets the number of unscheduled dependencies.
    int getUnscheduledDeps() const { return UnscheduledDeps; }
    /// Gets the number of dependencies.
    int getDependencies() const { return Dependencies; }
    /// Initializes the number of dependencies.
    void initDependencies() { Dependencies = 0; }
    /// Increments the number of dependencies.
    void incDependencies() { Dependencies++; }

    /// Gets scheduling region ID.
    int getSchedulingRegionID() const { return SchedulingRegionID; }

    /// Gets the instruction.
    Instruction *getInst() const { return Inst; }

    /// Gets the list of memory dependencies.
    ArrayRef<ScheduleData *> getMemoryDependencies() const {
      return MemoryDependencies;
    }
    /// Adds a memory dependency.
    void addMemoryDependency(ScheduleData *Dep) {
      MemoryDependencies.push_back(Dep);
    }
    /// Gets the list of control dependencies.
    ArrayRef<ScheduleData *> getControlDependencies() const {
      return ControlDependencies;
    }
    /// Adds a control dependency.
    void addControlDependency(ScheduleData *Dep) {
      ControlDependencies.push_back(Dep);
    }
    /// Gets/sets the next load/store instruction in the block.
    ScheduleData *getNextLoadStore() const { return NextLoadStore; }
    void setNextLoadStore(ScheduleData *Next) { NextLoadStore = Next; }

    void dump(raw_ostream &OS) const { OS << *Inst; }

    LLVM_DUMP_METHOD void dump() const {
      dump(dbgs());
      dbgs() << '\n';
    }

  private:
    Instruction *Inst = nullptr;

    /// Single linked list of all memory instructions (e.g. load, store, call)
    /// in the block - until the end of the scheduling region.
    ScheduleData *NextLoadStore = nullptr;

    /// The dependent memory instructions.
    /// This list is derived on demand in calculateDependencies().
    SmallVector<ScheduleData *> MemoryDependencies;

    /// List of instructions which this instruction could be control dependent
    /// on.  Allowing such nodes to be scheduled below this one could introduce
    /// a runtime fault which didn't exist in the original program.
    /// ex: this is a load or udiv following a readonly call which inf loops
    SmallVector<ScheduleData *> ControlDependencies;

    /// This ScheduleData is in the current scheduling region if this matches
    /// the current SchedulingRegionID of BlockScheduling.
    int SchedulingRegionID = 0;

    /// The number of dependencies. Constitutes of the number of users of the
    /// instruction plus the number of dependent memory instructions (if any).
    /// This value is calculated on demand.
    /// If InvalidDeps, the number of dependencies is not calculated yet.
    int Dependencies = InvalidDeps;

    /// The number of dependencies minus the number of dependencies of scheduled
    /// instructions. As soon as this is zero, the instruction/bundle gets ready
    /// for scheduling.
    /// Note that this is negative as long as Dependencies is not calculated.
    int UnscheduledDeps = InvalidDeps;
  };

#ifndef NDEBUG
  friend inline raw_ostream &operator<<(raw_ostream &OS,
                                        const BoUpSLP::ScheduleData &SD) {
    SD.dump(OS);
    return OS;
  }
#endif

  class ScheduleBundle final : public ScheduleEntity {
    /// The schedule data for the instructions in the bundle.
    SmallVector<ScheduleEntity *> Bundle;
    /// True if this bundle is valid.
    bool IsValid = true;
    /// The TreeEntry that this instruction corresponds to.
    TreeEntry *TE = nullptr;
    ScheduleBundle(bool IsValid)
        : ScheduleEntity(Kind::ScheduleBundle), IsValid(IsValid) {}

  public:
    ScheduleBundle() : ScheduleEntity(Kind::ScheduleBundle) {}
    static bool classof(const ScheduleEntity *Entity) {
      return Entity->getKind() == Kind::ScheduleBundle;
    }

    /// Verify basic self consistency properties
    void verify() const {
      for (const ScheduleEntity *SD : Bundle) {
        if (SD->hasValidDependencies()) {
          assert(SD->getUnscheduledDeps() <= SD->getDependencies() &&
                 "invariant");
        } else {
          assert(SD->getUnscheduledDeps() == SD->getDependencies() &&
                 "invariant");
        }

        if (isScheduled()) {
          assert(SD->hasValidDependencies() && SD->getUnscheduledDeps() == 0 &&
                 "unexpected scheduled state");
        }
      }
    }

    /// Returns the number of unscheduled dependencies in the bundle.
    int unscheduledDepsInBundle() const {
      assert(*this && "bundle must not be empty");
      int Sum = 0;
      for (const ScheduleEntity *BundleMember : Bundle) {
        if (BundleMember->getUnscheduledDeps() == ScheduleData::InvalidDeps)
          return ScheduleData::InvalidDeps;
        Sum += BundleMember->getUnscheduledDeps();
      }
      return Sum;
    }

    /// Returns true if the dependency information has been calculated.
    /// Note that depenendency validity can vary between instructions within
    /// a single bundle.
    bool hasValidDependencies() const {
      return all_of(Bundle, [](const ScheduleEntity *SD) {
        return SD->hasValidDependencies();
      });
    }

    /// Returns true if it is ready for scheduling, i.e. it has no more
    /// unscheduled depending instructions/bundles.
    bool isReady() const {
      assert(*this && "bundle must not be empty");
      return unscheduledDepsInBundle() == 0 && !isScheduled();
    }

    /// Returns the bundle of scheduling data, associated with the current
    /// instruction.
    ArrayRef<ScheduleEntity *> getBundle() { return Bundle; }
    ArrayRef<const ScheduleEntity *> getBundle() const { return Bundle; }
    /// Adds an instruction to the bundle.
    void add(ScheduleEntity *SD) { Bundle.push_back(SD); }

    /// Gets/sets the associated tree entry.
    void setTreeEntry(TreeEntry *TE) { this->TE = TE; }
    TreeEntry *getTreeEntry() const { return TE; }

    static ScheduleBundle invalid() { return {false}; }

    operator bool() const { return IsValid; }

#ifndef NDEBUG
    void dump(raw_ostream &OS) const {
      if (!*this) {
        OS << "[]";
        return;
      }
      OS << '[';
      interleaveComma(Bundle, OS, [&](const ScheduleEntity *SD) {
        if (isa<ScheduleCopyableData>(SD))
          OS << "<Copyable>";
        OS << *SD->getInst();
      });
      OS << ']';
    }

    LLVM_DUMP_METHOD void dump() const {
      dump(dbgs());
      dbgs() << '\n';
    }
#endif // NDEBUG
  };

#ifndef NDEBUG
  friend inline raw_ostream &operator<<(raw_ostream &OS,
                                        const BoUpSLP::ScheduleBundle &Bundle) {
    Bundle.dump(OS);
    return OS;
  }
#endif

  /// Contains all scheduling relevant data for the copyable instruction.
  /// It models the virtual instructions, supposed to replace the original
  /// instructions. E.g., if instruction %0 = load is a part of the bundle [%0,
  /// %1], where %1 = add, then the ScheduleCopyableData models virtual
  /// instruction %virt = add %0, 0.
  class ScheduleCopyableData final : public ScheduleEntity {
    /// The source schedule data for the instruction.
    Instruction *Inst = nullptr;
    /// The edge information for the instruction.
    const EdgeInfo EI;
    /// This ScheduleData is in the current scheduling region if this matches
    /// the current SchedulingRegionID of BlockScheduling.
    int SchedulingRegionID = 0;
    /// Bundle, this data is part of.
    ScheduleBundle &Bundle;

  public:
    ScheduleCopyableData(int BlockSchedulingRegionID, Instruction *I,
                         const EdgeInfo &EI, ScheduleBundle &Bundle)
        : ScheduleEntity(Kind::ScheduleCopyableData), Inst(I), EI(EI),
          SchedulingRegionID(BlockSchedulingRegionID), Bundle(Bundle) {}
    static bool classof(const ScheduleEntity *Entity) {
      return Entity->getKind() == Kind::ScheduleCopyableData;
    }

    /// Verify basic self consistency properties
    void verify() {
      if (hasValidDependencies()) {
        assert(UnscheduledDeps <= Dependencies && "invariant");
      } else {
        assert(UnscheduledDeps == Dependencies && "invariant");
      }

      if (IsScheduled) {
        assert(hasValidDependencies() && UnscheduledDeps == 0 &&
               "unexpected scheduled state");
      }
    }

    /// Returns true if the dependency information has been calculated.
    /// Note that depenendency validity can vary between instructions within
    /// a single bundle.
    bool hasValidDependencies() const {
      return Dependencies != ScheduleData::InvalidDeps;
    }

    /// Returns true if it is ready for scheduling, i.e. it has no more
    /// unscheduled depending instructions/bundles.
    bool isReady() const { return UnscheduledDeps == 0 && !IsScheduled; }

    /// Modifies the number of unscheduled dependencies for this instruction,
    /// and returns the number of remaining dependencies for the containing
    /// bundle.
    int incrementUnscheduledDeps(int Incr) {
      assert(hasValidDependencies() &&
             "increment of unscheduled deps would be meaningless");
      UnscheduledDeps += Incr;
      assert(UnscheduledDeps >= 0 && "invariant");
      return UnscheduledDeps;
    }

    /// Sets the number of unscheduled dependencies to the number of
    /// dependencies.
    void resetUnscheduledDeps() { UnscheduledDeps = Dependencies; }

    /// Gets the number of unscheduled dependencies.
    int getUnscheduledDeps() const { return UnscheduledDeps; }
    /// Gets the number of dependencies.
    int getDependencies() const { return Dependencies; }
    /// Initializes the number of dependencies.
    void initDependencies() { Dependencies = 0; }
    /// Increments the number of dependencies.
    void incDependencies() { Dependencies++; }

    /// Gets scheduling region ID.
    int getSchedulingRegionID() const { return SchedulingRegionID; }

    /// Gets the instruction.
    Instruction *getInst() const { return Inst; }

    /// Clears all dependency information.
    void clearDependencies() {
      Dependencies = ScheduleData::InvalidDeps;
      UnscheduledDeps = ScheduleData::InvalidDeps;
      IsScheduled = false;
    }

    /// Gets the edge information.
    const EdgeInfo &getEdgeInfo() const { return EI; }

    /// Gets the bundle.
    ScheduleBundle &getBundle() { return Bundle; }
    const ScheduleBundle &getBundle() const { return Bundle; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dump(raw_ostream &OS) const { OS << "[Copyable]" << *getInst(); }

    LLVM_DUMP_METHOD void dump() const {
      dump(dbgs());
      dbgs() << '\n';
    }
#endif // !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

  private:
    /// true, if it has valid dependency information. These nodes always have
    /// only single dependency.
    int Dependencies = ScheduleData::InvalidDeps;

    /// The number of dependencies minus the number of dependencies of scheduled
    /// instructions. As soon as this is zero, the instruction/bundle gets ready
    /// for scheduling.
    /// Note that this is negative as long as Dependencies is not calculated.
    int UnscheduledDeps = ScheduleData::InvalidDeps;
  };

#ifndef NDEBUG
  friend inline raw_ostream &
  operator<<(raw_ostream &OS, const BoUpSLP::ScheduleCopyableData &SD) {
    SD.dump(OS);
    return OS;
  }
#endif

  friend struct GraphTraits<BoUpSLP *>;
  friend struct DOTGraphTraits<BoUpSLP *>;

  /// Contains all scheduling data for a basic block.
  /// It does not schedules instructions, which are not memory read/write
  /// instructions and their operands are either constants, or arguments, or
  /// phis, or instructions from others blocks, or their users are phis or from
  /// the other blocks. The resulting vector instructions can be placed at the
  /// beginning of the basic block without scheduling (if operands does not need
  /// to be scheduled) or at the end of the block (if users are outside of the
  /// block). It allows to save some compile time and memory used by the
  /// compiler.
  /// ScheduleData is assigned for each instruction in between the boundaries of
  /// the tree entry, even for those, which are not part of the graph. It is
  /// required to correctly follow the dependencies between the instructions and
  /// their correct scheduling. The ScheduleData is not allocated for the
  /// instructions, which do not require scheduling, like phis, nodes with
  /// extractelements/insertelements only or nodes with instructions, with
  /// uses/operands outside of the block.
  struct BlockScheduling {
    BlockScheduling(BasicBlock *BB);

    void clear() {
      ScheduledBundles.clear();
      ScheduledBundlesList.clear();
      ScheduleCopyableDataMap.clear();
      ScheduleCopyableDataMapByInst.clear();
      ScheduleCopyableDataMapByInstUser.clear();
      ScheduleCopyableDataMapByUsers.clear();
      ReadyInsts.clear();
      RecalcCopyableOperandDeps.clear();
      IgnoredMemDeps.clear();
      ScheduleStart = nullptr;
      ScheduleEnd = nullptr;
      FirstLoadStoreInRegion = nullptr;
      LastLoadStoreInRegion = nullptr;
      RegionHasStackSave = false;

      // Reduce the maximum schedule region size by the size of the
      // previous scheduling run.
      ScheduleRegionSizeLimit -= ScheduleRegionSize;
      if (ScheduleRegionSizeLimit < MinScheduleRegionSize)
        ScheduleRegionSizeLimit = MinScheduleRegionSize;
      ScheduleRegionSize = 0;

      // Make a new scheduling region, i.e. all existing ScheduleData is not
      // in the new region yet.
      ++SchedulingRegionID;
    }

    ScheduleData *getScheduleData(Instruction *I) {
      if (!I)
        return nullptr;
      if (BB != I->getParent())
        // Avoid lookup if can't possibly be in map.
        return nullptr;
      ScheduleData *SD = ScheduleDataMap.lookup(I);
      if (SD && isInSchedulingRegion(*SD))
        return SD;
      return nullptr;
    }

    ScheduleData *getScheduleData(Value *V) {
      return getScheduleData(dyn_cast<Instruction>(V));
    }

    /// Returns the ScheduleCopyableData for the given edge (user tree entry and
    /// operand number) and value.
    ScheduleCopyableData *getScheduleCopyableData(const EdgeInfo &EI,
                                                  const Value *V) const {
      if (ScheduleCopyableDataMap.empty())
        return nullptr;
      auto It = ScheduleCopyableDataMap.find(std::make_pair(EI, V));
      if (It == ScheduleCopyableDataMap.end())
        return nullptr;
      ScheduleCopyableData *SD = It->getSecond().get();
      if (!isInSchedulingRegion(*SD))
        return nullptr;
      return SD;
    }

    /// Returns the ScheduleCopyableData for the given user \p User, operand
    /// number and operand \p V.
    SmallVector<ScheduleCopyableData *>
    getScheduleCopyableData(const Value *User, unsigned OperandIdx,
                            const Value *V) {
      if (ScheduleCopyableDataMapByInstUser.empty())
        return {};
      const auto It = ScheduleCopyableDataMapByInstUser.find(
          std::make_pair(std::make_pair(User, OperandIdx), V));
      if (It == ScheduleCopyableDataMapByInstUser.end())
        return {};
      SmallVector<ScheduleCopyableData *> Res;
      for (ScheduleCopyableData *SD : It->getSecond()) {
        if (isInSchedulingRegion(*SD))
          Res.push_back(SD);
      }
      return Res;
    }

    /// Returns true if all operands of the given instruction \p User are
    /// replaced by copyable data.
    /// \param User The user instruction.
    /// \param Op The operand, which might be replaced by the copyable data.
    /// \param SLP The SLP tree.
    /// \param NumOps The number of operands used. If the instruction uses the
    /// same operand several times, check for the first use, then the second,
    /// etc.
    bool areAllOperandsReplacedByCopyableData(Instruction *User,
                                              Instruction *Op, BoUpSLP &SLP,
                                              unsigned NumOps) const {
      assert(NumOps > 0 && "No operands");
      if (ScheduleCopyableDataMap.empty())
        return false;
      SmallDenseMap<TreeEntry *, unsigned> PotentiallyReorderedEntriesCount;
      ArrayRef<TreeEntry *> Entries = SLP.getTreeEntries(User);
      if (Entries.empty())
        return false;
      unsigned CurNumOps = 0;
      for (const Use &U : User->operands()) {
        if (U.get() != Op)
          continue;
        ++CurNumOps;
        // Check all tree entries, if they have operands replaced by copyable
        // data.
        for (TreeEntry *TE : Entries) {
          unsigned Inc = 0;
          bool IsNonSchedulableWithParentPhiNode =
              TE->doesNotNeedToSchedule() && TE->UserTreeIndex &&
              TE->UserTreeIndex.UserTE->hasState() &&
              TE->UserTreeIndex.UserTE->State != TreeEntry::SplitVectorize &&
              TE->UserTreeIndex.UserTE->getOpcode() == Instruction::PHI;
          // Count the number of unique phi nodes, which are the parent for
          // parent entry, and exit, if all the unique phis are processed.
          if (IsNonSchedulableWithParentPhiNode) {
            SmallPtrSet<Value *, 4> ParentsUniqueUsers;
            const TreeEntry *ParentTE = TE->UserTreeIndex.UserTE;
            for (Value *V : ParentTE->Scalars) {
              auto *PHI = dyn_cast<PHINode>(V);
              if (!PHI)
                continue;
              if (ParentsUniqueUsers.insert(PHI).second &&
                  is_contained(PHI->incoming_values(), User))
                ++Inc;
            }
          } else {
            Inc = count(TE->Scalars, User);
          }

          // Check if the user is commutative.
          // The commutatives are handled later, as their operands can be
          // reordered.
          // Same applies even for non-commutative cmps, because we can invert
          // their predicate potentially and, thus, reorder the operands.
          bool IsCommutativeUser =
              isCommutative(User) &&
              isCommutableOperand(User, User, U.getOperandNo());
          if (!IsCommutativeUser) {
            Instruction *MainOp = TE->getMatchingMainOpOrAltOp(User);
            IsCommutativeUser =
                isCommutative(MainOp, User) &&
                isCommutableOperand(MainOp, User, U.getOperandNo());
          }
          // The commutative user with the same operands can be safely
          // considered as non-commutative, operands reordering does not change
          // the semantics. Same for cmps with the same operands: inverting
          // the predicate does not change the operand columns in this case.
          assert(
              (!IsCommutativeUser ||
               (((isCommutative(User) && isCommutableOperand(User, User, 0) &&
                  isCommutableOperand(User, User, 1)) ||
                 (isCommutative(TE->getMatchingMainOpOrAltOp(User), User) &&
                  isCommutableOperand(TE->getMatchingMainOpOrAltOp(User), User,
                                      0) &&
                  isCommutableOperand(TE->getMatchingMainOpOrAltOp(User), User,
                                      1))))) &&
              "Expected commutative user with 2 first commutable operands");
          bool IsCommutativeWithSameOps =
              IsCommutativeUser && User->getOperand(0) == User->getOperand(1);
          if ((!IsCommutativeUser || IsCommutativeWithSameOps) &&
              (!isa<CmpInst>(User) ||
               User->getOperand(0) == User->getOperand(1))) {
            if (CurNumOps != NumOps)
              continue;
            // A reassociated node flattens the operand chain, so the operand
            // may be placed in any operand column rather than at the
            // instruction's operand number.
            if (TE->hasReassocScalars()) {
              bool ReplacedByCopyable = false;
              for (auto It = find(TE->Scalars, User); It != TE->Scalars.end();
                   It = find(make_range(std::next(It), TE->Scalars.end()),
                             User)) {
                int Lane = std::distance(TE->Scalars.begin(), It);
                for (unsigned OpIdx : seq<unsigned>(TE->getNumOperands()))
                  ReplacedByCopyable |=
                      TE->getOperand(OpIdx)[Lane] == Op &&
                      getScheduleCopyableData(EdgeInfo(TE, OpIdx), Op);
              }
              if (ReplacedByCopyable)
                continue;
              return false;
            }
            EdgeInfo EI(TE, U.getOperandNo());
            if (getScheduleCopyableData(EI, Op))
              continue;
            return false;
          }
          // Only count the occurrence matching this call's NumOps.
          if (CurNumOps != NumOps)
            continue;
          PotentiallyReorderedEntriesCount.try_emplace(TE, 0)
              .first->getSecond() += Inc;
        }
      }
      if (PotentiallyReorderedEntriesCount.empty())
        return true;
      // Check the commutative/cmp entries.
      for (auto &P : PotentiallyReorderedEntriesCount) {
        SmallPtrSet<Value *, 4> ParentsUniqueUsers;
        bool IsNonSchedulableWithParentPhiNode =
            P.first->doesNotNeedToSchedule() && P.first->UserTreeIndex &&
            P.first->UserTreeIndex.UserTE->hasState() &&
            P.first->UserTreeIndex.UserTE->State != TreeEntry::SplitVectorize &&
            P.first->UserTreeIndex.UserTE->getOpcode() == Instruction::PHI;
        auto *It = find(P.first->Scalars, User);
        do {
          assert(It != P.first->Scalars.end() &&
                 "User is not in the tree entry");
          int Lane = std::distance(P.first->Scalars.begin(), It);
          assert(Lane >= 0 && "Lane is not found");
          if (isa<StoreInst, InsertValueInst>(User) &&
              !P.first->ReorderIndices.empty())
            Lane = P.first->ReorderIndices[Lane];
          assert(Lane < static_cast<int>(P.first->Scalars.size()) &&
                 "Couldn't find extract lane");
          // Count the number of unique phi nodes, which are the parent for
          // parent entry, and exit, if all the unique phis are processed.
          if (IsNonSchedulableWithParentPhiNode) {
            const TreeEntry *ParentTE = P.first->UserTreeIndex.UserTE;
            Value *User = ParentTE->Scalars[Lane];
            if (!ParentsUniqueUsers.insert(User).second) {
              It =
                  find(make_range(std::next(It), P.first->Scalars.end()), User);
              continue;
            }
          }
          // Flattened nodes may place an operand in any column; scan all of
          // them so copyable scheduling does not double-count.
          for (unsigned OpIdx :
               seq<unsigned>(P.first->hasReassocScalars()
                                 ? P.first->getNumOperands()
                                 : getNumberOfPotentiallyCommutativeOps(
                                       P.first->getMainOp()))) {
            if (P.first->getOperand(OpIdx)[Lane] == Op &&
                getScheduleCopyableData(EdgeInfo(P.first, OpIdx), Op))
              --P.getSecond();
          }
          // If parent node is schedulable, it will be handled correctly.
          It = find(make_range(std::next(It), P.first->Scalars.end()), User);
        } while (It != P.first->Scalars.end());
      }
      return all_of(PotentiallyReorderedEntriesCount,
                    [&](const auto &P) { return P.second == NumOps - 1; });
    }

    SmallVector<ScheduleCopyableData *>
    getScheduleCopyableData(const Instruction *I) const {
      if (ScheduleCopyableDataMapByInst.empty())
        return {};
      const auto It = ScheduleCopyableDataMapByInst.find(I);
      if (It == ScheduleCopyableDataMapByInst.end())
        return {};
      SmallVector<ScheduleCopyableData *> Res;
      for (ScheduleCopyableData *SD : It->getSecond()) {
        if (isInSchedulingRegion(*SD))
          Res.push_back(SD);
      }
      return Res;
    }

    SmallVector<ScheduleCopyableData *>
    getScheduleCopyableDataUsers(const Instruction *User) const {
      if (ScheduleCopyableDataMapByUsers.empty())
        return {};
      const auto It = ScheduleCopyableDataMapByUsers.find(User);
      if (It == ScheduleCopyableDataMapByUsers.end())
        return {};
      SmallVector<ScheduleCopyableData *> Res;
      for (ScheduleCopyableData *SD : It->getSecond()) {
        if (isInSchedulingRegion(*SD))
          Res.push_back(SD);
      }
      return Res;
    }

    /// Reordering \p TE permutes its operand columns and may move an operand
    /// between the edges covered and not covered by copyable scheduling
    /// data, making the computed dependency counts stale. Mark the schedule
    /// data of \p TE's copyable-modeled operands for recalculation at the
    /// next bundle scheduling.
    void markCopyableDepsForRecalc(const TreeEntry &TE) {
      for (unsigned OpIdx : seq<unsigned>(TE.getNumOperands()))
        for (Value *V : TE.getOperand(OpIdx))
          if (auto *I = dyn_cast<Instruction>(V))
            if (ScheduleData *SD = getScheduleData(I);
                SD && !getScheduleCopyableData(I).empty())
              RecalcCopyableOperandDeps.insert(SD);
    }

    ScheduleCopyableData &addScheduleCopyableData(const EdgeInfo &EI,
                                                  Instruction *I,
                                                  int SchedulingRegionID,
                                                  ScheduleBundle &Bundle) {
      assert(!getScheduleCopyableData(EI, I) && "already in the map");
      ScheduleCopyableData *CD =
          ScheduleCopyableDataMap
              .try_emplace(std::make_pair(EI, I),
                           std::make_unique<ScheduleCopyableData>(
                               SchedulingRegionID, I, EI, Bundle))
              .first->getSecond()
              .get();
      ScheduleCopyableDataMapByInst[I].push_back(CD);
      if (EI.UserTE) {
        ArrayRef<Value *> Op = EI.UserTE->getOperand(EI.EdgeIdx);
        const auto *It = find(Op, I);
        assert(It != Op.end() && "Lane not set");
        SmallPtrSet<Instruction *, 4> Visited;
        do {
          int Lane = std::distance(Op.begin(), It);
          assert(Lane >= 0 && "Lane not set");
          if (isa<StoreInst, InsertValueInst>(EI.UserTE->Scalars[Lane]) &&
              !EI.UserTE->ReorderIndices.empty())
            Lane = EI.UserTE->ReorderIndices[Lane];
          assert(Lane < static_cast<int>(EI.UserTE->Scalars.size()) &&
                 "Couldn't find extract lane");
          auto *In = cast<Instruction>(EI.UserTE->Scalars[Lane]);
          if (!Visited.insert(In).second) {
            It = find(make_range(std::next(It), Op.end()), I);
            continue;
          }
          ScheduleCopyableDataMapByInstUser
              .try_emplace(std::make_pair(std::make_pair(In, EI.EdgeIdx), I))
              .first->getSecond()
              .push_back(CD);
          ScheduleCopyableDataMapByUsers.try_emplace(I)
              .first->getSecond()
              .insert(CD);
          // Remove extra deps for users, becoming non-immediate users of the
          // instruction. It may happen, if the chain of same copyable elements
          // appears in the tree.
          if (In == I) {
            EdgeInfo UserEI = EI.UserTE->UserTreeIndex;
            if (ScheduleCopyableData *UserCD =
                    getScheduleCopyableData(UserEI, In))
              ScheduleCopyableDataMapByUsers[I].remove(UserCD);
          }
          It = find(make_range(std::next(It), Op.end()), I);
        } while (It != Op.end());
      } else {
        ScheduleCopyableDataMapByUsers.try_emplace(I).first->getSecond().insert(
            CD);
      }
      return *CD;
    }

    ArrayRef<ScheduleBundle *> getScheduleBundles(Value *V) const {
      auto *I = dyn_cast<Instruction>(V);
      if (!I)
        return {};
      auto It = ScheduledBundles.find(I);
      if (It == ScheduledBundles.end())
        return {};
      return It->getSecond();
    }

    /// Returns true if the entity is in the scheduling region.
    bool isInSchedulingRegion(const ScheduleEntity &SD) const {
      if (const auto *Data = dyn_cast<ScheduleData>(&SD))
        return Data->getSchedulingRegionID() == SchedulingRegionID;
      if (const auto *CD = dyn_cast<ScheduleCopyableData>(&SD))
        return CD->getSchedulingRegionID() == SchedulingRegionID;
      return all_of(cast<ScheduleBundle>(SD).getBundle(),
                    [&](const ScheduleEntity *BundleMember) {
                      return isInSchedulingRegion(*BundleMember);
                    });
    }

    /// Marks an instruction as scheduled and puts all dependent ready
    /// instructions into the ready-list.
    template <typename ReadyListType>
    void schedule(const BoUpSLP &R, const InstructionsState &S,
                  const EdgeInfo &EI, ScheduleEntity *Data,
                  ReadyListType &ReadyList) {
      auto ProcessBundleMember = [&](ScheduleEntity *BundleMember,
                                     ArrayRef<ScheduleBundle *> Bundles) {
        // Handle the def-use chain dependencies.

        // Decrement the unscheduled counter and insert to ready list if ready.
        auto DecrUnsched = [&](auto *Data, bool IsControl = false) {
          if ((IsControl || Data->hasValidDependencies()) &&
              Data->incrementUnscheduledDeps(-1) == 0) {
            // There are no more unscheduled dependencies after
            // decrementing, so we can put the dependent instruction
            // into the ready list.
            SmallVector<ScheduleBundle *, 1> CopyableBundle;
            ArrayRef<ScheduleBundle *> Bundles;
            if (auto *CD = dyn_cast<ScheduleCopyableData>(Data)) {
              CopyableBundle.push_back(&CD->getBundle());
              Bundles = CopyableBundle;
            } else {
              Bundles = getScheduleBundles(Data->getInst());
            }
            if (!Bundles.empty()) {
              for (ScheduleBundle *Bundle : Bundles) {
                if (Bundle->unscheduledDepsInBundle() == 0) {
                  assert(!Bundle->isScheduled() &&
                         "already scheduled bundle gets ready");
                  ReadyList.insert(Bundle);
                  LLVM_DEBUG(dbgs()
                             << "SLP:    gets ready: " << *Bundle << "\n");
                }
              }
              return;
            }
            assert(!Data->isScheduled() &&
                   "already scheduled bundle gets ready");
            assert(!isa<ScheduleCopyableData>(Data) &&
                   "Expected non-copyable data");
            ReadyList.insert(Data);
            LLVM_DEBUG(dbgs() << "SLP:    gets ready: " << *Data << "\n");
          }
        };

        auto DecrUnschedForInst = [&](Instruction *User, unsigned OpIdx,
                                      Instruction *I) {
          if (!ScheduleCopyableDataMap.empty()) {
            SmallVector<ScheduleCopyableData *> CopyableData =
                getScheduleCopyableData(User, OpIdx, I);
            bool ReleasedAsCopyable = false;
            for (ScheduleCopyableData *CD : CopyableData) {
              // Copyable elements modeled on a copyable user lane depend on
              // the user's copyable scheduling data, not on the user itself,
              // and are released when that copyable data is scheduled. The
              // user's own schedule data still carries the def-use dependency
              // in this case, so it must be released below.
              if (CD->getEdgeInfo().UserTE->isCopyableElement(User))
                continue;
              DecrUnsched(CD, /*IsControl=*/false);
              ReleasedAsCopyable = true;
            }
            if (ReleasedAsCopyable)
              return;
          }
          if (ScheduleData *OpSD = getScheduleData(I))
            DecrUnsched(OpSD, /*IsControl=*/false);
        };

        // If BundleMember is a vector bundle, its operands may have been
        // reordered during buildTree(). We therefore need to get its operands
        // through the TreeEntry.
        if (!Bundles.empty()) {
          auto *In = BundleMember->getInst();
          // Count uses of each instruction operand.
          SmallDenseMap<const Instruction *, unsigned> OperandsUses;
          unsigned TotalOpCount = 0;
          if (isa<ScheduleCopyableData>(BundleMember)) {
            // Copyable data is used only once (uses itself).
            TotalOpCount = OperandsUses[In] = 1;
          } else {
            for (const Use &U : In->operands()) {
              if (auto *I = dyn_cast<Instruction>(U.get())) {
                auto Res = OperandsUses.try_emplace(I, 0);
                unsigned ExtraDeps = 1;
                // Count all expanded operands in the binops.
                for (ScheduleBundle *Bundle : Bundles) {
                  if (const TreeEntry *TE = Bundle->getTreeEntry()) {
                    if (TE->isExpandedBinOp(In))
                      ++ExtraDeps;
                  } else if (S.isExpandedBinOp(In)) {
                    ++ExtraDeps;
                  }
                }
                Res.first->getSecond() += ExtraDeps;
                TotalOpCount += ExtraDeps;
              }
            }
          }
          // Tracks whether the bundle member instruction itself shows up in
          // some operand column of its node (only copyable elements modeled
          // through their own operands, like absorbed fmuls, do not).
          bool FoundInOpColumns = false;
          // Decrement the unscheduled counter and insert to ready list if
          // ready.
          auto DecrUnschedForInst =
              [&](Instruction *I, TreeEntry *UserTE, unsigned OpIdx,
                  SmallDenseSet<std::pair<const ScheduleEntity *, unsigned>>
                      &Checked,
                  bool IsExpandedOperand = false,
                  bool CopyableDepsOnly = false) {
                if (!ScheduleCopyableDataMap.empty()) {
                  const EdgeInfo EI = {UserTE, OpIdx};
                  if (ScheduleCopyableData *CD =
                          getScheduleCopyableData(EI, I)) {
                    if (!Checked.insert(std::make_pair(CD, OpIdx)).second)
                      return;
                    DecrUnsched(CD, /*IsControl=*/false);
                    return;
                  }
                }
                if (CopyableDepsOnly)
                  return;
                auto It = OperandsUses.find(I);
                if (It == OperandsUses.end()) {
                  // Column value may be a peeled intermediate, not a direct
                  // operand of In; its deps are released when it is scheduled.
                  LLVM_DEBUG(dbgs() << "SLP:   operand " << *I
                                    << " not modeled as a direct operand of "
                                    << *In << ", skipping.\n");
                  return;
                }
                if (It->second > 0) {
                  if (ScheduleData *OpSD = getScheduleData(I)) {
                    if (!IsExpandedOperand &&
                        !Checked.insert(std::make_pair(OpSD, OpIdx)).second)
                      return;
                    --It->getSecond();
                    assert(TotalOpCount > 0 && "No more operands to decrement");
                    --TotalOpCount;
                    DecrUnsched(OpSD, /*IsControl=*/false);
                  } else {
                    --It->getSecond();
                    assert(TotalOpCount > 0 && "No more operands to decrement");
                    --TotalOpCount;
                  }
                }
              };

          SmallDenseSet<std::pair<const ScheduleEntity *, unsigned>> Checked;
          for (ScheduleBundle *Bundle : Bundles) {
            if (ScheduleCopyableDataMap.empty() && TotalOpCount == 0)
              break;
            SmallPtrSet<Value *, 4> ParentsUniqueUsers;
            // Need to search for the lane since the tree entry can be
            // reordered.
            auto *It = find(Bundle->getTreeEntry()->Scalars, In);
            bool IsNonSchedulableWithParentPhiNode =
                Bundle->getTreeEntry()->doesNotNeedToSchedule() &&
                Bundle->getTreeEntry()->UserTreeIndex &&
                Bundle->getTreeEntry()->UserTreeIndex.UserTE->hasState() &&
                Bundle->getTreeEntry()->UserTreeIndex.UserTE->State !=
                    TreeEntry::SplitVectorize &&
                Bundle->getTreeEntry()->UserTreeIndex.UserTE->getOpcode() ==
                    Instruction::PHI;
            do {
              int Lane =
                  std::distance(Bundle->getTreeEntry()->Scalars.begin(), It);
              assert(Lane >= 0 && "Lane not set");
              if (isa<StoreInst, InsertValueInst>(In) &&
                  !Bundle->getTreeEntry()->ReorderIndices.empty())
                Lane = Bundle->getTreeEntry()->ReorderIndices[Lane];
              assert(Lane < static_cast<int>(
                                Bundle->getTreeEntry()->Scalars.size()) &&
                     "Couldn't find extract lane");

              // Since vectorization tree is being built recursively this
              // assertion ensures that the tree entry has all operands set
              // before reaching this code. Couple of exceptions known at the
              // moment are extracts where their second (immediate) operand is
              // not added. Since immediates do not affect scheduler behavior
              // this is considered okay.
              assert(
                  In &&
                  (isa<ExtractValueInst, ExtractElementInst, CallBase>(In) ||
                   In->getNumOperands() ==
                       Bundle->getTreeEntry()->getNumOperands() ||
                   (isa<ZExtInst>(In) && Bundle->getTreeEntry()->getOpcode() ==
                                             Instruction::Select) ||
                   Bundle->getTreeEntry()->isCopyableElement(In) ||
                   Bundle->getTreeEntry()->hasReassocScalars()) &&
                  "Missed TreeEntry operands?");

              // Count the number of unique phi nodes, which are the parent
              // entry, and handle the non-copyable deps only on the first lane
              // for each such phi. Copyable deps are counted per operand column
              // lane and are released on every lane.
              bool CopyableDepsOnly =
                  IsNonSchedulableWithParentPhiNode &&
                  !ParentsUniqueUsers
                       .insert(Bundle->getTreeEntry()
                                   ->UserTreeIndex.UserTE->Scalars[Lane])
                       .second;

              // A blended-load operand node is the synthetic blend mask, not an
              // IR operand of the load. Use the real pointer operand for
              // scheduling so the def-use counters stay balanced; the mask is
              // available earlier through the pointer's select.
              bool IsBlended = Bundle->getTreeEntry()->State ==
                               TreeEntry::BlendedLoadVectorize;
              for (unsigned OpIdx :
                   seq<unsigned>(Bundle->getTreeEntry()->getNumOperands()))
                if (auto *I = dyn_cast<Instruction>(
                        IsBlended ? In->getOperand(OpIdx)
                                  : Bundle->getTreeEntry()->getOperand(
                                        OpIdx)[Lane])) {
                  FoundInOpColumns |= (I == In) && !CopyableDepsOnly;
                  LLVM_DEBUG(dbgs() << "SLP:   check for readiness (def): "
                                    << *I << "\n");
                  DecrUnschedForInst(
                      I, Bundle->getTreeEntry(), OpIdx, Checked,
                      Bundle->getTreeEntry()->isExpandedOperand(In, OpIdx),
                      /*CopyableDepsOnly=*/CopyableDepsOnly);
                }
              // If parent node is schedulable, it will be handled correctly.
              if (Bundle->getTreeEntry()->isCopyableElement(In))
                break;
              It = std::find(std::next(It),
                             Bundle->getTreeEntry()->Scalars.end(), In);
            } while (It != Bundle->getTreeEntry()->Scalars.end());
          }
          // A copyable element absorbed into its user modeling (e.g. a
          // copyable fmul turned into fmuladd(a, b, -0.0)) does not appear in
          // the operand columns of its own node, so the scan above never
          // releases the schedule data of the copyable instruction itself.
          // Release it here to keep the unscheduled-deps counters balanced,
          // consuming its self-use count so the reassociated-operand release
          // below cannot release the same schedule data twice.
          if (isa<ScheduleCopyableData>(BundleMember) && !FoundInOpColumns) {
            auto UseIt = OperandsUses.find(In);
            if (UseIt != OperandsUses.end() && UseIt->second > 0) {
              --UseIt->getSecond();
              --TotalOpCount;
            }
            if (ScheduleData *OpSD = getScheduleData(In))
              DecrUnsched(OpSD, /*IsControl=*/false);
          }
          // Vector intrinsics may keep some arguments scalar (e.g. the
          // exponent of llvm.powi). Such scalar arguments are not modeled as
          // tree-entry operands, so the per-lane loop above never releases the
          // dependency that calculateDependencies() registered for the
          // definition feeding such an argument. Release it here to keep the
          // unscheduled-deps counters balanced; otherwise the operand's bundle
          // may never become ready and scheduling would assert.
          if (TotalOpCount > 0) {
            if (auto *CI = dyn_cast<CallInst>(In)) {
              Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, R.TLI);
              for (unsigned ArgIdx : seq<unsigned>(CI->arg_size())) {
                if (!isVectorIntrinsicWithScalarOpAtArg(ID, ArgIdx, R.TTI))
                  continue;
                auto *OpI = dyn_cast<Instruction>(CI->getArgOperand(ArgIdx));
                if (!OpI)
                  continue;
                auto UseIt = OperandsUses.find(OpI);
                if (UseIt == OperandsUses.end() || UseIt->second == 0)
                  continue;
                --UseIt->getSecond();
                --TotalOpCount;
                if (ScheduleData *OpSD = getScheduleData(OpI)) {
                  LLVM_DEBUG(dbgs()
                             << "SLP:   check for readiness (scalar arg): "
                             << *OpI << "\n");
                  DecrUnsched(OpSD, /*IsControl=*/false);
                }
              }
            }
            // Peeled intermediates stay as direct operands but drop out of
            // operand columns; release their scheduling deps here.
            for (const ScheduleBundle *Bundle : Bundles) {
              if (TotalOpCount == 0)
                break;
              TreeEntry *TE = Bundle->getTreeEntry();
              if (!TE->hasReassocScalars())
                continue;
              for (Value *V : TE->getReassocScalars()) {
                auto *OpI = dyn_cast<Instruction>(V);
                if (!OpI)
                  continue;
                auto UseIt = OperandsUses.find(OpI);
                if (UseIt == OperandsUses.end() || UseIt->second == 0)
                  continue;
                LLVM_DEBUG(dbgs() << "SLP:   check for readiness "
                                     "(reassociated operand): "
                                  << *OpI << "\n");
                // Copyable deps may live on per-edge ScheduleCopyableData.
                bool ReleasedAsCopyable = false;
                if (!ScheduleCopyableDataMap.empty()) {
                  for (const Use &U : In->operands()) {
                    if (U.get() != OpI)
                      continue;
                    for (ScheduleCopyableData *CD :
                         getScheduleCopyableData(In, U.getOperandNo(), OpI)) {
                      // Deps of reassoc scalars modeled as copyable tree
                      // operands are released by the operand scan above;
                      // release each remaining dep only once.
                      if (Checked.insert(std::make_pair(CD, U.getOperandNo()))
                              .second)
                        DecrUnsched(CD, /*IsControl=*/false);
                    }
                  }
                  // The dep is released through copyable data only if this
                  // very entry models the scalar as a copyable operand on one
                  // of its edges, mirroring the dependency calculation;
                  // copyable data on some other entry's edge does not cover
                  // the dep registered for this entry.
                  for (auto It = find(TE->Scalars, In);
                       It != TE->Scalars.end() && !ReleasedAsCopyable;
                       It = find(make_range(std::next(It), TE->Scalars.end()),
                                 In)) {
                    int Lane = std::distance(TE->Scalars.begin(), It);
                    for (unsigned OpIdx : seq<unsigned>(TE->getNumOperands()))
                      ReleasedAsCopyable |=
                          TE->getOperand(OpIdx)[Lane] == OpI &&
                          getScheduleCopyableData(EdgeInfo(TE, OpIdx), OpI);
                  }
                }
                if (!ReleasedAsCopyable) {
                  if (ScheduleData *OpSD = getScheduleData(OpI))
                    for (unsigned I = 0, E = UseIt->second; I != E; ++I)
                      DecrUnsched(OpSD, /*IsControl=*/false);
                }
                TotalOpCount -= UseIt->second;
                UseIt->second = 0;
              }
            }
          }
        } else {
          // If BundleMember is a stand-alone instruction, no operand reordering
          // has taken place, so we directly access its operands.
          for (Use &U : BundleMember->getInst()->operands()) {
            if (auto *I = dyn_cast<Instruction>(U.get())) {
              LLVM_DEBUG(dbgs()
                         << "SLP:   check for readiness (def): " << *I << "\n");
              DecrUnschedForInst(BundleMember->getInst(), U.getOperandNo(), I);
            }
          }
        }
        // Handle the memory dependencies.
        auto *SD = dyn_cast<ScheduleData>(BundleMember);
        if (!SD)
          return;
        SmallPtrSet<const ScheduleData *, 4> VisitedMemory;
        for (ScheduleData *MemoryDep : SD->getMemoryDependencies()) {
          if (!VisitedMemory.insert(MemoryDep).second)
            continue;
          // There are no more unscheduled dependencies after decrementing,
          // so we can put the dependent instruction into the ready list.
          LLVM_DEBUG(dbgs() << "SLP:   check for readiness (mem): "
                            << *MemoryDep << "\n");
          DecrUnsched(MemoryDep);
        }
        // Handle the control dependencies.
        SmallPtrSet<const ScheduleData *, 4> VisitedControl;
        for (ScheduleData *Dep : SD->getControlDependencies()) {
          if (!VisitedControl.insert(Dep).second)
            continue;
          // There are no more unscheduled dependencies after decrementing,
          // so we can put the dependent instruction into the ready list.
          LLVM_DEBUG(dbgs()
                     << "SLP:   check for readiness (ctrl): " << *Dep << "\n");
          DecrUnsched(Dep, /*IsControl=*/true);
        }
      };
      if (auto *SD = dyn_cast<ScheduleData>(Data)) {
        SD->setScheduled(/*Scheduled=*/true);
        LLVM_DEBUG(dbgs() << "SLP:   schedule " << *SD << "\n");
        SmallVector<std::unique_ptr<ScheduleBundle>> PseudoBundles;
        SmallVector<ScheduleBundle *> Bundles;
        Instruction *In = SD->getInst();
        ArrayRef<TreeEntry *> Entries = R.getTreeEntries(In);
        if (!Entries.empty()) {
          for (TreeEntry *TE : Entries) {
            if (!isa<ExtractValueInst, ExtractElementInst, CallBase>(In) &&
                In->getNumOperands() != TE->getNumOperands() &&
                !TE->hasReassocScalars())
              continue;
            auto &BundlePtr =
                PseudoBundles.emplace_back(std::make_unique<ScheduleBundle>());
            BundlePtr->setTreeEntry(TE);
            BundlePtr->add(SD);
            Bundles.push_back(BundlePtr.get());
          }
        }
        ProcessBundleMember(SD, Bundles);
      } else {
        ScheduleBundle &Bundle = *cast<ScheduleBundle>(Data);
        Bundle.setScheduled(/*Scheduled=*/true);
        LLVM_DEBUG(dbgs() << "SLP:   schedule " << Bundle << "\n");
        auto AreAllBundlesScheduled =
            [&](const ScheduleEntity *SD,
                ArrayRef<ScheduleBundle *> SDBundles) {
              if (isa<ScheduleCopyableData>(SD))
                return true;
              return !SDBundles.empty() &&
                     all_of(SDBundles, [&](const ScheduleBundle *SDBundle) {
                       return SDBundle->isScheduled();
                     });
            };
        for (ScheduleEntity *SD : Bundle.getBundle()) {
          ArrayRef<ScheduleBundle *> SDBundles;
          if (!isa<ScheduleCopyableData>(SD))
            SDBundles = getScheduleBundles(SD->getInst());
          if (!AreAllBundlesScheduled(SD, SDBundles))
            continue;
          SD->setScheduled(/*Scheduled=*/true);
          Instruction *In = SD->getInst();
          // The instruction may also belong to tree entries that do not need
          // scheduling (e.g. all their values are used outside the block), so
          // no schedule bundle is registered for them. Such an entry can still
          // model one of this instruction's operands as a copyable element, or
          // model the instruction itself as an expanded binop, registered on
          // that non-scheduled parent edge. That dependency would never be
          // decremented when the instruction is scheduled through a different
          // bundle, leaving the operand's bundle permanently unscheduled and
          // tripping the unscheduled-deps assertion. Add pseudo-bundles for
          // these missing tree entries, so their operand dependencies are
          // decremented here as well. Real operand dependencies are protected
          // against double counting by the per-operand use counter.
          if (isa<ScheduleCopyableData>(SD) ||
              (ScheduleCopyableDataMap.empty() &&
               none_of(R.getTreeEntries(In), [&](const TreeEntry *TE) {
                 return TE->isExpandedBinOp(In);
               }))) {
            ProcessBundleMember(SD, isa<ScheduleCopyableData>(SD) ? &Bundle
                                                                  : SDBundles);
            continue;
          }
          SmallVector<std::unique_ptr<ScheduleBundle>> PseudoBundles;
          SmallVector<ScheduleBundle *> AllBundles(SDBundles.begin(),
                                                   SDBundles.end());
          for (TreeEntry *TE : R.getTreeEntries(In)) {
            if (TE->isCopyableElement(In))
              continue;
            if (!isa<ExtractValueInst, ExtractElementInst, CallBase>(In) &&
                In->getNumOperands() != TE->getNumOperands() &&
                !TE->hasReassocScalars())
              continue;
            if (any_of(SDBundles, [&](const ScheduleBundle *SDBundle) {
                  return SDBundle->getTreeEntry() == TE;
                }))
              continue;
            ScheduleBundle &PseudoBundle =
                *PseudoBundles.emplace_back(std::make_unique<ScheduleBundle>());
            PseudoBundle.setTreeEntry(TE);
            PseudoBundle.add(SD);
            AllBundles.push_back(&PseudoBundle);
          }
          ProcessBundleMember(SD, AllBundles);
        }
      }
    }

    /// Verify basic self consistency properties of the data structure.
    void verify() {
      if (!ScheduleStart)
        return;

      assert(ScheduleStart->getParent() == ScheduleEnd->getParent() &&
             ScheduleStart->comesBefore(ScheduleEnd) &&
             "Not a valid scheduling region?");

      for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
        ArrayRef<ScheduleBundle *> Bundles = getScheduleBundles(I);
        if (!Bundles.empty()) {
          for (ScheduleBundle *Bundle : Bundles) {
            assert(isInSchedulingRegion(*Bundle) &&
                   "primary schedule data not in window?");
            Bundle->verify();
          }
          continue;
        }
        auto *SD = getScheduleData(I);
        if (!SD)
          continue;
        assert(isInSchedulingRegion(*SD) &&
               "primary schedule data not in window?");
        SD->verify();
      }

      assert(all_of(ReadyInsts,
                    [](const ScheduleEntity *Bundle) {
                      return Bundle->isReady();
                    }) &&
             "item in ready list not ready?");
    }

    /// Put all instructions into the ReadyList which are ready for scheduling.
    template <typename ReadyListType>
    void initialFillReadyList(ReadyListType &ReadyList) {
      SmallPtrSet<ScheduleBundle *, 16> Visited;
      for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
        ScheduleData *SD = getScheduleData(I);
        if (SD && SD->hasValidDependencies() && SD->isReady()) {
          if (ArrayRef<ScheduleBundle *> Bundles = getScheduleBundles(I);
              !Bundles.empty()) {
            for (ScheduleBundle *Bundle : Bundles) {
              if (!Visited.insert(Bundle).second)
                continue;
              if (Bundle->hasValidDependencies() && Bundle->isReady()) {
                ReadyList.insert(Bundle);
                LLVM_DEBUG(dbgs() << "SLP:    initially in ready list: "
                                  << *Bundle << "\n");
              }
            }
            continue;
          }
          ReadyList.insert(SD);
          LLVM_DEBUG(dbgs()
                     << "SLP:    initially in ready list: " << *SD << "\n");
        }
      }
    }

    /// Build a bundle from the ScheduleData nodes corresponding to the
    /// scalar instruction for each lane.
    /// \param VL The list of scalar instructions.
    /// \param S The state of the instructions.
    /// \param EI The edge in the SLP graph or the user node/operand number.
    ScheduleBundle &buildBundle(ArrayRef<Value *> VL,
                                const InstructionsState &S, const EdgeInfo &EI);

    /// Checks if a bundle of instructions can be scheduled, i.e. has no
    /// cyclic dependencies. This is only a dry-run, no instructions are
    /// actually moved at this stage.
    /// \returns the scheduling bundle. The returned Optional value is not
    /// std::nullopt if \p VL is allowed to be scheduled.
    std::optional<ScheduleBundle *>
    tryScheduleBundle(ArrayRef<Value *> VL, BoUpSLP *SLP,
                      const InstructionsState &S, const EdgeInfo &EI);

    /// Allocates schedule data chunk.
    ScheduleData *allocateScheduleDataChunks();

    /// Extends the scheduling region so that V is inside the region.
    /// \returns true if the region size is within the limit.
    bool extendSchedulingRegion(Value *V, const InstructionsState &S);

    /// Initialize the ScheduleData structures for new instructions in the
    /// scheduling region.
    void initScheduleData(Instruction *FromI, Instruction *ToI,
                          ScheduleData *PrevLoadStore,
                          ScheduleData *NextLoadStore);

    /// Updates the dependency information of a bundle and of all instructions/
    /// bundles which depend on the original bundle.
    void calculateDependencies(ScheduleBundle &Bundle, bool InsertInReadyList,
                               BoUpSLP *SLP,
                               const SmallPtrSetImpl<Value *> &ExpandedOps,
                               ArrayRef<ScheduleData *> ControlDeps = {});

    /// Sets all instruction in the scheduling region to un-scheduled.
    void resetSchedule();

    BasicBlock *BB;

    /// Simple memory allocation for ScheduleData.
    SmallVector<std::unique_ptr<ScheduleData[]>> ScheduleDataChunks;

    /// The size of a ScheduleData array in ScheduleDataChunks.
    int ChunkSize;

    /// The allocator position in the current chunk, which is the last entry
    /// of ScheduleDataChunks.
    int ChunkPos;

    /// Attaches ScheduleData to Instruction.
    /// Note that the mapping survives during all vectorization iterations, i.e.
    /// ScheduleData structures are recycled.
    SmallDenseMap<Instruction *, ScheduleData *> ScheduleDataMap;

    /// Attaches ScheduleCopyableData to EdgeInfo (UserTreeEntry + operand
    /// number) and the operand instruction, represented as copyable element.
    SmallDenseMap<std::pair<EdgeInfo, const Value *>,
                  std::unique_ptr<ScheduleCopyableData>>
        ScheduleCopyableDataMap;

    /// Represents mapping between instruction and all related
    /// ScheduleCopyableData (for all uses in the tree, represenedt as copyable
    /// element). The SLP tree may contain several representations of the same
    /// instruction.
    SmallDenseMap<const Instruction *, SmallVector<ScheduleCopyableData *>>
        ScheduleCopyableDataMapByInst;

    /// Represents mapping between user value and operand number, the operand
    /// value and all related ScheduleCopyableData. The relation is 1:n, because
    /// the same user may refernce the same operand in different tree entries
    /// and the operand may be modelled by the different copyable data element.
    SmallDenseMap<std::pair<std::pair<const Value *, unsigned>, const Value *>,
                  SmallVector<ScheduleCopyableData *>>
        ScheduleCopyableDataMapByInstUser;

    /// Represents mapping between instruction and all related
    /// ScheduleCopyableData. It represents the mapping between the actual
    /// instruction and the last copyable data element in the chain. E.g., if
    /// the graph models the following instructions:
    /// %0 = non-add instruction ...
    /// ...
    /// %4 = add %3, 1
    /// %5 = add %4, 1
    /// %6 = insertelement poison, %0, 0
    /// %7 = insertelement %6, %5, 1
    /// And the graph is modeled as:
    /// [%5, %0] -> [%4, copyable %0 <0> ] -> [%3, copyable %0 <1> ]
    ///          -> [1, 0]                 -> [%1, 0]
    ///
    /// this map will map %0 only to the copyable element <1>, which is the last
    /// user (direct user of the actual instruction). <0> uses <1>, so <1> will
    /// keep the map to <0>, not the %0.
    SmallDenseMap<const Instruction *,
                  SmallSetVector<ScheduleCopyableData *, 4>>
        ScheduleCopyableDataMapByUsers;

    /// Attaches ScheduleBundle to Instruction.
    SmallDenseMap<Instruction *, SmallVector<ScheduleBundle *>>
        ScheduledBundles;
    /// The list of ScheduleBundles.
    SmallVector<std::unique_ptr<ScheduleBundle>> ScheduledBundlesList;

    /// The ready-list for scheduling (only used for the dry-run).
    SetVector<ScheduleEntity *> ReadyInsts;

    /// The first instruction of the scheduling region.
    Instruction *ScheduleStart = nullptr;

    /// The first instruction _after_ the scheduling region.
    Instruction *ScheduleEnd = nullptr;

    /// The first memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *FirstLoadStoreInRegion = nullptr;

    /// The last memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *LastLoadStoreInRegion = nullptr;

    /// Is there an llvm.stacksave or llvm.stackrestore in the scheduling
    /// region?  Used to optimize the dependence calculation for the
    /// common case where there isn't.
    bool RegionHasStackSave = false;

    /// The current size of the scheduling region.
    int ScheduleRegionSize = 0;

    /// The maximum size allowed for the scheduling region.
    int ScheduleRegionSizeLimit;

    /// Operands that are modeled as copyable elements in a previously built
    /// vectorized node and that are used directly by another,
    /// not-yet-registered node sharing a schedulable instruction with it. Their
    /// direct dependencies must be recomputed at the next bundle scheduling,
    /// when the new node is already registered in the tree, so that the direct
    /// use is accounted for. If the new node is the last scheduled bundle and
    /// no further scheduling consumes this list, the leftover entries are
    /// dropped on the next region reset and the dependencies are recomputed
    /// against the full tree in scheduleBlock instead. A set is used to avoid
    /// recomputing the same operand more than once.
    SmallSetVector<ScheduleData *, 8> RecalcCopyableOperandDeps;

    /// Ordered pairs (Src, Dst) of memory instructions whose may-alias
    /// dependency has been dropped in favor of a runtime alias check.
    SmallDenseSet<std::pair<Instruction *, Instruction *>, 8> IgnoredMemDeps;

    /// The ID of the scheduling region. For a new vectorization iteration this
    /// is incremented which "removes" all ScheduleData from the region.
    /// Make sure that the initial SchedulingRegionID is greater than the
    /// initial SchedulingRegionID in ScheduleData (which is 0).
    int SchedulingRegionID = 1;
  };

  /// Attaches the BlockScheduling structures to basic blocks.
  MapVector<BasicBlock *, std::unique_ptr<BlockScheduling>> BlocksSchedules;

  /// Performs the "real" scheduling. Done before vectorization is actually
  /// performed in a basic block.
  void scheduleBlock(const BoUpSLP &R, BlockScheduling *BS);

  /// List of users to ignore during scheduling and that don't need extracting.
  const SmallDenseSet<Value *> *UserIgnoreList = nullptr;

  /// Narrowed reduction chain instructions, dropped together with the
  /// reduction. Subset of UserIgnoreList.
  SmallPtrSet<Value *, 4> NarrowedChainInsts;

  /// A DenseMapInfo implementation for holding DenseMaps and DenseSets of
  /// sorted SmallVectors of unsigned.
  struct OrdersTypeDenseMapInfo {
    static unsigned getHashValue(const OrdersType &V) {
      return static_cast<unsigned>(hash_combine_range(V));
    }

    static bool isEqual(const OrdersType &LHS, const OrdersType &RHS) {
      return LHS == RHS;
    }
  };

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;
  TargetLibraryInfo *TLI;
  LoopInfo *LI;
  DominatorTree *DT;
  AssumptionCache *AC;
  DemandedBits *DB;
  const DataLayout *DL;
  OptimizationRemarkEmitter *ORE;
  /// Cached cost-model mode for this function.
  /// If -Os/-Oz, use CodeSize. Otherwise use RecipThroughput.
  const TargetTransformInfo::TargetCostKind CostKind;

  unsigned MaxVecRegSize; // This is set by TTI or overridden by cl::opt.
  unsigned MinVecRegSize; // Set by cl::opt (default: 128).

  /// Instruction builder to construct the vectorized tree.
  IRBuilder<TargetFolder> Builder;

  /// A map of scalar integer values to the smallest bit width with which they
  /// can legally be represented. The values map to (width, signed) pairs,
  /// where "width" indicates the minimum bit width and "signed" is True if the
  /// value must be signed-extended, rather than zero-extended, back to its
  /// original width.
  DenseMap<const TreeEntry *, std::pair<uint64_t, bool>> MinBWs;

  /// Final size of the reduced vector, if the current graph represents the
  /// input for the reduction and it was possible to narrow the size of the
  /// reduction.
  unsigned ReductionBitWidth = 0;

  /// Canonical graph size before the transformations.
  unsigned BaseGraphSize = 1;

  /// If the tree contains any zext/sext/trunc nodes, contains max-min pair of
  /// type sizes, used in the tree.
  std::optional<std::pair<unsigned, unsigned>> CastMaxMinBWSizes;

  /// Indices of the vectorized nodes, which supposed to be the roots of the new
  /// bitwidth analysis attempt, like trunc, IToFP or ICmp.
  DenseSet<unsigned> ExtraBitWidthNodes;
};

using slpvectorizer::BoUpSLP;
using slpvectorizer::isSplat;

template <> struct DenseMapInfo<BoUpSLP::EdgeInfo> {
  using FirstInfo = DenseMapInfo<BoUpSLP::TreeEntry *>;
  using SecondInfo = DenseMapInfo<unsigned>;
  static unsigned getHashValue(const BoUpSLP::EdgeInfo &Val) {
    return detail::combineHashValue(FirstInfo::getHashValue(Val.UserTE),
                                    SecondInfo::getHashValue(Val.EdgeIdx));
  }

  static bool isEqual(const BoUpSLP::EdgeInfo &LHS,
                      const BoUpSLP::EdgeInfo &RHS) {
    return LHS == RHS;
  }
};

template <> struct GraphTraits<BoUpSLP *> {
  using TreeEntry = BoUpSLP::TreeEntry;

  /// NodeRef has to be a pointer per the GraphWriter.
  using NodeRef = TreeEntry *;

  using ContainerTy = BoUpSLP::TreeEntry::VecTreeTy;

  /// Add the VectorizableTree to the index iterator to be able to return
  /// TreeEntry pointers.
  struct ChildIteratorType
      : public iterator_adaptor_base<
            ChildIteratorType, SmallVector<BoUpSLP::EdgeInfo, 1>::iterator> {
    ContainerTy &VectorizableTree;

    ChildIteratorType(SmallVector<BoUpSLP::EdgeInfo, 1>::iterator W,
                      ContainerTy &VT)
        : ChildIteratorType::iterator_adaptor_base(W), VectorizableTree(VT) {}

    NodeRef operator*() { return I->UserTE; }
  };

  static NodeRef getEntryNode(BoUpSLP &R) { return &R.getRootNode(); }

  static ChildIteratorType child_begin(NodeRef N) {
    return {&N->UserTreeIndex, N->Container};
  }

  static ChildIteratorType child_end(NodeRef N) {
    return {&N->UserTreeIndex + 1, N->Container};
  }

  /// For the node iterator we just need to turn the TreeEntry iterator into a
  /// TreeEntry* iterator so that it dereferences to NodeRef.
  class nodes_iterator {
    using ItTy = ContainerTy::iterator;
    ItTy It;

  public:
    nodes_iterator(const ItTy &It2) : It(It2) {}
    NodeRef operator*() { return It->get(); }
    nodes_iterator operator++() {
      ++It;
      return *this;
    }
    bool operator!=(const nodes_iterator &N2) const { return N2.It != It; }
  };

  static nodes_iterator nodes_begin(BoUpSLP *R) {
    return nodes_iterator(R->VectorizableTree.begin());
  }

  static nodes_iterator nodes_end(BoUpSLP *R) {
    return nodes_iterator(R->VectorizableTree.end());
  }

  static unsigned size(BoUpSLP *R) { return R->VectorizableTree.size(); }
};

template <> struct DOTGraphTraits<BoUpSLP *> : public DefaultDOTGraphTraits {
  using TreeEntry = BoUpSLP::TreeEntry;

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  std::string getNodeLabel(const TreeEntry *Entry, const BoUpSLP *R) {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << Entry->Idx << ".\n";
    if (isSplat(Entry->Scalars))
      OS << "<splat> ";
    for (auto *V : Entry->Scalars) {
      OS << *V;
      if (llvm::any_of(R->ExternalUses, [&](const BoUpSLP::ExternalUser &EU) {
            return EU.Scalar == V;
          }))
        OS << " <extract>";
      OS << "\n";
    }
    return Str;
  }

  static std::string getNodeAttributes(const TreeEntry *Entry,
                                       const BoUpSLP *) {
    if (Entry->isGather())
      return "color=red";
    if (Entry->State == TreeEntry::ScatterVectorize ||
        Entry->State == TreeEntry::StridedVectorize ||
        Entry->State == TreeEntry::ExpandVectorize ||
        Entry->State == TreeEntry::CompressVectorize ||
        Entry->State == TreeEntry::BlendedLoadVectorize)
      return "color=blue";
    return "";
  }
};

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTREE_H
