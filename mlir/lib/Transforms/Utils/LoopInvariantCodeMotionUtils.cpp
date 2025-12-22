//===- LoopInvariantCodeMotionUtils.cpp - LICM Utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the core LICM algorithm.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include <queue>
#include <utility>

#define DEBUG_TYPE "licm"

using namespace mlir;

/// Checks whether the given op can be hoisted by checking that
/// - the op and none of its contained operations depend on values inside of the
///   loop (by means of calling definedOutside).
static bool canBeHoisted(Operation *op,
                         function_ref<bool(OpOperand &)> condition) {
  // Do not move terminators.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;

  // Walk the nested operations and check that all used values are either
  // defined outside of the loop or in a nested region, but not at the level of
  // the loop body.
  auto walkFn = [&](Operation *child) {
    for (OpOperand &operand : child->getOpOperands()) {
      // Ignore values defined in a nested region.
      if (op->isAncestor(operand.get().getParentRegion()->getParentOp()))
        continue;
      if (!condition(operand))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  return !op->walk(walkFn).wasInterrupted();
}

static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  return canBeHoisted(
      op, [&](OpOperand &operand) { return definedOutside(operand.get()); });
}

/// Merges srcEffect's Memory Effect on its resource into the
/// resourceConflicts map, flagging the resource if the srcEffect
/// results in a conflict.
///
/// \param resourceConflicts The map to store resources' conflicts status.
/// \param srcEffect The effect to merge into the resourceConflicts map.
/// \param srcHasConflict Whether the srcEffect results in a conflict based
/// on higher level analysis.
///
/// resourceConflicts is modified by the function and will be non-empty
static void mergeResource(MemoryConflictMap &resourceConflicts,
                          const MemoryEffects::EffectInstance &srcEffect,
                          bool srcHasConflict) {

  TypeID srcResourceID = srcEffect.getResource()->getResourceID();

  bool srcIsAllocOrFree = isa<MemoryEffects::Allocate>(srcEffect.getEffect()) ||
                          isa<MemoryEffects::Free>(srcEffect.getEffect());

  LDBG() << "<Merging Effect> : \"" << srcEffect.getEffect()->getEffectName()
         << " on resource <" << srcEffect.getResource()->getName() << ">\""
         << "\n";

  bool conflict = srcHasConflict || srcIsAllocOrFree;

  auto [dstIt, inserted] = resourceConflicts.insert(
        std::make_pair(srcResourceID, std::make_pair(conflict, srcEffect)));
  if (inserted) {
    LDBG() << ". . . . "
           << "Effect inserted to map"
           << "\n";
    return;
  }

  // resource already in use
  bool dstHasConflict = dstIt->second.first;
  auto dstEffect = dstIt->second.second;

  if (dstHasConflict) {
    LDBG() << ". . . . "
           << "Resource has existing conflict from Effect Mem"
           << dstEffect.getValue() << "\n";
    return;
  }

  bool srcWrite = isa<MemoryEffects::Write>(srcEffect.getEffect());
  bool dstRead = isa<MemoryEffects::Read>(dstEffect.getEffect());
  bool readBeforeWrite = dstRead && srcWrite;

  conflict = conflict || readBeforeWrite;

  LDBG() << ". . . . "
         << "Resource conflict status updated to = " << conflict << "\n";

  dstIt->second = std::make_pair(conflict, srcEffect);
}

/// Returns true if any of op's operands is defined inside the loop.
static bool hasLoopVariantInput(LoopLikeOpInterface loopLike, Operation *op) {
  return llvm::any_of(op->getOperands(), [&] (Value v) {
    return !loopLike.isDefinedOutsideOfLoop(v);
  });
}

/// Returns true if:
/// (a) any of the resources used by op's Memory Effects have been
/// flagged as having a conflict within the resourceConflicts map OR
/// (b) op doesn't have a MemoryEffectOpInterface or has one but
/// without any specific effects.
static bool mayHaveMemoryEffectConflict(Operation *op,
                                        MemoryConflictMap *resourceConflicts) {
  LDBG() << "<Checking for memory effect conflict on op> : "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());

  auto condSpecInterface = dyn_cast<ConditionallySpeculatable>(op);

  // If op implements the ConditionallySpeculatable interface, it must be
  // speculatable for us to continue evaluating if it has Memory Effect
  // Conflicts.
  if (condSpecInterface && !isSpeculatable(op))
    return true;

  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);

  // op does not implement the memory effect op interface
  // shouldn't be flagged as movable to be conservative
  if (!memInterface)
    return true;

  // Ops with Recursive Memory Effects are special-cased here.
  // For now we'll only allow them to be moved if they're effect
  // free.
  // A potential solution is to recursively gather all resources on all
  // contained ops and then run the for-loop further below. Requires discussions
  // re: obscure corner cases.
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return !isMemoryEffectFree(op);

  // gather all effects on op
  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  memInterface.getEffects(effects);

  // Op has interface but no effects --> no conflicts.
  if (effects.empty())
    return false;

  // op has no conflicts IFF all resources are flagged as having no conflicts
  for (const MemoryEffects::EffectInstance &effect : effects) {
    auto resourceID = effect.getResource()->getResourceID();

    auto resConIter = resourceConflicts->find(resourceID);
    assert(resConIter != resourceConflicts->end());

    bool hasConflict = resConIter->second.first;
    if (hasConflict) {
      LDBG() << ". . . . "
             << "Conflict deteceted on resource <"
             << effect.getResource()->getName() << "> from Memory Effect Mem"
             << effect.getValue() << "\n";
      return true;
    }
  }

  return false;
}

static void
mapLoopResourceUsage(LoopLikeOpInterface loopLike, Operation *op,
                     MemoryConflictMap &resourceConflicts,
                     llvm::SmallSet<Operation *, 8> &opsWithReadBeforeWrite) {

  LDBG() << "<Mapping resource usage on op> : "
         << OpWithFlags(op, OpPrintingFlags().skipRegions()) << "\n";

  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    LDBG() << ". . . . "
           << "op has MemoryEffectsOpInterface"
           << "\n";

    // gather all effects on op
    SmallVector<MemoryEffects::EffectInstance> effects =
        MemoryEffects::getMemoryEffectsSorted(op);

    LDBG() << ". . . . "
           << "# of effects = " << effects.size();

    if (!effects.empty()) {
      // Any input to the op could be the input data source
      // for write effects in the same op. E.g.,
      // scf.for ... {
      //    %0 = foo.bar(...) : ...
      //    foo.baz(%0) // foo.baz has a MemWrite<SomeResource, 0> effect
      // }
      // The input %0 could be the data source for the write effect in
      // foo.baz. Since %0 is loop-variant, this may cause a conflict on 
      // SomeResource as the MemWrite contents may change between loop iterations. 
      // A more complex analysis would be needed to determine
      // if this is a true conflict or not.
      bool writesConflict = hasLoopVariantInput(loopLike, op);
      LDBG() << ". . . . "
             << "Has loop-variant input = "
             << (writesConflict ? "true" : "false") << "\n";

      bool hasRead = false;

      for (const MemoryEffects::EffectInstance &effect : effects) {
        bool inConflict = false;

        if (isa<MemoryEffects::Write>(effect.getEffect())) {
          if (hasRead) {
            LDBG() << ". . . . "
                   << "read-before-write detected!"
                   << "\n";
            LDBG() << ". . . . "
                   << ". . . . "
                   << "Inserting op into set for later checks!"
                   << "\n";
            opsWithReadBeforeWrite.insert(op);
          }

          inConflict = writesConflict;
        }

        mergeResource(resourceConflicts, effect, inConflict);

        // All writes to a resource that follow a read on any other resource
        // need additional logic to check if the read will result in a conflict
        // on the following write op(s)'s resource(s).
        // Need to keep track of ops that have read before writes.
        // If the resource for the read effect has a conflict after all loop
        // resource usages have been mapped, then the conflict will be
        // propagated to the resources used by the following writes. LOGIC: if
        // the read resource is in conflict, that means the value stored is no
        // longer loop invariant --> the read could be the data source for the
        // write --> the write is not guaranteed to be loop invariant.
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
          TypeID resourceID = effect.getResource()->getResourceID();
          auto resConIter = resourceConflicts.find(resourceID);

          if (resConIter != resourceConflicts.end()) {
            hasRead = true;
          }
        }
      }
    }
  }

  for (Region &region : op->getRegions())
    for (Operation &opInner : region.getOps())
      mapLoopResourceUsage(loopLike, &opInner, resourceConflicts,
                           opsWithReadBeforeWrite);
}

static void propagateSameOpReadBeforeWriteConflicts(
    LoopLikeOpInterface loopLike, Operation *op,
    MemoryConflictMap &resourceConflicts,
    llvm::SmallSet<Operation *, 8> &opsWithReadBeforeWrite) {

  for (auto *opInner : opsWithReadBeforeWrite) {
    // gather all effects on op
    SmallVector<MemoryEffects::EffectInstance> effects =
        MemoryEffects::getMemoryEffectsSorted(opInner);

    bool writesConflict = false;

    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (writesConflict && isa<MemoryEffects::Write>(effect.getEffect())) {

        TypeID resourceID = effect.getResource()->getResourceID();
        auto resConIter = resourceConflicts.find(resourceID);

        // already has conflict, move on
        if (resConIter->getSecond().first)
          continue;

        resConIter->getSecond().first = true;
        resConIter->getSecond().second = effect;
      }

      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        TypeID resourceID = effect.getResource()->getResourceID();
        auto resConIter = resourceConflicts.find(resourceID);

        if (resConIter != resourceConflicts.end() &&
            resConIter->getSecond().first) {
          writesConflict = true;
        }
      }
    }
  }
}

void mlir::mapResourceConflicts(LoopLikeOpInterface loopLike, Operation *op,
                                MemoryConflictMap &resourceConflicts) {
  llvm::SmallSet<Operation *, 8> opsWithReadBeforeWrite;
  mapLoopResourceUsage(loopLike, loopLike.getOperation(), resourceConflicts,
                       opsWithReadBeforeWrite);
  propagateSameOpReadBeforeWriteConflicts(loopLike, loopLike.getOperation(),
                                          resourceConflicts,
                                          opsWithReadBeforeWrite);
}

size_t mlir::moveLoopInvariantCode(
    LoopLikeOpInterface loopLike,
    function_ref<bool(Value, Region *)> isDefinedOutsideRegion,
    function_ref<bool(Operation *, Region *)> shouldMoveSpeculatable,
    function_ref<bool(Operation *, MemoryConflictMap *)> shouldMoveMemoryEffect,
    function_ref<void(Operation *, Region *)> moveOutOfRegion) {
  size_t numMovedTotal = 0;

  // Check that the loop isn't dead.
  // Two separate methods used to check this, depending on what the loopLike op
  // implements. If neither is available, we can't guarantee loop liveness.
  auto isMaybeDead = loopLike.isZeroTrip();
  auto tripCount = loopLike.getStaticTripCount();

  bool confirmedDead = (isMaybeDead.has_value() && isMaybeDead.value()) ||
                       (tripCount.has_value() && tripCount.value() == 0);
  bool ambiguousLiveness = !isMaybeDead.has_value() && !tripCount.has_value();
  bool loopIsLive = !confirmedDead && !ambiguousLiveness;

  LDBG() << "Running LICM on loop op. . . ."
         << "\n";
  LDBG() << "<LoopLikeOp> : "
         << OpWithFlags(loopLike.getOperation(),
                        OpPrintingFlags().skipRegions())
         << "\n";
  LDBG() << ". . . . "
         << "confirmedDead = " << confirmedDead << "\n";
  LDBG() << ". . . . "
         << "ambiguousLiveness = " << ambiguousLiveness << "\n";
  LDBG() << ". . . . "
         << "loopIsLive = " << loopIsLive << "\n";

  int iteration = 0;
  int numMoved = 0;

  do {
    // reset value for iteration
    numMoved = 0;

    MemoryConflictMap resourceConflicts;

    // For loops that are guaranteed to execute at least one iterations:
    // Go through loop body and map out resource usages.
    // op->regions are essentially merged sequentially.
    // E.g., an if's "then" and "else" regions are treated like one
    // continuous region --> need to add fork checking.
    //
    // loop "do" and "then" regions also merged.
    if (loopIsLive)
      mapResourceConflicts(loopLike, loopLike.getOperation(),
                           resourceConflicts);

    auto regions = loopLike.getLoopRegions();
    for (Region *region : regions) {
      std::queue<Operation *> worklist;

      // Add top-level operations in the loop body to the worklist.
      for (Operation &op : region->getOps())
        worklist.push(&op);

      auto definedOutside = [&](Value value) {
        return isDefinedOutsideRegion(value, region);
      };

      while (!worklist.empty()) {
        Operation *op = worklist.front();
        worklist.pop();

        // Skip ops that have already been moved. Check if the op can be hoisted.
        if (op->getParentRegion() != region)
          continue;

        LDBG() << ". . . . "
               << "<Checking Op> : "
               << OpWithFlags(op, OpPrintingFlags().skipRegions());

        bool isHoistable = canBeHoisted(op, definedOutside);
        LDBG() << ". . . . "
               << ". . . . "
               << "isHoistable = " << isHoistable << "\n";

        if (!isHoistable)
          continue;

        // Check Speculability Path first since it's cheapeer.
        bool movableUnderSpeculabilityPath = shouldMoveSpeculatable(op, region);
        LDBG() << ". . . . "
               << ". . . . "
               << "movableUnderSpeculabilityPath = "
               << movableUnderSpeculabilityPath << "\n";

        // Check other path if first one fails.
        if (!movableUnderSpeculabilityPath) {
          bool movableUnderMemoryEffectsPath =
              loopIsLive && shouldMoveMemoryEffect(op, &resourceConflicts);
          LDBG() << ". . . . "
                 << ". . . . "
                 << "movableUnderMemoryEffectsPath = "
                 << movableUnderMemoryEffectsPath << "\n";

          // Cannot be moved.
          if (!movableUnderMemoryEffectsPath)
            continue;
        }

        moveOutOfRegion(op, region);
        ++numMoved;

        // Since the op has been moved, we need to check its users within the
        // top-level of the loop body.
        for (Operation *user : op->getUsers())
          if (user->getParentRegion() == region)
            worklist.push(user);
      }
    }

    numMovedTotal += numMoved;

    LDBG() << ". . . . "
           << "Finishing LICM iteration " << iteration++ << "\n";
    LDBG() << ". . . . "
           << ". . . . "
           << "Number of ops moved = " << numMoved << "\n";
    LDBG() << ". . . . "
           << ". . . . "
           << "Total number of ops moved across iterations = " << numMovedTotal
           << "\n";

  } while (numMoved > 0);

  return numMovedTotal;
}

size_t mlir::moveLoopInvariantCode(LoopLikeOpInterface loopLike) {
  return moveLoopInvariantCode(
      loopLike,
      [&](Value value, Region *) {
        return loopLike.isDefinedOutsideOfLoop(value);
      },
      [&](Operation *op, Region *) {
        return isMemoryEffectFree(op) && isSpeculatable(op);
      },
      [&](Operation *op, MemoryConflictMap *resourceConflicts) {
        return !mayHaveMemoryEffectConflict(op, resourceConflicts);
      },
      [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
}

namespace {
/// Helper data structure that keeps track of equivalent/disjoint subset ops.
class MatchingSubsets {
public:
  /// Insert a subset op.
  void insert(SubsetOpInterface op, bool collectHoistableOps = true) {
    allSubsetOps.push_back(op);
    if (!collectHoistableOps)
      return;
    if (auto extractionOp =
            dyn_cast<SubsetExtractionOpInterface>(op.getOperation()))
      insertExtractionOp(extractionOp);
    if (auto insertionOp =
            dyn_cast<SubsetInsertionOpInterface>(op.getOperation()))
      insertInsertionOp(insertionOp);
  }

  /// Return a range of matching extraction-insertion subset ops. If there is no
  /// matching extraction/insertion op, the respective value is empty. Ops are
  /// skipped if there are other subset ops that are not guaranteed to operate
  /// on disjoint subsets.
  auto getHoistableSubsetOps() {
    return llvm::make_filter_range(
        llvm::zip(extractions, insertions), [&](auto pair) {
          auto [extractionOp, insertionOp] = pair;
          // Hoist only if the extracted and inserted values have the same type.
          if (extractionOp && insertionOp &&
              extractionOp->getResult(0).getType() !=
                  insertionOp.getSourceOperand().get().getType())
            return false;
          // Hoist only if there are no conflicting subset ops.
          return allDisjoint(extractionOp, insertionOp);
        });
  }

  /// Populate subset ops starting from the given region iter_arg. Return
  /// "failure" if non-subset ops are found along the path to the loop yielding
  /// op or if there is no single path to the tied yielded operand. If
  /// `collectHoistableOps` is set to "false", subset ops are gathered
  /// throughout the traversal, but not enumerated by `getHoistableSubsetOps`.
  LogicalResult populateSubsetOpsAtIterArg(LoopLikeOpInterface loopLike,
                                           BlockArgument iterArg,
                                           bool collectHoistableOps = true);

private:
  /// Helper function for equivalence of tensor values. Since only insertion
  /// subset ops (that are also destination style ops) are followed when
  /// traversing the SSA use-def chain, all tensor values are equivalent.
  static bool isEquivalent(Value v1, Value v2) { return true; }

  /// Return "true" if the subsets of the given extraction and insertion ops
  /// are operating disjoint from the subsets that all other known subset ops
  /// are operating on.
  bool allDisjoint(SubsetExtractionOpInterface extractionOp,
                   SubsetInsertionOpInterface insertionOp) const {
    for (SubsetOpInterface other : allSubsetOps) {
      if (other == extractionOp || other == insertionOp)
        continue;
      if (extractionOp &&
          !other.operatesOnDisjointSubset(extractionOp, isEquivalent))
        return false;
      if (insertionOp &&
          !other.operatesOnDisjointSubset(insertionOp, isEquivalent))
        return false;
    }
    return true;
  }

  /// Insert a subset extraction op. If the subset is equivalent to an existing
  /// subset insertion op, pair them up. (If there is already a paired up subset
  /// extraction op, overwrite the subset extraction op.)
  void insertExtractionOp(SubsetExtractionOpInterface extractionOp) {
    for (auto it : llvm::enumerate(insertions)) {
      if (!it.value())
        continue;
      auto other = cast<SubsetOpInterface>(it.value().getOperation());
      if (other.operatesOnEquivalentSubset(extractionOp, isEquivalent)) {
        extractions[it.index()] = extractionOp;
        return;
      }
    }
    // There is no known equivalent insertion op. Create a new entry.
    extractions.push_back(extractionOp);
    insertions.push_back({});
  }

  /// Insert a subset insertion op. If the subset is equivalent to an existing
  /// subset extraction op, pair them up. (If there is already a paired up
  /// subset insertion op, overwrite the subset insertion op.)
  void insertInsertionOp(SubsetInsertionOpInterface insertionOp) {
    for (auto it : llvm::enumerate(extractions)) {
      if (!it.value())
        continue;
      auto other = cast<SubsetOpInterface>(it.value().getOperation());
      if (other.operatesOnEquivalentSubset(insertionOp, isEquivalent)) {
        insertions[it.index()] = insertionOp;
        return;
      }
    }
    // There is no known equivalent extraction op. Create a new entry.
    extractions.push_back({});
    insertions.push_back(insertionOp);
  }

  SmallVector<SubsetExtractionOpInterface> extractions;
  SmallVector<SubsetInsertionOpInterface> insertions;
  SmallVector<SubsetOpInterface> allSubsetOps;
};
} // namespace

/// If the given value has a single use by an op that is a terminator, return
/// that use. Otherwise, return nullptr.
static OpOperand *getSingleTerminatorUse(Value value) {
  if (!value.hasOneUse())
    return nullptr;
  OpOperand &use = *value.getUses().begin();
  if (use.getOwner()->hasTrait<OpTrait::IsTerminator>())
    return &use;
  return nullptr;
}

LogicalResult
MatchingSubsets::populateSubsetOpsAtIterArg(LoopLikeOpInterface loopLike,
                                            BlockArgument iterArg,
                                            bool collectHoistableOps) {
  assert(iterArg.getOwner()->getParentOp() == loopLike && "invalid iter_arg");
  Value value = iterArg;

  // Traverse use-def chain. Subset ops can be hoisted only if all ops along the
  // use-def chain starting from the region iter_arg are subset extraction or
  // subset insertion ops. The chain must terminate at the corresponding yield
  // operand (e.g., no swapping of iter_args).
  OpOperand *yieldedOperand = nullptr;
  // Iterate until the single use of the current SSA value is a terminator,
  // which is expected to be the yielding operation of the loop.
  while (!(yieldedOperand = getSingleTerminatorUse(value))) {
    Value nextValue = {};

    for (OpOperand &use : value.getUses()) {
      if (auto nestedLoop = dyn_cast<LoopLikeOpInterface>(use.getOwner())) {
        // Subset ops in nested loops are collected to check if there are only
        // disjoint subset ops, but such subset ops are not subject to hoisting.
        // To hoist subset ops from nested loops, the hoisting transformation
        // should be run on the nested loop.
        auto nestedIterArg = nestedLoop.getTiedLoopRegionIterArg(&use);
        if (!nestedIterArg)
          return failure();
        // Note: `populateSubsetOpsAtIterArg` fails if there is no single SSA
        // use-def chain starting at `nestedIterArg` and terminating in the
        // tied, yielding operand.
        if (failed(populateSubsetOpsAtIterArg(nestedLoop, nestedIterArg,
                                              /*collectHoistableOps=*/false)))
          return failure();
        nextValue = nestedLoop.getTiedLoopResult(&use);
        continue;
      }

      auto subsetOp = dyn_cast<SubsetOpInterface>(use.getOwner());
      if (!subsetOp)
        return failure();
      insert(subsetOp);

      if (auto insertionOp =
              dyn_cast<SubsetInsertionOpInterface>(use.getOwner())) {
        // Current implementation expects that the insertionOp implement
        // the DestinationStyleOpInterface and with pure tensor semantics
        // as well. Abort if that is not the case.
        auto dstOp = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
        if (!dstOp || !dstOp.hasPureTensorSemantics())
          return failure();

        // The value must be used as a destination. (In case of a source, the
        // entire tensor would be read, which would prevent any hoisting.)
        if (&use != &insertionOp.getDestinationOperand())
          return failure();
        // There must be a single use-def chain from the region iter_arg to the
        // terminator. I.e., only one insertion op. Branches are not supported.
        if (nextValue)
          return failure();
        nextValue = insertionOp.getUpdatedDestination();
      }
    }

    // Nothing can be hoisted if the chain does not continue with loop yielding
    // op or a subset insertion op.
    if (!nextValue)
      return failure();
    value = nextValue;
  }

  // Hoist only if the SSA use-def chain ends in the yielding terminator of the
  // loop and the yielded value is the `idx`-th operand. (I.e., there is no
  // swapping yield.)
  if (loopLike.getTiedLoopYieldedValue(iterArg) != yieldedOperand)
    return failure();

  return success();
}

/// Hoist all subset ops that operate on the idx-th region iter_arg of the given
/// loop-like op and index into loop-invariant subset locations. Return the
/// newly created loop op (that has extra iter_args) or the original loop op if
/// nothing was hoisted.
static LoopLikeOpInterface hoistSubsetAtIterArg(RewriterBase &rewriter,
                                                LoopLikeOpInterface loopLike,
                                                BlockArgument iterArg) {
  assert(iterArg.getOwner()->getParentOp() == loopLike && "invalid iter_arg");
  BlockArgument *it = llvm::find(loopLike.getRegionIterArgs(), iterArg);
  int64_t iterArgIdx = std::distance(loopLike.getRegionIterArgs().begin(), it);
  MatchingSubsets subsets;
  if (failed(subsets.populateSubsetOpsAtIterArg(loopLike, iterArg)))
    return loopLike;

  // Hoist all matching extraction-insertion pairs one-by-one.
  for (auto it : subsets.getHoistableSubsetOps()) {
    auto extractionOp = std::get<0>(it);
    auto insertionOp = std::get<1>(it);

    // Ops cannot be hoisted if they depend on loop-variant values.
    if (extractionOp) {
      if (!canBeHoisted(extractionOp, [&](OpOperand &operand) {
            return loopLike.isDefinedOutsideOfLoop(operand.get()) ||
                   &operand == &extractionOp.getSourceOperand();
          }))
        extractionOp = {};
    }
    if (insertionOp) {
      if (!canBeHoisted(insertionOp, [&](OpOperand &operand) {
            return loopLike.isDefinedOutsideOfLoop(operand.get()) ||
                   &operand == &insertionOp.getSourceOperand() ||
                   &operand == &insertionOp.getDestinationOperand();
          }))
        insertionOp = {};
    }

    // Only hoist extraction-insertion pairs for now. Standalone extractions/
    // insertions that are loop-invariant could be hoisted, but there may be
    // easier ways to canonicalize the IR.
    if (extractionOp && insertionOp) {
      // Create a new loop with an additional iter_arg.
      NewYieldValuesFn newYieldValuesFn =
          [&](OpBuilder &b, Location loc,
              ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
        return {insertionOp.getSourceOperand().get()};
      };
      FailureOr<LoopLikeOpInterface> newLoop =
          loopLike.replaceWithAdditionalYields(
              rewriter, extractionOp.getResult(),
              /*replaceInitOperandUsesInLoop=*/true, newYieldValuesFn);
      if (failed(newLoop))
        return loopLike;
      loopLike = *newLoop;

      // Hoist the extraction/insertion ops.
      iterArg = loopLike.getRegionIterArgs()[iterArgIdx];
      OpResult loopResult = loopLike.getTiedLoopResult(iterArg);
      OpResult newLoopResult = loopLike.getLoopResults()->back();
      rewriter.moveOpBefore(extractionOp, loopLike);
      rewriter.moveOpAfter(insertionOp, loopLike);
      rewriter.replaceAllUsesWith(insertionOp.getUpdatedDestination(),
                                  insertionOp.getDestinationOperand().get());
      extractionOp.getSourceOperand().set(
          loopLike.getTiedLoopInit(iterArg)->get());
      rewriter.replaceAllUsesWith(loopResult,
                                  insertionOp.getUpdatedDestination());
      insertionOp.getSourceOperand().set(newLoopResult);
      insertionOp.getDestinationOperand().set(loopResult);
    }
  }

  return loopLike;
}

LoopLikeOpInterface
mlir::hoistLoopInvariantSubsets(RewriterBase &rewriter,
                                LoopLikeOpInterface loopLike) {
  // Note: As subset ops are getting hoisted, the number of region iter_args
  // increases. This can enable further hoisting opportunities on the new
  // iter_args.
  for (int64_t i = 0;
       i < static_cast<int64_t>(loopLike.getRegionIterArgs().size()); ++i) {
    loopLike = hoistSubsetAtIterArg(rewriter, loopLike,
                                    loopLike.getRegionIterArgs()[i]);
  }
  return loopLike;
}
