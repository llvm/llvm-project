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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "licm"

using namespace mlir;

/// Checks whether the given op can be hoisted by checking that
/// - the op and none of its contained operations depend on values inside of the
///   loop (by means of calling definedOutside).
/// - the op has no side-effects.
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

size_t mlir::moveLoopInvariantCode(
    ArrayRef<Region *> regions,
    function_ref<bool(Value, Region *)> isDefinedOutsideRegion,
    function_ref<bool(Operation *, Region *)> shouldMoveOutOfRegion,
    function_ref<void(Operation *, Region *)> moveOutOfRegion) {
  size_t numMoved = 0;

  for (Region *region : regions) {
    LLVM_DEBUG(llvm::dbgs() << "Original loop:\n"
                            << *region->getParentOp() << "\n");

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

      LLVM_DEBUG(llvm::dbgs() << "Checking op: " << *op << "\n");
      if (!shouldMoveOutOfRegion(op, region) ||
          !canBeHoisted(op, definedOutside))
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Moving loop-invariant op: " << *op << "\n");
      moveOutOfRegion(op, region);
      ++numMoved;

      // Since the op has been moved, we need to check its users within the
      // top-level of the loop body.
      for (Operation *user : op->getUsers())
        if (user->getParentRegion() == region)
          worklist.push(user);
    }
  }

  return numMoved;
}

size_t mlir::moveLoopInvariantCode(LoopLikeOpInterface loopLike) {
  return moveLoopInvariantCode(
      loopLike.getLoopRegions(),
      [&](Value value, Region *) {
        return loopLike.isDefinedOutsideOfLoop(value);
      },
      [&](Operation *op, Region *) {
        return isMemoryEffectFree(op) && isSpeculatable(op);
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
        // the destinationStyleOpInterface as well. Abort if that tha is not
        // the case
        if (!isa<DestinationStyleOpInterface>(use.getOwner())) {
          return failure();
        }

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
  auto it = llvm::find(loopLike.getRegionIterArgs(), iterArg);
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
