//===- BufferOptimizations.cpp - pre-pass optimizations for bufferization -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for two optimization passes. These passes try to
// hoist alloc nodes to reduce the number of allocations and copies during
// buffer deallocation.

#include "PassDetail.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// BufferAllocationHoisting
//===----------------------------------------------------------------------===//

/// A base implementation compatible with the `BufferAllocationHoisting` class.
struct BufferAllocationHoistingStateBase {
  /// A pointer to the current dominance info.
  DominanceInfo *dominators;

  /// The current allocation value.
  Value allocValue;

  /// The current placement block (if any).
  Block *placementBlock;

  /// Initializes the state base.
  BufferAllocationHoistingStateBase(DominanceInfo *dominators, Value allocValue,
                                    Block *placementBlock)
      : dominators(dominators), allocValue(allocValue),
        placementBlock(placementBlock) {}
};

/// Implements the actual hoisting logic for allocation nodes.
template <typename StateT>
class BufferAllocationHoisting : public BufferPlacementTransformationBase {
private:
  /// Returns true if the given operation implements a known high-level region-
  /// based control-flow interface.
  static bool isKnownControlFlowInterface(Operation *op) {
    return isa<LoopLikeOpInterface, RegionBranchOpInterface>(op);
  }

public:
  BufferAllocationHoisting(Operation *op)
      : BufferPlacementTransformationBase(op), dominators(op),
        postDominators(op) {}

  /// Moves allocations upwards.
  void hoist() {
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);
      Operation *definingOp = allocValue.getDefiningOp();
      assert(definingOp && "No defining op");
      auto operands = definingOp->getOperands();
      auto resultAliases = aliases.resolve(allocValue);
      // Determine the common dominator block of all aliases.
      Block *dominatorBlock =
          findCommonDominator(allocValue, resultAliases, dominators);
      // Init the initial hoisting state.
      StateT state(&dominators, allocValue, allocValue.getParentBlock());
      // Check for additional allocation dependencies to compute an upper bound
      // for hoisting.
      Block *dependencyBlock = nullptr;
      if (!operands.empty()) {
        // If this node has dependencies, check all dependent nodes with respect
        // to a common post dominator. This ensures that all dependency values
        // have been computed before allocating the buffer.
        ValueSetT dependencies(std::next(operands.begin()), operands.end());
        dependencyBlock = findCommonDominator(*operands.begin(), dependencies,
                                              postDominators);
      }

      // Find the actual placement block and determine the start operation using
      // an upper placement-block boundary. The idea is that placement block
      // cannot be moved any further upwards than the given upper bound.
      Block *placementBlock = findPlacementBlock(
          state, state.computeUpperBound(dominatorBlock, dependencyBlock));
      Operation *startOperation = BufferPlacementAllocs::getStartOperation(
          allocValue, placementBlock, liveness);

      // Move the alloc in front of the start operation.
      Operation *allocOperation = allocValue.getDefiningOp();
      allocOperation->moveBefore(startOperation);
    }
  }

private:
  /// Finds a valid placement block by walking upwards in the CFG until we
  /// either cannot continue our walk due to constraints (given by the StateT
  /// implementation) or we have reached the upper-most dominator block.
  Block *findPlacementBlock(StateT &state, Block *upperBound) {
    Block *currentBlock = state.placementBlock;
    // Walk from the innermost regions/loops to the outermost regions/loops and
    // find an appropriate placement block that satisfies the constraint of the
    // current StateT implementation. Walk until we reach the upperBound block
    // (if any).

    // If we are not able to find a valid parent operation or an associated
    // parent block, break the walk loop.
    Operation *parentOp;
    Block *parentBlock;
    while ((parentOp = currentBlock->getParentOp()) &&
           (parentBlock = parentOp->getBlock()) &&
           (!upperBound ||
            dominators.properlyDominates(upperBound, currentBlock))) {
      // Try to find an immediate dominator and check whether the parent block
      // is above the immediate dominator (if any).
      DominanceInfoNode *idom = dominators.getNode(currentBlock)->getIDom();
      if (idom && dominators.properlyDominates(parentBlock, idom->getBlock())) {
        // If the current immediate dominator is below the placement block, move
        // to the immediate dominator block.
        currentBlock = idom->getBlock();
        state.recordMoveToDominator(currentBlock);
      } else {
        // We have to move to our parent block since an immediate dominator does
        // either not exist or is above our parent block. If we cannot move to
        // our parent operation due to constraints given by the StateT
        // implementation, break the walk loop. Furthermore, we should not move
        // allocations out of unknown region-based control-flow operations.
        if (!isKnownControlFlowInterface(parentOp) ||
            !state.isLegalPlacement(parentOp))
          break;
        // Move to our parent block by notifying the current StateT
        // implementation.
        currentBlock = parentBlock;
        state.recordMoveToParent(currentBlock);
      }
    }
    // Return the finally determined placement block.
    return state.placementBlock;
  }

  /// The dominator info to find the appropriate start operation to move the
  /// allocs.
  DominanceInfo dominators;

  /// The post dominator info to move the dependent allocs in the right
  /// position.
  PostDominanceInfo postDominators;

  /// The map storing the final placement blocks of a given alloc value.
  llvm::DenseMap<Value, Block *> placementBlocks;
};

/// A state implementation compatible with the `BufferAllocationHoisting` class
/// that hoists allocations into dominator blocks while keeping them inside of
/// loops.
struct BufferAllocationHoistingState : BufferAllocationHoistingStateBase {
  using BufferAllocationHoistingStateBase::BufferAllocationHoistingStateBase;

  /// Computes the upper bound for the placement block search.
  Block *computeUpperBound(Block *dominatorBlock, Block *dependencyBlock) {
    // If we do not have a dependency block, the upper bound is given by the
    // dominator block.
    if (!dependencyBlock)
      return dominatorBlock;

    // Find the "lower" block of the dominator and the dependency block to
    // ensure that we do not move allocations above this block.
    return dominators->properlyDominates(dominatorBlock, dependencyBlock)
               ? dependencyBlock
               : dominatorBlock;
  }

  /// Returns true if the given operation does not represent a loop.
  bool isLegalPlacement(Operation *op) {
    return !BufferPlacementTransformationBase::isLoop(op);
  }

  /// Sets the current placement block to the given block.
  void recordMoveToDominator(Block *block) { placementBlock = block; }

  /// Sets the current placement block to the given block.
  void recordMoveToParent(Block *block) { recordMoveToDominator(block); }
};

/// A state implementation compatible with the `BufferAllocationHoisting` class
/// that hoists allocations out of loops.
struct BufferAllocationLoopHoistingState : BufferAllocationHoistingStateBase {
  using BufferAllocationHoistingStateBase::BufferAllocationHoistingStateBase;

  /// Remembers the dominator block of all aliases.
  Block *aliasDominatorBlock;

  /// Computes the upper bound for the placement block search.
  Block *computeUpperBound(Block *dominatorBlock, Block *dependencyBlock) {
    aliasDominatorBlock = dominatorBlock;
    // If there is a dependency block, we have to use this block as an upper
    // bound to satisfy all allocation value dependencies.
    return dependencyBlock ? dependencyBlock : nullptr;
  }

  /// Returns true if the given operation represents a loop and one of the
  /// aliases caused the `aliasDominatorBlock` to be "above" the block of the
  /// given loop operation. If this is the case, it indicates that the
  /// allocation is passed via a back edge.
  bool isLegalPlacement(Operation *op) {
    return BufferPlacementTransformationBase::isLoop(op) &&
           !dominators->dominates(aliasDominatorBlock, op->getBlock());
  }

  /// Does not change the internal placement block, as we want to move
  /// operations out of loops only.
  void recordMoveToDominator(Block *block) {}

  /// Sets the current placement block to the given block.
  void recordMoveToParent(Block *block) { placementBlock = block; }
};

//===----------------------------------------------------------------------===//
// BufferOptimizationPasses
//===----------------------------------------------------------------------===//

/// The buffer hoisting pass that hoists allocation nodes into dominating
/// blocks.
struct BufferHoistingPass : BufferHoistingBase<BufferHoistingPass> {

  void runOnFunction() override {
    // Hoist all allocations into dominator blocks.
    BufferAllocationHoisting<BufferAllocationHoistingState> optimizer(
        getFunction());
    optimizer.hoist();
  }
};

/// The buffer loop hoisting pass that hoists allocation nodes out of loops.
struct BufferLoopHoistingPass : BufferLoopHoistingBase<BufferLoopHoistingPass> {

  void runOnFunction() override {
    // Hoist all allocations out of loops.
    BufferAllocationHoisting<BufferAllocationLoopHoistingState> optimizer(
        getFunction());
    optimizer.hoist();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::createBufferHoistingPass() {
  return std::make_unique<BufferHoistingPass>();
}

std::unique_ptr<Pass> mlir::createBufferLoopHoistingPass() {
  return std::make_unique<BufferLoopHoistingPass>();
}
