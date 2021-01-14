//===- BufferOptimizations.cpp - pre-pass optimizations for bufferization -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for three optimization passes. The first two
// passes try to move alloc nodes out of blocks to reduce the number of
// allocations and copies during buffer deallocation. The third pass tries to
// convert heap-based allocations to stack-based allocations, if possible.

#include "PassDetail.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

/// Returns true if the given operation implements a known high-level region-
/// based control-flow interface.
static bool isKnownControlFlowInterface(Operation *op) {
  return isa<LoopLikeOpInterface, RegionBranchOpInterface>(op);
}

/// Check if the size of the allocation is less than the given size. The
/// transformation is only applied to small buffers since large buffers could
/// exceed the stack space.
static bool isSmallAlloc(Value alloc, unsigned maximumSizeInBytes,
                         unsigned bitwidthOfIndexType) {
  auto type = alloc.getType().dyn_cast<ShapedType>();
  if (!type || !type.hasStaticShape())
    return false;
  // For index types, use the provided size, as the type does not know.
  unsigned int bitwidth = type.getElementType().isIndex()
                              ? bitwidthOfIndexType
                              : type.getElementTypeBitWidth();
  return type.getNumElements() * bitwidth <= maximumSizeInBytes * 8;
}

/// Checks whether the given aliases leave the allocation scope.
static bool
leavesAllocationScope(Region *parentRegion,
                      const BufferAliasAnalysis::ValueSetT &aliases) {
  for (Value alias : aliases) {
    for (auto *use : alias.getUsers()) {
      // If there is at least one alias that leaves the parent region, we know
      // that this alias escapes the whole region and hence the associated
      // allocation leaves allocation scope.
      if (use->hasTrait<OpTrait::ReturnLike>() &&
          use->getParentRegion() == parentRegion)
        return true;
    }
  }
  return false;
}

/// Checks, if an automated allocation scope for a given alloc value exists.
static bool hasAllocationScope(Value alloc,
                               const BufferAliasAnalysis &aliasAnalysis) {
  Region *region = alloc.getParentRegion();
  do {
    if (Operation *parentOp = region->getParentOp()) {
      // Check if the operation is an automatic allocation scope and whether an
      // alias leaves the scope. This means, an allocation yields out of
      // this scope and can not be transformed in a stack-based allocation.
      if (parentOp->hasTrait<OpTrait::AutomaticAllocationScope>() &&
          !leavesAllocationScope(region, aliasAnalysis.resolve(alloc)))
        return true;
      // Check if the operation is a known control flow interface and break the
      // loop to avoid transformation in loops. Furthermore skip transformation
      // if the operation does not implement a RegionBeanchOpInterface.
      if (BufferPlacementTransformationBase::isLoop(parentOp) ||
          !isKnownControlFlowInterface(parentOp))
        break;
    }
  } while ((region = region->getParentRegion()));
  return false;
}

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
// BufferPlacementPromotion
//===----------------------------------------------------------------------===//

/// Promotes heap-based allocations to stack-based allocations (if possible).
class BufferPlacementPromotion : BufferPlacementTransformationBase {
public:
  BufferPlacementPromotion(Operation *op)
      : BufferPlacementTransformationBase(op) {}

  /// Promote buffers to stack-based allocations.
  void promote(unsigned maximumSize, unsigned bitwidthOfIndexType) {
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value alloc = std::get<0>(entry);
      Operation *dealloc = std::get<1>(entry);
      // Checking several requirements to transform an AllocOp into an AllocaOp.
      // The transformation is done if the allocation is limited to a given
      // size. Furthermore, a deallocation must not be defined for this
      // allocation entry and a parent allocation scope must exist.
      if (!isSmallAlloc(alloc, maximumSize, bitwidthOfIndexType) || dealloc ||
          !hasAllocationScope(alloc, aliases))
        continue;

      Operation *startOperation = BufferPlacementAllocs::getStartOperation(
          alloc, alloc.getParentBlock(), liveness);
      // Build a new alloca that is associated with its parent
      // `AutomaticAllocationScope` determined during the initialization phase.
      OpBuilder builder(startOperation);
      auto alloca = builder.create<AllocaOp>(
          alloc.getLoc(), alloc.getType().cast<MemRefType>());

      // Replace the original alloc by a newly created alloca.
      Operation *allocOp = alloc.getDefiningOp();
      allocOp->replaceAllUsesWith(alloca.getOperation());
      allocOp->erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// BufferReuse
//===----------------------------------------------------------------------===//

/// Reuses already allocated buffer to save allocation operations.
class BufferReuse : BufferPlacementTransformationBase {
public:
  BufferReuse(Operation *op)
      : BufferPlacementTransformationBase(op), dominators(op),
        postDominators(op) {}

  /// An implementation for the first and last use of a value.
  struct FirstAndLastUse {
    Operation *firstUse;
    Operation *lastUse;

    bool operator==(const FirstAndLastUse &other) const {
      return firstUse == other.firstUse && lastUse == other.lastUse;
    }

    bool operator!=(const FirstAndLastUse &other) const {
      return firstUse != other.firstUse || lastUse != other.lastUse;
    }
  };

  /// Reuses already allocated buffers to save allocation operations.
  void reuse() {
    // Find all first and last uses for all allocated values and their aliases
    // and save them in the useRangeMap.
    llvm::MapVector<Value, FirstAndLastUse> useRangeMap;
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);

      // Resolve all aliases for the allocValue to later save them in a cache.
      ValueSetT aliasSet = aliases.resolve(allocValue);

      // Iterate over the aliasSet and compute the use range.
      for (Value aliasValue : aliasSet) {
        Value::user_range users = aliasValue.getUsers();
        // Check if the allocValue/alias is already processed or has no users.
        if (useRangeMap.count(aliasValue) || users.empty())
          continue;

        FirstAndLastUse firstAndLastUse{};
        // Iterate over all uses of the allocValue/alias and find their first
        // and last use.
        for (Operation *user : users) {
          // No update is needed if the operation has already been considered.
          if (firstAndLastUse.firstUse == user ||
              firstAndLastUse.lastUse == user)
            continue;

          updateFirstOp(firstAndLastUse.firstUse, user, [&]() {
            return &findCommonDominator(aliasValue, ValueSetT{aliasValue},
                                        dominators)
                        ->back();
          });

          updateLastOp(firstAndLastUse.lastUse, user, [&]() {
            return &findCommonDominator(aliasValue, ValueSetT{aliasValue},
                                        postDominators)
                        ->front();
          });
        }
        useRangeMap.insert(
            std::pair<Value, FirstAndLastUse>(aliasValue, firstAndLastUse));
      }

      // Remove the allocValue from its own aliasList to prevent reflexive
      // checks and ensure correct behavior after we insert the aliases of the
      // reused buffer.
      aliasSet.erase(allocValue);
      aliasCache.insert(std::pair<Value, ValueSetT>(allocValue, aliasSet));
    }
  }

private:
  /// Updates the first Operation from the two given ones.
  template <typename DominatorFunc>
  void updateFirstOp(Operation *&op, Operation *user, DominatorFunc domFunc) {
    if (!op || isUsedBefore(user, op))
      op = user;
    else if (!isUsedBefore(op, user) && !isUsedBefore(user, op))
      op = domFunc();
  }

  /// Updates the last Operation from the two given ones.
  template <typename DominatorFunc>
  void updateLastOp(Operation *&op, Operation *user, DominatorFunc domFunc) {
    if (!op || isUsedBefore(op, user))
      op = user;
    else if (!isUsedBefore(op, user) && !isUsedBefore(user, op))
      op = domFunc();
  }

  /// Returns true if op is used before other.
  bool isUsedBefore(Operation *op, Operation *other) {
    Block *opBlock = op->getBlock();
    Block *otherBlock = other->getBlock();

    // Both Operations are in the same block.
    if (opBlock == otherBlock)
      return op->isBeforeInBlock(other);

    // Check if op is used in a dominator of other.
    if (dominators.dominates(opBlock, otherBlock))
      return true;

    // Recursive call to find if the otherBlock is a successor of opBlock. The
    // common postdominator is used as a termination condition.
    Block *postDom =
        postDominators.findNearestCommonDominator(opBlock, otherBlock);
    return isSuccessor(opBlock, otherBlock, postDom, SmallPtrSet<Block *, 6>{});
  }

  /// Recursive function that returns true if the target Block is a successor of
  /// the currentBlock.
  bool isSuccessor(Block *currentBlock, Block *target, Block *postDom,
                   SmallPtrSet<Block *, 6> visited) {
    if (currentBlock == target)
      return true;
    if (currentBlock == postDom)
      return false;
    for (Block *succ : currentBlock->getSuccessors()) {
      if (visited.insert(succ).second &&
          isSuccessor(succ, target, postDom, visited))
        return true;
    }
    return false;
  }

  /// Cache the alias lists for all values to avoid the recomputation.
  BufferAliasAnalysis::ValueMapT aliasCache;

  /// The current dominance info.
  DominanceInfo dominators;

  /// The current postdominance info.
  PostDominanceInfo postDominators;
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

/// The promote buffer to stack pass that tries to convert alloc nodes into
/// alloca nodes.
struct PromoteBuffersToStackPass
    : PromoteBuffersToStackBase<PromoteBuffersToStackPass> {

  PromoteBuffersToStackPass(unsigned maxAllocSizeInBytes,
                            unsigned bitwidthOfIndexType) {
    this->maxAllocSizeInBytes = maxAllocSizeInBytes;
    this->bitwidthOfIndexType = bitwidthOfIndexType;
  }

  void runOnFunction() override {
    // Move all allocation nodes and convert candidates into allocas.
    BufferPlacementPromotion optimizer(getFunction());
    optimizer.promote(this->maxAllocSizeInBytes, this->bitwidthOfIndexType);
  }
};

/// The buffer reuse pass that uses already allocated buffers if all critera
/// are met.
struct BufferReusePass : BufferReuseBase<BufferReusePass> {

  void runOnFunction() override {
    // Reuse allocated buffer instead of new allocation.
    BufferReuse optimizer(getFunction());
    optimizer.reuse();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::createBufferHoistingPass() {
  return std::make_unique<BufferHoistingPass>();
}

std::unique_ptr<Pass> mlir::createBufferLoopHoistingPass() {
  return std::make_unique<BufferLoopHoistingPass>();
}

std::unique_ptr<Pass>
mlir::createPromoteBuffersToStackPass(unsigned maxAllocSizeInBytes,
                                      unsigned bitwidthOfIndexType) {
  return std::make_unique<PromoteBuffersToStackPass>(maxAllocSizeInBytes,
                                                     bitwidthOfIndexType);
}

std::unique_ptr<Pass> mlir::createBufferReusePass() {
  return std::make_unique<BufferReusePass>();
}
