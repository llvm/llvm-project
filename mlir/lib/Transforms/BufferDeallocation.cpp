//===- BufferDeallocation.cpp - the impl for buffer deallocation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for computing correct alloc and dealloc positions.
// Furthermore, buffer placement also adds required new alloc and copy
// operations to ensure that all buffers are deallocated. The main class is the
// BufferDeallocationPass class that implements the underlying algorithm. In
// order to put allocations and deallocations at safe positions, it is
// significantly important to put them into the correct blocks. However, the
// liveness analysis does not pay attention to aliases, which can occur due to
// branches (and their associated block arguments) in general. For this purpose,
// BufferDeallocation firstly finds all possible aliases for a single value
// (using the BufferAliasAnalysis class). Consider the following
// example:
//
// ^bb0(%arg0):
//   cond_br %cond, ^bb1, ^bb2
// ^bb1:
//   br ^exit(%arg0)
// ^bb2:
//   %new_value = ...
//   br ^exit(%new_value)
// ^exit(%arg1):
//   return %arg1;
//
// We should place the dealloc for %new_value in exit. However, we have to free
// the buffer in the same block, because it cannot be freed in the post
// dominator. However, this requires a new copy buffer for %arg1 that will
// contain the actual contents. Using the class BufferAliasAnalysis, we
// will find out that %new_value has a potential alias %arg1. In order to find
// the dealloc position we have to find all potential aliases, iterate over
// their uses and find the common post-dominator block (note that additional
// copies and buffers remove potential aliases and will influence the placement
// of the deallocs). In all cases, the computed block can be safely used to free
// the %new_value buffer (may be exit or bb2) as it will die and we can use
// liveness information to determine the exact operation after which we have to
// insert the dealloc. However, the algorithm supports introducing copy buffers
// and placing deallocs in safe locations to ensure that all buffers will be
// freed in the end.
//
// TODO:
// The current implementation does not support explicit-control-flow loops and
// the resulting code will be invalid with respect to program semantics.
// However, structured control-flow loops are fully supported. Furthermore, it
// doesn't accept functions which return buffers already.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;

/// Walks over all immediate return-like terminators in the given region.
template <typename FuncT>
static void walkReturnOperations(Region *region, const FuncT &func) {
  for (Block &block : *region)
    for (Operation &operation : block) {
      // Skip non-return-like terminators.
      if (operation.hasTrait<OpTrait::ReturnLike>())
        func(&operation);
    }
}

/// Wrapper for the actual `RegionBranchOpInterface.getSuccessorRegions`
/// function that initializes the required `operandAttributes` array.
static void getSuccessorRegions(RegionBranchOpInterface regionInterface,
                                llvm::Optional<unsigned> index,
                                SmallVectorImpl<RegionSuccessor> &successors) {
  // Create a list of null attributes for each operand to comply with the
  // `getSuccessorRegions` interface definition that requires a single
  // attribute per operand.
  SmallVector<Attribute, 2> operandAttributes(
      regionInterface.getOperation()->getNumOperands());

  // Get all successor regions using the temporarily allocated
  // `operandAttributes`.
  regionInterface.getSuccessorRegions(index, operandAttributes, successors);
}

//===----------------------------------------------------------------------===//
// BufferPlacementAllocs
//===----------------------------------------------------------------------===//

/// Get the start operation to place the given alloc value withing the
// specified placement block.
Operation *BufferPlacementAllocs::getStartOperation(Value allocValue,
                                                    Block *placementBlock,
                                                    const Liveness &liveness) {
  // We have to ensure that we place the alloc before its first use in this
  // block.
  const LivenessBlockInfo &livenessInfo = *liveness.getLiveness(placementBlock);
  Operation *startOperation = livenessInfo.getStartOperation(allocValue);
  // Check whether the start operation lies in the desired placement block.
  // If not, we will use the terminator as this is the last operation in
  // this block.
  if (startOperation->getBlock() != placementBlock) {
    Operation *opInPlacementBlock =
        placementBlock->findAncestorOpInBlock(*startOperation);
    startOperation = opInPlacementBlock ? opInPlacementBlock
                                        : placementBlock->getTerminator();
  }

  return startOperation;
}

/// Finds associated deallocs that can be linked to our allocation nodes (if
/// any).
Operation *BufferPlacementAllocs::findDealloc(Value allocValue) {
  auto userIt = llvm::find_if(allocValue.getUsers(), [&](Operation *user) {
    auto effectInterface = dyn_cast<MemoryEffectOpInterface>(user);
    if (!effectInterface)
      return false;
    // Try to find a free effect that is applied to one of our values
    // that will be automatically freed by our pass.
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    effectInterface.getEffectsOnValue(allocValue, effects);
    return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &it) {
      return isa<MemoryEffects::Free>(it.getEffect());
    });
  });
  // Assign the associated dealloc operation (if any).
  return userIt != allocValue.user_end() ? *userIt : nullptr;
}

/// Initializes the internal list by discovering all supported allocation
/// nodes.
BufferPlacementAllocs::BufferPlacementAllocs(Operation *op) { build(op); }

/// Searches for and registers all supported allocation entries.
void BufferPlacementAllocs::build(Operation *op) {
  op->walk([&](MemoryEffectOpInterface opInterface) {
    // Try to find a single allocation result.
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    opInterface.getEffects(effects);

    SmallVector<MemoryEffects::EffectInstance, 2> allocateResultEffects;
    llvm::copy_if(
        effects, std::back_inserter(allocateResultEffects),
        [=](MemoryEffects::EffectInstance &it) {
          Value value = it.getValue();
          return isa<MemoryEffects::Allocate>(it.getEffect()) && value &&
                 value.isa<OpResult>() &&
                 it.getResource() !=
                     SideEffects::AutomaticAllocationScopeResource::get();
        });
    // If there is one result only, we will be able to move the allocation and
    // (possibly existing) deallocation ops.
    if (allocateResultEffects.size() != 1)
      return;
    // Get allocation result.
    Value allocValue = allocateResultEffects[0].getValue();
    // Find the associated dealloc value and register the allocation entry.
    allocs.push_back(std::make_tuple(allocValue, findDealloc(allocValue)));
  });
}

//===----------------------------------------------------------------------===//
// BufferPlacementTransformationBase
//===----------------------------------------------------------------------===//

/// Constructs a new transformation base using the given root operation.
BufferPlacementTransformationBase::BufferPlacementTransformationBase(
    Operation *op)
    : aliases(op), allocs(op), liveness(op) {}

/// Returns true if the given operation represents a loop by testing whether it
/// implements the `LoopLikeOpInterface` or the `RegionBranchOpInterface`. In
/// the case of a `RegionBranchOpInterface`, it checks all region-based control-
/// flow edges for cycles.
bool BufferPlacementTransformationBase::isLoop(Operation *op) {
  // If the operation implements the `LoopLikeOpInterface` it can be considered
  // a loop.
  if (isa<LoopLikeOpInterface>(op))
    return true;

  // If the operation does not implement the `RegionBranchOpInterface`, it is
  // (currently) not possible to detect a loop.
  RegionBranchOpInterface regionInterface;
  if (!(regionInterface = dyn_cast<RegionBranchOpInterface>(op)))
    return false;

  // Recurses into a region using the current region interface to find potential
  // cycles.
  SmallPtrSet<Region *, 4> visitedRegions;
  std::function<bool(Region *)> recurse = [&](Region *current) {
    if (!current)
      return false;
    // If we have found a back edge, the parent operation induces a loop.
    if (!visitedRegions.insert(current).second)
      return true;
    // Recurses into all region successors.
    SmallVector<RegionSuccessor, 2> successors;
    getSuccessorRegions(regionInterface, current->getRegionNumber(),
                        successors);
    for (RegionSuccessor &regionEntry : successors)
      if (recurse(regionEntry.getSuccessor()))
        return true;
    return false;
  };

  // Start with all entry regions and test whether they induce a loop.
  SmallVector<RegionSuccessor, 2> successorRegions;
  getSuccessorRegions(regionInterface, /*index=*/llvm::None, successorRegions);
  for (RegionSuccessor &regionEntry : successorRegions) {
    if (recurse(regionEntry.getSuccessor()))
      return true;
    visitedRegions.clear();
  }

  return false;
}

namespace {

//===----------------------------------------------------------------------===//
// Backedges analysis
//===----------------------------------------------------------------------===//

/// A straight-forward program analysis which detects loop backedges induced by
/// explicit control flow.
class Backedges {
public:
  using BlockSetT = SmallPtrSet<Block *, 16>;
  using BackedgeSetT = llvm::DenseSet<std::pair<Block *, Block *>>;

public:
  /// Constructs a new backedges analysis using the op provided.
  Backedges(Operation *op) { recurse(op, op->getBlock()); }

  /// Returns the number of backedges formed by explicit control flow.
  size_t size() const { return edgeSet.size(); }

  /// Returns the start iterator to loop over all backedges.
  BackedgeSetT::const_iterator begin() const { return edgeSet.begin(); }

  /// Returns the end iterator to loop over all backedges.
  BackedgeSetT::const_iterator end() const { return edgeSet.end(); }

private:
  /// Enters the current block and inserts a backedge into the `edgeSet` if we
  /// have already visited the current block. The inserted edge links the given
  /// `predecessor` with the `current` block.
  bool enter(Block &current, Block *predecessor) {
    bool inserted = visited.insert(&current).second;
    if (!inserted)
      edgeSet.insert(std::make_pair(predecessor, &current));
    return inserted;
  }

  /// Leaves the current block.
  void exit(Block &current) { visited.erase(&current); }

  /// Recurses into the given operation while taking all attached regions into
  /// account.
  void recurse(Operation *op, Block *predecessor) {
    Block *current = op->getBlock();
    // If the current op implements the `BranchOpInterface`, there can be
    // cycles in the scope of all successor blocks.
    if (isa<BranchOpInterface>(op)) {
      for (Block *succ : current->getSuccessors())
        recurse(*succ, current);
    }
    // Recurse into all distinct regions and check for explicit control-flow
    // loops.
    for (Region &region : op->getRegions())
      recurse(region.front(), current);
  }

  /// Recurses into explicit control-flow structures that are given by
  /// the successor relation defined on the block level.
  void recurse(Block &block, Block *predecessor) {
    // Try to enter the current block. If this is not possible, we are
    // currently processing this block and can safely return here.
    if (!enter(block, predecessor))
      return;

    // Recurse into all operations and successor blocks.
    for (Operation &op : block.getOperations())
      recurse(&op, predecessor);

    // Leave the current block.
    exit(block);
  }

  /// Stores all blocks that are currently visited and on the processing stack.
  BlockSetT visited;

  /// Stores all backedges in the format (source, target).
  BackedgeSetT edgeSet;
};

//===----------------------------------------------------------------------===//
// BufferDeallocation
//===----------------------------------------------------------------------===//

/// The buffer deallocation transformation which ensures that all allocs in the
/// program have a corresponding de-allocation. As a side-effect, it might also
/// introduce copies that in turn leads to additional allocs and de-allocations.
class BufferDeallocation : BufferPlacementTransformationBase {
public:
  BufferDeallocation(Operation *op)
      : BufferPlacementTransformationBase(op), dominators(op),
        postDominators(op) {}

  /// Performs the actual placement/creation of all temporary alloc, copy and
  /// dealloc nodes.
  void deallocate() {
    // Add additional allocations and copies that are required.
    introduceCopies();
    // Place deallocations for all allocation entries.
    placeDeallocs();
  }

private:
  /// Introduces required allocs and copy operations to avoid memory leaks.
  void introduceCopies() {
    // Initialize the set of values that require a dedicated memory free
    // operation since their operands cannot be safely deallocated in a post
    // dominator.
    SmallPtrSet<Value, 8> valuesToFree;
    llvm::SmallDenseSet<std::tuple<Value, Block *>> visitedValues;
    SmallVector<std::tuple<Value, Block *>, 8> toProcess;

    // Check dominance relation for proper dominance properties. If the given
    // value node does not dominate an alias, we will have to create a copy in
    // order to free all buffers that can potentially leak into a post
    // dominator.
    auto findUnsafeValues = [&](Value source, Block *definingBlock) {
      auto it = aliases.find(source);
      if (it == aliases.end())
        return;
      for (Value value : it->second) {
        if (valuesToFree.count(value) > 0)
          continue;
        Block *parentBlock = value.getParentBlock();
        // Check whether we have to free this particular block argument or
        // generic value. We have to free the current alias if it is either
        // defined in a non-dominated block or it is defined in the same block
        // but the current value is not dominated by the source value.
        if (!dominators.dominates(definingBlock, parentBlock) ||
            (definingBlock == parentBlock && value.isa<BlockArgument>())) {
          toProcess.emplace_back(value, parentBlock);
          valuesToFree.insert(value);
        } else if (visitedValues.insert(std::make_tuple(value, definingBlock))
                       .second)
          toProcess.emplace_back(value, definingBlock);
      }
    };

    // Detect possibly unsafe aliases starting from all allocations.
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);
      findUnsafeValues(allocValue, allocValue.getDefiningOp()->getBlock());
    }
    // Try to find block arguments that require an explicit free operation
    // until we reach a fix point.
    while (!toProcess.empty()) {
      auto current = toProcess.pop_back_val();
      findUnsafeValues(std::get<0>(current), std::get<1>(current));
    }

    // Update buffer aliases to ensure that we free all buffers and block
    // arguments at the correct locations.
    aliases.remove(valuesToFree);

    // Add new allocs and additional copy operations.
    for (Value value : valuesToFree) {
      if (auto blockArg = value.dyn_cast<BlockArgument>())
        introduceBlockArgCopy(blockArg);
      else
        introduceValueCopyForRegionResult(value);

      // Register the value to require a final dealloc. Note that we do not have
      // to assign a block here since we do not want to move the allocation node
      // to another location.
      allocs.registerAlloc(std::make_tuple(value, nullptr));
    }
  }

  /// Introduces temporary allocs in all predecessors and copies the source
  /// values into the newly allocated buffers.
  void introduceBlockArgCopy(BlockArgument blockArg) {
    // Allocate a buffer for the current block argument in the block of
    // the associated value (which will be a predecessor block by
    // definition).
    Block *block = blockArg.getOwner();
    for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
      // Get the terminator and the value that will be passed to our
      // argument.
      Operation *terminator = (*it)->getTerminator();
      auto branchInterface = cast<BranchOpInterface>(terminator);
      // Query the associated source value.
      Value sourceValue =
          branchInterface.getSuccessorOperands(it.getSuccessorIndex())
              .getValue()[blockArg.getArgNumber()];
      // Create a new alloc and copy at the current location of the terminator.
      Value alloc = introduceBufferCopy(sourceValue, terminator);
      // Wire new alloc and successor operand.
      auto mutableOperands =
          branchInterface.getMutableSuccessorOperands(it.getSuccessorIndex());
      if (!mutableOperands.hasValue())
        terminator->emitError() << "terminators with immutable successor "
                                   "operands are not supported";
      else
        mutableOperands.getValue()
            .slice(blockArg.getArgNumber(), 1)
            .assign(alloc);
    }

    // Check whether the block argument has implicitly defined predecessors via
    // the RegionBranchOpInterface. This can be the case if the current block
    // argument belongs to the first block in a region and the parent operation
    // implements the RegionBranchOpInterface.
    Region *argRegion = block->getParent();
    Operation *parentOp = argRegion->getParentOp();
    RegionBranchOpInterface regionInterface;
    if (!argRegion || &argRegion->front() != block ||
        !(regionInterface = dyn_cast<RegionBranchOpInterface>(parentOp)))
      return;

    introduceCopiesForRegionSuccessors(
        regionInterface, argRegion->getParentOp()->getRegions(), blockArg,
        [&](RegionSuccessor &successorRegion) {
          // Find a predecessor of our argRegion.
          return successorRegion.getSuccessor() == argRegion;
        });

    // Check whether the block argument belongs to an entry region of the
    // parent operation. In this case, we have to introduce an additional copy
    // for buffer that is passed to the argument.
    SmallVector<RegionSuccessor, 2> successorRegions;
    getSuccessorRegions(regionInterface, /*index=*/llvm::None,
                        successorRegions);
    auto *it =
        llvm::find_if(successorRegions, [&](RegionSuccessor &successorRegion) {
          return successorRegion.getSuccessor() == argRegion;
        });
    if (it == successorRegions.end())
      return;

    // Determine the actual operand to introduce a copy for and rewire the
    // operand to point to the copy instead.
    Value operand =
        regionInterface.getSuccessorEntryOperands(argRegion->getRegionNumber())
            [llvm::find(it->getSuccessorInputs(), blockArg).getIndex()];
    Value copy = introduceBufferCopy(operand, parentOp);

    auto op = llvm::find(parentOp->getOperands(), operand);
    assert(op != parentOp->getOperands().end() &&
           "parentOp does not contain operand");
    parentOp->setOperand(op.getIndex(), copy);
  }

  /// Introduces temporary allocs in front of all associated nested-region
  /// terminators and copies the source values into the newly allocated buffers.
  void introduceValueCopyForRegionResult(Value value) {
    // Get the actual result index in the scope of the parent terminator.
    Operation *operation = value.getDefiningOp();
    auto regionInterface = cast<RegionBranchOpInterface>(operation);
    // Filter successors that return to the parent operation.
    auto regionPredicate = [&](RegionSuccessor &successorRegion) {
      // If the RegionSuccessor has no associated successor, it will return to
      // its parent operation.
      return !successorRegion.getSuccessor();
    };
    // Introduce a copy for all region "results" that are returned to the parent
    // operation. This is required since the parent's result value has been
    // considered critical. Therefore, the algorithm assumes that a copy of a
    // previously allocated buffer is returned by the operation (like in the
    // case of a block argument).
    introduceCopiesForRegionSuccessors(regionInterface, operation->getRegions(),
                                       value, regionPredicate);
  }

  /// Introduces buffer copies for all terminators in the given regions. The
  /// regionPredicate is applied to every successor region in order to restrict
  /// the copies to specific regions.
  template <typename TPredicate>
  void introduceCopiesForRegionSuccessors(
      RegionBranchOpInterface regionInterface, MutableArrayRef<Region> regions,
      Value argValue, const TPredicate &regionPredicate) {
    for (Region &region : regions) {
      // Query the regionInterface to get all successor regions of the current
      // one.
      SmallVector<RegionSuccessor, 2> successorRegions;
      getSuccessorRegions(regionInterface, region.getRegionNumber(),
                          successorRegions);
      // Try to find a matching region successor.
      RegionSuccessor *regionSuccessor =
          llvm::find_if(successorRegions, regionPredicate);
      if (regionSuccessor == successorRegions.end())
        continue;
      // Get the operand index in the context of the current successor input
      // bindings.
      size_t operandIndex =
          llvm::find(regionSuccessor->getSuccessorInputs(), argValue)
              .getIndex();

      // Iterate over all immediate terminator operations to introduce
      // new buffer allocations. Thereby, the appropriate terminator operand
      // will be adjusted to point to the newly allocated buffer instead.
      walkReturnOperations(&region, [&](Operation *terminator) {
        // Extract the source value from the current terminator.
        Value sourceValue = terminator->getOperand(operandIndex);
        // Create a new alloc at the current location of the terminator.
        Value alloc = introduceBufferCopy(sourceValue, terminator);
        // Wire alloc and terminator operand.
        terminator->setOperand(operandIndex, alloc);
      });
    }
  }

  /// Creates a new memory allocation for the given source value and copies
  /// its content into the newly allocated buffer. The terminator operation is
  /// used to insert the alloc and copy operations at the right places.
  Value introduceBufferCopy(Value sourceValue, Operation *terminator) {
    // Avoid multiple copies of the same source value. This can happen in the
    // presence of loops when a branch acts as a backedge while also having
    // another successor that returns to its parent operation. Note: that
    // copying copied buffers can introduce memory leaks since the invariant of
    // BufferPlacement assumes that a buffer will be only copied once into a
    // temporary buffer. Hence, the construction of copy chains introduces
    // additional allocations that are not tracked automatically by the
    // algorithm.
    if (copiedValues.contains(sourceValue))
      return sourceValue;
    // Create a new alloc at the current location of the terminator.
    auto memRefType = sourceValue.getType().cast<MemRefType>();
    OpBuilder builder(terminator);

    // Extract information about dynamically shaped types by
    // extracting their dynamic dimensions.
    SmallVector<Value, 4> dynamicOperands;
    for (auto shapeElement : llvm::enumerate(memRefType.getShape())) {
      if (!ShapedType::isDynamic(shapeElement.value()))
        continue;
      dynamicOperands.push_back(builder.create<DimOp>(
          terminator->getLoc(), sourceValue, shapeElement.index()));
    }

    // TODO: provide a generic interface to create dialect-specific
    // Alloc and CopyOp nodes.
    auto alloc = builder.create<AllocOp>(terminator->getLoc(), memRefType,
                                         dynamicOperands);

    // Create a new copy operation that copies to contents of the old
    // allocation to the new one.
    builder.create<linalg::CopyOp>(terminator->getLoc(), sourceValue, alloc);

    // Remember the copy of original source value.
    copiedValues.insert(alloc);
    return alloc;
  }

  /// Finds correct dealloc positions according to the algorithm described at
  /// the top of the file for all alloc nodes and block arguments that can be
  /// handled by this analysis.
  void placeDeallocs() const {
    // Move or insert deallocs using the previously computed information.
    // These deallocations will be linked to their associated allocation nodes
    // since they don't have any aliases that can (potentially) increase their
    // liveness.
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value alloc = std::get<0>(entry);
      auto aliasesSet = aliases.resolve(alloc);
      assert(aliasesSet.size() > 0 && "must contain at least one alias");

      // Determine the actual block to place the dealloc and get liveness
      // information.
      Block *placementBlock =
          findCommonDominator(alloc, aliasesSet, postDominators);
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(placementBlock);

      // We have to ensure that the dealloc will be after the last use of all
      // aliases of the given value. We first assume that there are no uses in
      // the placementBlock and that we can safely place the dealloc at the
      // beginning.
      Operation *endOperation = &placementBlock->front();

      // Iterate over all aliases and ensure that the endOperation will point
      // to the last operation of all potential aliases in the placementBlock.
      for (Value alias : aliasesSet) {
        // Ensure that the start operation is at least the defining operation of
        // the current alias to avoid invalid placement of deallocs for aliases
        // without any uses.
        Operation *beforeOp = endOperation;
        if (alias.getDefiningOp() &&
            !(beforeOp = placementBlock->findAncestorOpInBlock(
                  *alias.getDefiningOp())))
          continue;

        Operation *aliasEndOperation =
            livenessInfo->getEndOperation(alias, beforeOp);
        // Check whether the aliasEndOperation lies in the desired block and
        // whether it is behind the current endOperation. If yes, this will be
        // the new endOperation.
        if (aliasEndOperation->getBlock() == placementBlock &&
            endOperation->isBeforeInBlock(aliasEndOperation))
          endOperation = aliasEndOperation;
      }
      // endOperation is the last operation behind which we can safely store
      // the dealloc taking all potential aliases into account.

      // If there is an existing dealloc, move it to the right place.
      Operation *deallocOperation = std::get<1>(entry);
      if (deallocOperation) {
        deallocOperation->moveAfter(endOperation);
      } else {
        // If the Dealloc position is at the terminator operation of the
        // block, then the value should escape from a deallocation.
        Operation *nextOp = endOperation->getNextNode();
        if (!nextOp)
          continue;
        // If there is no dealloc node, insert one in the right place.
        OpBuilder builder(nextOp);
        builder.create<DeallocOp>(alloc.getLoc(), alloc);
      }
    }
  }

  /// The dominator info to find the appropriate start operation to move the
  /// allocs.
  DominanceInfo dominators;

  /// The post dominator info to move the dependent allocs in the right
  /// position.
  PostDominanceInfo postDominators;

  /// Stores already copied allocations to avoid additional copies of copies.
  ValueSetT copiedValues;
};

//===----------------------------------------------------------------------===//
// BufferDeallocationPass
//===----------------------------------------------------------------------===//

/// The actual buffer deallocation pass that inserts and moves dealloc nodes
/// into the right positions. Furthermore, it inserts additional allocs and
/// copies if necessary. It uses the algorithm described at the top of the file.
struct BufferDeallocationPass : BufferDeallocationBase<BufferDeallocationPass> {

  void runOnFunction() override {
    // Ensure that there are supported loops only.
    Backedges backedges(getFunction());
    if (backedges.size()) {
      getFunction().emitError(
          "Structured control-flow loops are supported only.");
      return;
    }

    // Place all required temporary alloc, copy and dealloc nodes.
    BufferDeallocation deallocation(getFunction());
    deallocation.deallocate();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BufferDeallocationPass construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::createBufferDeallocationPass() {
  return std::make_unique<BufferDeallocationPass>();
}
