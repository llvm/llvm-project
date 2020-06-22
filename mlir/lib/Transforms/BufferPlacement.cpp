//===- BufferPlacement.cpp - the impl for buffer placement ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for computing correct alloc and dealloc positions.
// Furthermore, buffer placement also adds required new alloc and copy
// operations to ensure that all buffers are deallocated.The main class is the
// BufferPlacementPass class that implements the underlying algorithm. In order
// to put allocations and deallocations at safe positions, it is significantly
// important to put them into the correct blocks. However, the liveness analysis
// does not pay attention to aliases, which can occur due to branches (and their
// associated block arguments) in general. For this purpose, BufferPlacement
// firstly finds all possible aliases for a single value (using the
// BufferPlacementAliasAnalysis class). Consider the following example:
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
// Using liveness information on its own would cause us to place the allocs and
// deallocs in the wrong block. This is due to the fact that %new_value will not
// be liveOut of its block. Instead, we can place the alloc for %new_value
// in bb0 and its associated dealloc in exit. Alternatively, the alloc can stay
// (or even has to stay due to additional dependencies) at this location and we
// have to free the buffer in the same block, because it cannot be freed in the
// post dominator. However, this requires a new copy buffer for %arg1 that will
// contain the actual contents. Using the class BufferPlacementAliasAnalysis, we
// will find out that %new_value has a potential alias %arg1. In order to find
// the dealloc position we have to find all potential aliases, iterate over
// their uses and find the common post-dominator block (note that additional
// copies and buffers remove potential aliases and will influence the placement
// of the deallocs). In all cases, the computed block can be safely used to free
// the %new_value buffer (may be exit or bb2) as it will die and we can use
// liveness information to determine the exact operation after which we have to
// insert the dealloc. Finding the alloc position is similar and non-obvious.
// However, the algorithm supports moving allocs to other places and introducing
// copy buffers and placing deallocs in safe places to ensure that all buffers
// will be freed in the end.
//
// TODO:
// The current implementation does not support loops and the resulting code will
// be invalid with respect to program semantics. The only thing that is
// currently missing is a high-level loop analysis that allows us to move allocs
// and deallocs outside of the loop blocks. Furthermore, it doesn't also accept
// functions which return buffers already.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/BufferPlacement.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// BufferPlacementAliasAnalysis
//===----------------------------------------------------------------------===//

/// A straight-forward alias analysis which ensures that all aliases of all
/// values will be determined. This is a requirement for the BufferPlacement
/// class since you need to determine safe positions to place alloc and
/// deallocs.
class BufferPlacementAliasAnalysis {
public:
  using ValueSetT = SmallPtrSet<Value, 16>;
  using ValueMapT = llvm::DenseMap<Value, ValueSetT>;

public:
  /// Constructs a new alias analysis using the op provided.
  BufferPlacementAliasAnalysis(Operation *op) { build(op->getRegions()); }

  /// Find all immediate aliases this value could potentially have.
  ValueMapT::const_iterator find(Value value) const {
    return aliases.find(value);
  }

  /// Returns the end iterator that can be used in combination with find.
  ValueMapT::const_iterator end() const { return aliases.end(); }

  /// Find all immediate and indirect aliases this value could potentially
  /// have. Note that the resulting set will also contain the value provided as
  /// it is an alias of itself.
  ValueSetT resolve(Value value) const {
    ValueSetT result;
    resolveRecursive(value, result);
    return result;
  }

  /// Removes the given values from all alias sets.
  void remove(const SmallPtrSetImpl<BlockArgument> &aliasValues) {
    for (auto &entry : aliases)
      llvm::set_subtract(entry.second, aliasValues);
  }

private:
  /// Recursively determines alias information for the given value. It stores
  /// all newly found potential aliases in the given result set.
  void resolveRecursive(Value value, ValueSetT &result) const {
    if (!result.insert(value).second)
      return;
    auto it = aliases.find(value);
    if (it == aliases.end())
      return;
    for (Value alias : it->second)
      resolveRecursive(alias, result);
  }

  /// This function constructs a mapping from values to its immediate aliases.
  /// It iterates over all blocks, gets their predecessors, determines the
  /// values that will be passed to the corresponding block arguments and
  /// inserts them into the underlying map.
  void build(MutableArrayRef<Region> regions) {
    for (Region &region : regions) {
      for (Block &block : region) {
        // Iterate over all predecessor and get the mapped values to their
        // corresponding block arguments values.
        for (auto it = block.pred_begin(), e = block.pred_end(); it != e;
             ++it) {
          unsigned successorIndex = it.getSuccessorIndex();
          // Get the terminator and the values that will be passed to our block.
          auto branchInterface =
              dyn_cast<BranchOpInterface>((*it)->getTerminator());
          if (!branchInterface)
            continue;
          // Query the branch op interace to get the successor operands.
          auto successorOperands =
              branchInterface.getSuccessorOperands(successorIndex);
          if (successorOperands.hasValue()) {
            // Build the actual mapping of values to their immediate aliases.
            for (auto argPair : llvm::zip(block.getArguments(),
                                          successorOperands.getValue())) {
              aliases[std::get<1>(argPair)].insert(std::get<0>(argPair));
            }
          }
        }
      }
    }
  }

  /// Maps values to all immediate aliases this value can have.
  ValueMapT aliases;
};

//===----------------------------------------------------------------------===//
// BufferPlacement
//===----------------------------------------------------------------------===//

// The main buffer placement analysis used to place allocs, copies and deallocs.
class BufferPlacement {
public:
  using ValueSetT = BufferPlacementAliasAnalysis::ValueSetT;

  /// An intermediate representation of a single allocation node.
  struct AllocEntry {
    /// A reference to the associated allocation node.
    Value allocValue;

    /// The associated placement block in which the allocation should be
    /// performed.
    Block *placementBlock;

    /// The associated dealloc operation (if any).
    Operation *deallocOperation;
  };

  using AllocEntryList = SmallVector<AllocEntry, 8>;

public:
  BufferPlacement(Operation *op)
      : operation(op), aliases(op), liveness(op), dominators(op),
        postDominators(op) {
    // Gather all allocation nodes
    initBlockMapping();
  }

  /// Performs the actual placement/creation of all alloc, copy and dealloc
  /// nodes.
  void place() {
    // Place all allocations.
    placeAllocs();
    // Add additional allocations and copies that are required.
    introduceCopies();
    // Find all associated dealloc nodes.
    findDeallocs();
    // Place deallocations for all allocation entries.
    placeDeallocs();
  }

private:
  /// Initializes the internal block mapping by discovering allocation nodes. It
  /// maps all allocation nodes to their initial block in which they can be
  /// safely allocated.
  void initBlockMapping() {
    operation->walk([&](MemoryEffectOpInterface opInterface) {
      // Try to find a single allocation result.
      SmallVector<MemoryEffects::EffectInstance, 2> effects;
      opInterface.getEffects(effects);

      SmallVector<MemoryEffects::EffectInstance, 2> allocateResultEffects;
      llvm::copy_if(effects, std::back_inserter(allocateResultEffects),
                    [=](MemoryEffects::EffectInstance &it) {
                      Value value = it.getValue();
                      return isa<MemoryEffects::Allocate>(it.getEffect()) &&
                             value && value.isa<OpResult>();
                    });
      // If there is one result only, we will be able to move the allocation and
      // (possibly existing) deallocation ops.
      if (allocateResultEffects.size() != 1)
        return;
      // Get allocation result.
      auto allocResult = allocateResultEffects[0].getValue().cast<OpResult>();
      // Find the initial allocation block and register this result.
      allocs.push_back(
          {allocResult, getInitialAllocBlock(allocResult), nullptr});
    });
  }

  /// Computes a valid allocation position in a dominator (if possible) for the
  /// given allocation result.
  Block *getInitialAllocBlock(OpResult result) {
    // Get all allocation operands as these operands are important for the
    // allocation operation.
    auto operands = result.getOwner()->getOperands();
    if (operands.size() < 1)
      return findCommonDominator(result, aliases.resolve(result), dominators);

    // If this node has dependencies, check all dependent nodes with respect
    // to a common post dominator in which all values are available.
    ValueSetT dependencies(++operands.begin(), operands.end());
    return findCommonDominator(*operands.begin(), dependencies, postDominators);
  }

  /// Finds correct alloc positions according to the algorithm described at
  /// the top of the file for all alloc nodes that can be handled by this
  /// analysis.
  void placeAllocs() const {
    for (auto &entry : allocs) {
      Value alloc = entry.allocValue;
      // Get the actual block to place the alloc and get liveness information
      // for the placement block.
      Block *placementBlock = entry.placementBlock;
      // We have to ensure that we place the alloc before its first use in this
      // block.
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(placementBlock);
      Operation *startOperation = livenessInfo->getStartOperation(alloc);
      // Check whether the start operation lies in the desired placement block.
      // If not, we will use the terminator as this is the last operation in
      // this block.
      if (startOperation->getBlock() != placementBlock)
        startOperation = placementBlock->getTerminator();

      // Move the alloc in front of the start operation.
      Operation *allocOperation = alloc.getDefiningOp();
      allocOperation->moveBefore(startOperation);
    }
  }

  /// Introduces required allocs and copy operations to avoid memory leaks.
  void introduceCopies() {
    // Initialize the set of block arguments that require a dedicated memory
    // free operation since their arguments cannot be safely deallocated in a
    // post dominator.
    SmallPtrSet<BlockArgument, 8> blockArgsToFree;
    llvm::SmallDenseSet<std::tuple<BlockArgument, Block *>> visitedBlockArgs;
    SmallVector<std::tuple<BlockArgument, Block *>, 8> toProcess;

    // Check dominance relation for proper dominance properties. If the given
    // value node does not dominate an alias, we will have to create a copy in
    // order to free all buffers that can potentially leak into a post
    // dominator.
    auto findUnsafeValues = [&](Value source, Block *definingBlock) {
      auto it = aliases.find(source);
      if (it == aliases.end())
        return;
      for (Value value : it->second) {
        auto blockArg = value.cast<BlockArgument>();
        if (blockArgsToFree.count(blockArg) > 0)
          continue;
        // Check whether we have to free this particular block argument.
        if (!dominators.dominates(definingBlock, blockArg.getOwner())) {
          toProcess.emplace_back(blockArg, blockArg.getParentBlock());
          blockArgsToFree.insert(blockArg);
        } else if (visitedBlockArgs
                       .insert(std::make_tuple(blockArg, definingBlock))
                       .second)
          toProcess.emplace_back(blockArg, definingBlock);
      }
    };

    // Detect possibly unsafe aliases starting from all allocations.
    for (auto &entry : allocs)
      findUnsafeValues(entry.allocValue, entry.placementBlock);

    // Try to find block arguments that require an explicit free operation
    // until we reach a fix point.
    while (!toProcess.empty()) {
      auto current = toProcess.pop_back_val();
      findUnsafeValues(std::get<0>(current), std::get<1>(current));
    }

    // Update buffer aliases to ensure that we free all buffers and block
    // arguments at the correct locations.
    aliases.remove(blockArgsToFree);

    // Add new allocs and additional copy operations.
    for (BlockArgument blockArg : blockArgsToFree) {
      Block *block = blockArg.getOwner();

      // Allocate a buffer for the current block argument in the block of
      // the associated value (which will be a predecessor block by
      // definition).
      for (auto it = block->pred_begin(), e = block->pred_end(); it != e;
           ++it) {
        // Get the terminator and the value that will be passed to our
        // argument.
        Operation *terminator = (*it)->getTerminator();
        auto branchInterface = cast<BranchOpInterface>(terminator);
        // Convert the mutable operand range to an immutable range and query the
        // associated source value.
        Value sourceValue =
            branchInterface.getSuccessorOperands(it.getSuccessorIndex())
                .getValue()[blockArg.getArgNumber()];
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
        // Wire new alloc and successor operand.
        branchInterface.getMutableSuccessorOperands(it.getSuccessorIndex())
            .getValue()
            .slice(blockArg.getArgNumber(), 1)
            .assign(alloc);
        // Create a new copy operation that copies to contents of the old
        // allocation to the new one.
        builder.create<linalg::CopyOp>(terminator->getLoc(), sourceValue,
                                       alloc);
      }

      // Register the block argument to require a final dealloc. Note that
      // we do not have to assign a block here since we do not want to
      // move the allocation node to another location.
      allocs.push_back({blockArg, nullptr, nullptr});
    }
  }

  /// Finds associated deallocs that can be linked to our allocation nodes (if
  /// any).
  void findDeallocs() {
    for (auto &entry : allocs) {
      auto userIt =
          llvm::find_if(entry.allocValue.getUsers(), [&](Operation *user) {
            auto effectInterface = dyn_cast<MemoryEffectOpInterface>(user);
            if (!effectInterface)
              return false;
            // Try to find a free effect that is applied to one of our values
            // that will be automatically freed by our pass.
            SmallVector<MemoryEffects::EffectInstance, 2> effects;
            effectInterface.getEffectsOnValue(entry.allocValue, effects);
            return llvm::any_of(
                effects, [&](MemoryEffects::EffectInstance &it) {
                  return isa<MemoryEffects::Free>(it.getEffect());
                });
          });
      // Assign the associated dealloc operation (if any).
      if (userIt != entry.allocValue.user_end())
        entry.deallocOperation = *userIt;
    }
  }

  /// Finds correct dealloc positions according to the algorithm described at
  /// the top of the file for all alloc nodes and block arguments that can be
  /// handled by this analysis.
  void placeDeallocs() const {
    // Move or insert deallocs using the previously computed information.
    // These deallocations will be linked to their associated allocation nodes
    // since they don't have any aliases that can (potentially) increase their
    // liveness.
    for (auto &entry : allocs) {
      Value alloc = entry.allocValue;
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
        Operation *aliasEndOperation =
            livenessInfo->getEndOperation(alias, endOperation);
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
      if (entry.deallocOperation) {
        entry.deallocOperation->moveAfter(endOperation);
      } else {
        // If the Dealloc position is at the terminator operation of the block,
        // then the value should escape from a deallocation.
        Operation *nextOp = endOperation->getNextNode();
        if (!nextOp)
          continue;
        // If there is no dealloc node, insert one in the right place.
        OpBuilder builder(nextOp);
        builder.create<DeallocOp>(alloc.getLoc(), alloc);
      }
    }
  }

  /// Finds a common dominator for the given value while taking the positions
  /// of the values in the value set into account. It supports dominator and
  /// post-dominator analyses via template arguments.
  template <typename DominatorT>
  Block *findCommonDominator(Value value, const ValueSetT &values,
                             const DominatorT &doms) const {
    // Start with the current block the value is defined in.
    Block *dom = value.getParentBlock();
    // Iterate over all aliases and their uses to find a safe placement block
    // according to the given dominator information.
    for (Value childValue : values)
      for (Operation *user : childValue.getUsers()) {
        // Move upwards in the dominator tree to find an appropriate
        // dominator block that takes the current use into account.
        dom = doms.findNearestCommonDominator(dom, user->getBlock());
      }
    return dom;
  }

  /// The operation this transformation was constructed from.
  Operation *operation;

  /// Alias information that can be updated during the insertion of copies.
  BufferPlacementAliasAnalysis aliases;

  /// Maps allocation nodes to their associated blocks.
  AllocEntryList allocs;

  /// The underlying liveness analysis to compute fine grained information
  /// about alloc and dealloc positions.
  Liveness liveness;

  /// The dominator analysis to place deallocs in the appropriate blocks.
  DominanceInfo dominators;

  /// The post dominator analysis to place deallocs in the appropriate blocks.
  PostDominanceInfo postDominators;
};

//===----------------------------------------------------------------------===//
// BufferPlacementPass
//===----------------------------------------------------------------------===//

/// The actual buffer placement pass that moves alloc and dealloc nodes into
/// the right positions. It uses the algorithm described at the top of the
/// file.
struct BufferPlacementPass
    : mlir::PassWrapper<BufferPlacementPass, FunctionPass> {

  void runOnFunction() override {
    // Place all required alloc, copy and dealloc nodes.
    BufferPlacement placement(getFunction());
    placement.place();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BufferAssignmentPlacer
//===----------------------------------------------------------------------===//

/// Creates a new assignment placer.
BufferAssignmentPlacer::BufferAssignmentPlacer(Operation *op) : operation(op) {}

/// Computes the actual position to place allocs for the given value.
OpBuilder::InsertPoint
BufferAssignmentPlacer::computeAllocPosition(OpResult result) {
  Operation *owner = result.getOwner();
  return OpBuilder::InsertPoint(owner->getBlock(), Block::iterator(owner));
}

//===----------------------------------------------------------------------===//
// BufferAssignmentTypeConverter
//===----------------------------------------------------------------------===//

/// Registers conversions into BufferAssignmentTypeConverter
BufferAssignmentTypeConverter::BufferAssignmentTypeConverter() {
  // Keep all types unchanged.
  addConversion([](Type type) { return type; });
  // A type conversion that converts ranked-tensor type to memref type.
  addConversion([](RankedTensorType type) {
    return (Type)MemRefType::get(type.getShape(), type.getElementType());
  });
}

/// Checks if `type` has been converted from non-memref type to memref.
bool BufferAssignmentTypeConverter::isConvertedMemref(Type type, Type before) {
  return type.isa<MemRefType>() && !before.isa<MemRefType>();
}

//===----------------------------------------------------------------------===//
// BufferPlacementPass construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::createBufferPlacementPass() {
  return std::make_unique<BufferPlacementPass>();
}
