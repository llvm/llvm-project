//===- Liveness.cpp - Liveness analysis for MLIR --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
/// Builds and holds block information during the construction phase.
struct BlockInfoBuilder {
  using ValueSetT = Liveness::ValueSetT;

  /// Constructs an empty block builder.
  BlockInfoBuilder() = default;

  /// Fills the block builder with initial liveness information.
  BlockInfoBuilder(Block *block) : block(block) {
    auto gatherOutValues = [&](Value value) {
      // Check whether this value will be in the outValues set (its uses escape
      // this block). Due to the SSA properties of the program, the uses must
      // occur after the definition. Therefore, we do not have to check
      // additional conditions to detect an escaping value.
      for (Operation *useOp : value.getUsers()) {
        Block *ownerBlock = useOp->getBlock();
        // Find an owner block in the current region. Note that a value does not
        // escape this block if it is used in a nested region.
        ownerBlock = block->getParent()->findAncestorBlockInRegion(*ownerBlock);
        assert(ownerBlock && "Use leaves the current parent region");
        if (ownerBlock != block) {
          outValues.insert(value);
          break;
        }
      }
    };

    // Mark all block arguments (phis) as defined.
    for (BlockArgument argument : block->getArguments()) {
      // Insert value into the set of defined values.
      defValues.insert(argument);

      // Gather all out values of all arguments in the current block.
      gatherOutValues(argument);
    }

    // Gather out values of all operations in the current block.
    for (Operation &operation : *block)
      for (Value result : operation.getResults())
        gatherOutValues(result);

    // Mark all nested operation results as defined, and nested operation
    // operands as used. All defined value will be removed from the used set
    // at the end.
    block->walk([&](Operation *op) {
      for (Value result : op->getResults())
        defValues.insert(result);
      for (Value operand : op->getOperands())
        useValues.insert(operand);
      for (Region &region : op->getRegions())
        for (Block &child : region.getBlocks())
          for (BlockArgument arg : child.getArguments())
            defValues.insert(arg);
    });
    llvm::set_subtract(useValues, defValues);
  }

  /// Updates live-in information of the current block. To do so it uses the
  /// default liveness-computation formula: newIn = use union out \ def. The
  /// methods returns true, if the set has changed (newIn != in), false
  /// otherwise.
  bool updateLiveIn() {
    ValueSetT newIn = useValues;
    llvm::set_union(newIn, outValues);
    llvm::set_subtract(newIn, defValues);

    // It is sufficient to check the set sizes (instead of their contents) since
    // the live-in set can only grow monotonically during all update operations.
    if (newIn.size() == inValues.size())
      return false;

    inValues = std::move(newIn);
    return true;
  }

  /// Updates live-out information of the current block. It iterates over all
  /// successors and unifies their live-in values with the current live-out
  /// values.
  void updateLiveOut(const DenseMap<Block *, BlockInfoBuilder> &builders) {
    for (Block *succ : block->getSuccessors()) {
      const BlockInfoBuilder &builder = builders.find(succ)->second;
      llvm::set_union(outValues, builder.inValues);
    }
  }

  /// The current block.
  Block *block{nullptr};

  /// The set of all live in values.
  ValueSetT inValues;

  /// The set of all live out values.
  ValueSetT outValues;

  /// The set of all defined values.
  ValueSetT defValues;

  /// The set of all used values.
  ValueSetT useValues;
};
} // namespace

/// Builds the internal liveness block mapping.
static void buildBlockMapping(Operation *operation,
                              DenseMap<Block *, BlockInfoBuilder> &builders) {
  SetVector<Block *> toProcess;

  operation->walk<WalkOrder::PreOrder>([&](Block *block) {
    BlockInfoBuilder &builder =
        builders.try_emplace(block, block).first->second;

    if (builder.updateLiveIn())
      toProcess.insert(block->pred_begin(), block->pred_end());
  });

  // Propagate the in and out-value sets (fixpoint iteration).
  while (!toProcess.empty()) {
    Block *current = toProcess.pop_back_val();
    BlockInfoBuilder &builder = builders[current];

    // Update the current out values.
    builder.updateLiveOut(builders);

    // Compute (potentially) updated live in values.
    if (builder.updateLiveIn())
      toProcess.insert(current->pred_begin(), current->pred_end());
  }
}

//===----------------------------------------------------------------------===//
// Liveness
//===----------------------------------------------------------------------===//

/// Creates a new Liveness analysis that computes liveness information for all
/// associated regions.
Liveness::Liveness(Operation *op) : operation(op) { build(); }

/// Initializes the internal mappings.
void Liveness::build() {
  // Build internal block mapping.
  DenseMap<Block *, BlockInfoBuilder> builders;
  buildBlockMapping(operation, builders);

  // Store internal block data.
  for (auto &entry : builders) {
    BlockInfoBuilder &builder = entry.second;
    LivenessBlockInfo &info = blockMapping[entry.first];

    info.block = builder.block;
    info.inValues = std::move(builder.inValues);
    info.outValues = std::move(builder.outValues);
  }
}

/// Gets liveness info (if any) for the given value.
Liveness::OperationListT Liveness::resolveLiveness(Value value) const {
  OperationListT result;
  SmallPtrSet<Block *, 32> visited;
  SmallVector<Block *, 8> toProcess;

  // Start with the defining block
  Block *currentBlock;
  if (Operation *defOp = value.getDefiningOp())
    currentBlock = defOp->getBlock();
  else
    currentBlock = cast<BlockArgument>(value).getOwner();
  toProcess.push_back(currentBlock);
  visited.insert(currentBlock);

  // Start with all associated blocks
  for (OpOperand &use : value.getUses()) {
    Block *useBlock = use.getOwner()->getBlock();
    if (visited.insert(useBlock).second)
      toProcess.push_back(useBlock);
  }

  while (!toProcess.empty()) {
    // Get block and block liveness information.
    Block *block = toProcess.back();
    toProcess.pop_back();
    const LivenessBlockInfo *blockInfo = getLiveness(block);

    // Note that start and end will be in the same block.
    Operation *start = blockInfo->getStartOperation(value);
    Operation *end = blockInfo->getEndOperation(value, start);

    result.push_back(start);
    while (start != end) {
      start = start->getNextNode();
      result.push_back(start);
    }

    for (Block *successor : block->getSuccessors()) {
      if (getLiveness(successor)->isLiveIn(value) &&
          visited.insert(successor).second)
        toProcess.push_back(successor);
    }
  }

  return result;
}

/// Gets liveness info (if any) for the block.
const LivenessBlockInfo *Liveness::getLiveness(Block *block) const {
  auto it = blockMapping.find(block);
  return it == blockMapping.end() ? nullptr : &it->second;
}

/// Returns a reference to a set containing live-in values.
const Liveness::ValueSetT &Liveness::getLiveIn(Block *block) const {
  return getLiveness(block)->in();
}

/// Returns a reference to a set containing live-out values.
const Liveness::ValueSetT &Liveness::getLiveOut(Block *block) const {
  return getLiveness(block)->out();
}

/// Returns true if `value` is not live after `operation`.
bool Liveness::isDeadAfter(Value value, Operation *operation) const {
  Block *block = operation->getBlock();
  const LivenessBlockInfo *blockInfo = getLiveness(block);

  // The given value escapes the associated block.
  if (blockInfo->isLiveOut(value))
    return false;

  Operation *endOperation = blockInfo->getEndOperation(value, operation);
  // If the operation is a real user of `value` the first check is sufficient.
  // If not, we will have to test whether the end operation is executed before
  // the given operation in the block.
  return endOperation == operation || endOperation->isBeforeInBlock(operation);
}

/// Dumps the liveness information in a human readable format.
void Liveness::dump() const { print(llvm::errs()); }

/// Dumps the liveness information to the given stream.
void Liveness::print(raw_ostream &os) const {
  os << "// ---- Liveness -----\n";

  // Builds unique block/value mappings for testing purposes.
  DenseMap<Block *, size_t> blockIds;
  DenseMap<Operation *, size_t> operationIds;
  DenseMap<Value, size_t> valueIds;
  operation->walk<WalkOrder::PreOrder>([&](Block *block) {
    blockIds.insert({block, blockIds.size()});
    for (BlockArgument argument : block->getArguments())
      valueIds.insert({argument, valueIds.size()});
    for (Operation &operation : *block) {
      operationIds.insert({&operation, operationIds.size()});
      for (Value result : operation.getResults())
        valueIds.insert({result, valueIds.size()});
    }
  });

  // Local printing helpers
  auto printValueRef = [&](Value value) {
    if (value.getDefiningOp())
      os << "val_" << valueIds[value];
    else {
      auto blockArg = cast<BlockArgument>(value);
      os << "arg" << blockArg.getArgNumber() << "@"
         << blockIds[blockArg.getOwner()];
    }
    os << " ";
  };

  auto printValueRefs = [&](const ValueSetT &values) {
    std::vector<Value> orderedValues(values.begin(), values.end());
    llvm::sort(orderedValues, [&](Value left, Value right) {
      return valueIds[left] < valueIds[right];
    });
    for (Value value : orderedValues)
      printValueRef(value);
  };

  // Dump information about in and out values.
  operation->walk<WalkOrder::PreOrder>([&](Block *block) {
    os << "// - Block: " << blockIds[block] << "\n";
    const auto *liveness = getLiveness(block);
    os << "// --- LiveIn: ";
    printValueRefs(liveness->inValues);
    os << "\n// --- LiveOut: ";
    printValueRefs(liveness->outValues);
    os << "\n";

    // Print liveness intervals.
    os << "// --- BeginLivenessIntervals";
    for (Operation &op : *block) {
      if (op.getNumResults() < 1)
        continue;
      os << "\n";
      for (Value result : op.getResults()) {
        os << "// ";
        printValueRef(result);
        os << ":";
        auto liveOperations = resolveLiveness(result);
        llvm::sort(liveOperations, [&](Operation *left, Operation *right) {
          return operationIds[left] < operationIds[right];
        });
        for (Operation *operation : liveOperations) {
          os << "\n//     ";
          operation->print(os);
        }
      }
    }
    os << "\n// --- EndLivenessIntervals\n";

    // Print currently live values.
    os << "// --- BeginCurrentlyLive\n";
    for (Operation &op : *block) {
      auto currentlyLive = liveness->currentlyLiveValues(&op);
      if (currentlyLive.empty())
        continue;
      os << "//     ";
      op.print(os);
      os << " [";
      printValueRefs(currentlyLive);
      os << "\b]\n";
    }
    os << "// --- EndCurrentlyLive\n";
  });
  os << "// -------------------\n";
}

//===----------------------------------------------------------------------===//
// LivenessBlockInfo
//===----------------------------------------------------------------------===//

/// Returns true if the given value is in the live-in set.
bool LivenessBlockInfo::isLiveIn(Value value) const {
  return inValues.count(value);
}

/// Returns true if the given value is in the live-out set.
bool LivenessBlockInfo::isLiveOut(Value value) const {
  return outValues.count(value);
}

/// Gets the start operation for the given value (must be referenced in this
/// block).
Operation *LivenessBlockInfo::getStartOperation(Value value) const {
  Operation *definingOp = value.getDefiningOp();
  // The given value is either live-in or is defined
  // in the scope of this block.
  if (isLiveIn(value) || !definingOp)
    return &block->front();
  return definingOp;
}

/// Gets the end operation for the given value using the start operation
/// provided (must be referenced in this block).
Operation *LivenessBlockInfo::getEndOperation(Value value,
                                              Operation *startOperation) const {
  // The given value is either dying in this block or live-out.
  if (isLiveOut(value))
    return &block->back();

  // Resolve the last operation (must exist by definition).
  Operation *endOperation = startOperation;
  for (Operation *useOp : value.getUsers()) {
    // Find the associated operation in the current block (if any).
    useOp = block->findAncestorOpInBlock(*useOp);
    // Check whether the use is in our block and after the current end
    // operation.
    if (useOp && endOperation->isBeforeInBlock(useOp))
      endOperation = useOp;
  }
  return endOperation;
}

/// Return the values that are currently live as of the given operation.
LivenessBlockInfo::ValueSetT
LivenessBlockInfo::currentlyLiveValues(Operation *op) const {
  ValueSetT liveSet;

  // Given a value, check which ops are within its live range. For each of
  // those ops, add the value to the set of live values as-of that op.
  auto addValueToCurrentlyLiveSets = [&](Value value) {
    // Determine the live range of this value inside this block.
    Operation *startOfLiveRange = value.getDefiningOp();
    Operation *endOfLiveRange = nullptr;
    // If it's a live in or a block argument, then the start is the beginning
    // of the block.
    if (isLiveIn(value) || isa<BlockArgument>(value))
      startOfLiveRange = &block->front();
    else
      startOfLiveRange = block->findAncestorOpInBlock(*startOfLiveRange);

    // If it's a live out, then the end is the back of the block.
    if (isLiveOut(value))
      endOfLiveRange = &block->back();

    // We must have at least a startOfLiveRange at this point. Given this, we
    // can use the existing getEndOperation to find the end of the live range.
    if (startOfLiveRange && !endOfLiveRange)
      endOfLiveRange = getEndOperation(value, startOfLiveRange);

    assert(endOfLiveRange && "Must have endOfLiveRange at this point!");
    // If this op is within the live range, insert the value into the set.
    if (!(op->isBeforeInBlock(startOfLiveRange) ||
          endOfLiveRange->isBeforeInBlock(op)))
      liveSet.insert(value);
  };

  // Handle block arguments if any.
  for (Value arg : block->getArguments())
    addValueToCurrentlyLiveSets(arg);

  // Handle live-ins. Between the live ins and all the op results that gives us
  // every value in the block.
  for (Value in : inValues)
    addValueToCurrentlyLiveSets(in);

  // Now walk the block and handle all values used in the block and values
  // defined by the block.
  for (Operation &walkOp :
       llvm::make_range(block->begin(), ++op->getIterator()))
    for (auto result : walkOp.getResults())
      addValueToCurrentlyLiveSets(result);

  return liveSet;
}
