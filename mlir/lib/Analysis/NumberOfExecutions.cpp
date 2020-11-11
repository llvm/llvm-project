//===- NumberOfExecutions.cpp - Number of executions analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the number of executions analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/NumberOfExecutions.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "number-of-executions-analysis"

using namespace mlir;

//===----------------------------------------------------------------------===//
// NumberOfExecutions
//===----------------------------------------------------------------------===//

/// Computes blocks number of executions information for the given region.
static void computeRegionBlockNumberOfExecutions(
    Region &region, DenseMap<Block *, BlockNumberOfExecutionsInfo> &blockInfo) {
  Operation *parentOp = region.getParentOp();
  int regionId = region.getRegionNumber();

  auto regionKindInterface = dyn_cast<RegionKindInterface>(parentOp);
  bool isGraphRegion =
      regionKindInterface &&
      regionKindInterface.getRegionKind(regionId) == RegionKind::Graph;

  // CFG analysis does not make sense for Graph regions, set the number of
  // executions for all blocks as unknown.
  if (isGraphRegion) {
    for (Block &block : region)
      blockInfo.insert({&block, {&block, None, None}});
    return;
  }

  // Number of region invocations for all attached regions.
  SmallVector<int64_t, 4> numRegionsInvocations;

  // Query RegionBranchOpInterface interface if it is available.
  if (auto regionInterface = dyn_cast<RegionBranchOpInterface>(parentOp)) {
    SmallVector<Attribute, 4> operands(parentOp->getNumOperands());
    for (auto operandIt : llvm::enumerate(parentOp->getOperands()))
      matchPattern(operandIt.value(), m_Constant(&operands[operandIt.index()]));

    regionInterface.getNumRegionInvocations(operands, numRegionsInvocations);
  }

  // Number of region invocations *each time* parent operation is invoked.
  Optional<int64_t> numRegionInvocations;

  if (!numRegionsInvocations.empty() &&
      numRegionsInvocations[regionId] != kUnknownNumRegionInvocations) {
    numRegionInvocations = numRegionsInvocations[regionId];
  }

  // DFS traversal looking for loops in the CFG.
  llvm::SmallSet<Block *, 4> loopStart;

  llvm::unique_function<void(Block *, llvm::SmallSet<Block *, 4> &)> dfs =
      [&](Block *block, llvm::SmallSet<Block *, 4> &visited) {
        // Found a loop in the CFG that starts at the `block`.
        if (visited.contains(block)) {
          loopStart.insert(block);
          return;
        }

        // Continue DFS traversal.
        visited.insert(block);
        for (Block *successor : block->getSuccessors())
          dfs(successor, visited);
        visited.erase(block);
      };

  llvm::SmallSet<Block *, 4> visited;
  dfs(&region.front(), visited);

  // Start from the entry block and follow only blocks with single succesor.
  Block *block = &region.front();
  while (block && !loopStart.contains(block)) {
    // Block will be executed exactly once.
    blockInfo.insert(
        {block, BlockNumberOfExecutionsInfo(block, numRegionInvocations,
                                            /*numberOfBlockExecutions=*/1)});

    // We reached the exit block or block with multiple successors.
    if (block->getNumSuccessors() != 1)
      break;

    // Continue traversal.
    block = block->getSuccessor(0);
  }

  // For all blocks that we did not visit set the executions number to unknown.
  for (Block &block : region)
    if (blockInfo.count(&block) == 0)
      blockInfo.insert({&block, BlockNumberOfExecutionsInfo(
                                    &block, numRegionInvocations,
                                    /*numberOfBlockExecutions=*/None)});
}

/// Creates a new NumberOfExecutions analysis that computes how many times a
/// block within a region is executed for all associated regions.
NumberOfExecutions::NumberOfExecutions(Operation *op) : operation(op) {
  operation->walk([&](Region *region) {
    computeRegionBlockNumberOfExecutions(*region, blockNumbersOfExecution);
  });
}

Optional<int64_t>
NumberOfExecutions::getNumberOfExecutions(Operation *op,
                                          Region *perEntryOfThisRegion) const {
  // Assuming that all operations complete in a finite amount of time (do not
  // abort and do not go into the infinite loop), the number of operation
  // executions is equal to the number of block executions that contains the
  // operation.
  return getNumberOfExecutions(op->getBlock(), perEntryOfThisRegion);
}

Optional<int64_t>
NumberOfExecutions::getNumberOfExecutions(Block *block,
                                          Region *perEntryOfThisRegion) const {
  // Return None if the given `block` does not lie inside the
  // `perEntryOfThisRegion` region.
  if (!perEntryOfThisRegion->findAncestorBlockInRegion(*block))
    return None;

  // Find the block information for the given `block.
  auto blockIt = blockNumbersOfExecution.find(block);
  if (blockIt == blockNumbersOfExecution.end())
    return None;
  const auto &blockInfo = blockIt->getSecond();

  // Override the number of region invocations with `1` if the
  // `perEntryOfThisRegion` region owns the block.
  auto getNumberOfExecutions = [&](const BlockNumberOfExecutionsInfo &info) {
    if (info.getBlock()->getParent() == perEntryOfThisRegion)
      return info.getNumberOfExecutions(/*numberOfRegionInvocations=*/1);
    return info.getNumberOfExecutions();
  };

  // Immediately return None if we do not know the block number of executions.
  auto blockExecutions = getNumberOfExecutions(blockInfo);
  if (!blockExecutions.hasValue())
    return None;

  // Follow parent operations until we reach the operations that owns the
  // `perEntryOfThisRegion`.
  int64_t numberOfExecutions = *blockExecutions;
  Operation *parentOp = block->getParentOp();

  while (parentOp != perEntryOfThisRegion->getParentOp()) {
    // Find how many times will be executed the block that owns the parent
    // operation.
    Block *parentBlock = parentOp->getBlock();

    auto parentBlockIt = blockNumbersOfExecution.find(parentBlock);
    if (parentBlockIt == blockNumbersOfExecution.end())
      return None;
    const auto &parentBlockInfo = parentBlockIt->getSecond();
    auto parentBlockExecutions = getNumberOfExecutions(parentBlockInfo);

    // We stumbled upon an operation with unknown number of executions.
    if (!parentBlockExecutions.hasValue())
      return None;

    // Number of block executions is a product of all parent blocks executions.
    numberOfExecutions *= *parentBlockExecutions;
    parentOp = parentOp->getParentOp();

    assert(parentOp != nullptr);
  }

  return numberOfExecutions;
}

void NumberOfExecutions::printBlockExecutions(
    raw_ostream &os, Region *perEntryOfThisRegion) const {
  unsigned blockId = 0;

  operation->walk([&](Block *block) {
    llvm::errs() << "Block: " << blockId++ << "\n";
    llvm::errs() << "Number of executions: ";
    if (auto n = getNumberOfExecutions(block, perEntryOfThisRegion))
      llvm::errs() << *n << "\n";
    else
      llvm::errs() << "<unknown>\n";
  });
}

void NumberOfExecutions::printOperationExecutions(
    raw_ostream &os, Region *perEntryOfThisRegion) const {
  operation->walk([&](Block *block) {
    block->walk([&](Operation *operation) {
      // Skip the operation that was used to build the analysis.
      if (operation == this->operation)
        return;

      llvm::errs() << "Operation: " << operation->getName() << "\n";
      llvm::errs() << "Number of executions: ";
      if (auto n = getNumberOfExecutions(operation, perEntryOfThisRegion))
        llvm::errs() << *n << "\n";
      else
        llvm::errs() << "<unknown>\n";
    });
  });
}

//===----------------------------------------------------------------------===//
// BlockNumberOfExecutionsInfo
//===----------------------------------------------------------------------===//

BlockNumberOfExecutionsInfo::BlockNumberOfExecutionsInfo(
    Block *block, Optional<int64_t> numberOfRegionInvocations,
    Optional<int64_t> numberOfBlockExecutions)
    : block(block), numberOfRegionInvocations(numberOfRegionInvocations),
      numberOfBlockExecutions(numberOfBlockExecutions) {}

Optional<int64_t> BlockNumberOfExecutionsInfo::getNumberOfExecutions() const {
  if (numberOfRegionInvocations && numberOfBlockExecutions)
    return *numberOfRegionInvocations * *numberOfBlockExecutions;
  return None;
}

Optional<int64_t> BlockNumberOfExecutionsInfo::getNumberOfExecutions(
    int64_t numberOfRegionInvocations) const {
  if (numberOfBlockExecutions)
    return numberOfRegionInvocations * *numberOfBlockExecutions;
  return None;
}
