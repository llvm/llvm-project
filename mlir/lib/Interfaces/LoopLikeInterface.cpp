//===- LoopLikeInterface.cpp - Loop-like operations in MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LoopLikeInterface.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

/// Include the definitions of the loop-like interfaces.
#include "mlir/Interfaces/LoopLikeInterface.cpp.inc"

bool LoopLikeOpInterface::blockIsInLoop(Block *block) {
  Operation *parent = block->getParentOp();

  // The block could be inside a loop-like operation
  if (isa<LoopLikeOpInterface>(parent) ||
      parent->getParentOfType<LoopLikeOpInterface>())
    return true;

  // This block might be nested inside another block, which is in a loop
  if (!isa<FunctionOpInterface>(parent))
    if (mlir::Block *parentBlock = parent->getBlock())
      if (blockIsInLoop(parentBlock))
        return true;

  // Or the block could be inside a control flow graph loop:
  // A block is in a control flow graph loop if it can reach itself in a graph
  // traversal
  DenseSet<Block *> visited;
  SmallVector<Block *> stack;
  stack.push_back(block);
  while (!stack.empty()) {
    Block *current = stack.pop_back_val();
    auto [it, inserted] = visited.insert(current);
    if (!inserted) {
      // loop detected
      if (current == block)
        return true;
      continue;
    }

    stack.reserve(stack.size() + current->getNumSuccessors());
    for (Block *successor : current->getSuccessors())
      stack.push_back(successor);
  }
  return false;
}

LogicalResult detail::verifyLoopLikeOpInterface(Operation *op) {
  // Note: These invariants are also verified by the RegionBranchOpInterface,
  // but the LoopLikeOpInterface provides better error messages.
  auto loopLikeOp = cast<LoopLikeOpInterface>(op);

  // Verify number of inits/iter_args/yielded values/loop results.
  if (loopLikeOp.getInits().size() != loopLikeOp.getRegionIterArgs().size())
    return op->emitOpError("different number of inits and region iter_args: ")
           << loopLikeOp.getInits().size()
           << " != " << loopLikeOp.getRegionIterArgs().size();
  if (!loopLikeOp.getYieldedValues().empty() &&
      loopLikeOp.getRegionIterArgs().size() !=
          loopLikeOp.getYieldedValues().size())
    return op->emitOpError(
               "different number of region iter_args and yielded values: ")
           << loopLikeOp.getRegionIterArgs().size()
           << " != " << loopLikeOp.getYieldedValues().size();
  if (loopLikeOp.getLoopResults() && loopLikeOp.getLoopResults()->size() !=
                                         loopLikeOp.getRegionIterArgs().size())
    return op->emitOpError(
               "different number of loop results and region iter_args: ")
           << loopLikeOp.getLoopResults()->size()
           << " != " << loopLikeOp.getRegionIterArgs().size();

  // Verify types of inits/iter_args/yielded values/loop results.
  int64_t i = 0;
  auto yieldedValues = loopLikeOp.getYieldedValues();
  for (const auto [index, init, regionIterArg] :
       llvm::enumerate(loopLikeOp.getInits(), loopLikeOp.getRegionIterArgs())) {
    if (init.getType() != regionIterArg.getType())
      return op->emitOpError(std::to_string(index))
             << "-th init and " << index
             << "-th region iter_arg have different type: " << init.getType()
             << " != " << regionIterArg.getType();
    if (!yieldedValues.empty()) {
      if (regionIterArg.getType() != yieldedValues[index].getType())
        return op->emitOpError(std::to_string(index))
               << "-th region iter_arg and " << index
               << "-th yielded value have different type: "
               << regionIterArg.getType()
               << " != " << yieldedValues[index].getType();
    }
    ++i;
  }
  i = 0;
  if (loopLikeOp.getLoopResults()) {
    for (const auto it : llvm::zip_equal(loopLikeOp.getRegionIterArgs(),
                                         *loopLikeOp.getLoopResults())) {
      if (std::get<0>(it).getType() != std::get<1>(it).getType())
        return op->emitOpError(std::to_string(i))
               << "-th region iter_arg and " << i
               << "-th loop result have different type: "
               << std::get<0>(it).getType()
               << " != " << std::get<1>(it).getType();
    }
    ++i;
  }

  return success();
}

LoopLikeOpInterface mlir::createFused(LoopLikeOpInterface target,
                                      LoopLikeOpInterface source,
                                      RewriterBase &rewriter,
                                      NewYieldValuesFn newYieldValuesFn,
                                      FuseTerminatorFn fuseTerminatorFn) {
  auto targetIterArgs = target.getRegionIterArgs();
  std::optional<SmallVector<Value>> targetInductionVar =
      target.getLoopInductionVars();
  SmallVector<Value> targetYieldOperands(target.getYieldedValues());
  auto sourceIterArgs = source.getRegionIterArgs();
  std::optional<SmallVector<Value>> sourceInductionVar =
      *source.getLoopInductionVars();
  SmallVector<Value> sourceYieldOperands(source.getYieldedValues());
  auto sourceRegion = source.getLoopRegions().front();

  FailureOr<LoopLikeOpInterface> maybeFusedLoop =
      target.replaceWithAdditionalYields(rewriter, source.getInits(),
                                         /*replaceInitOperandUsesInLoop=*/false,
                                         newYieldValuesFn);
  if (failed(maybeFusedLoop))
    llvm_unreachable("failed to replace loop");
  LoopLikeOpInterface fusedLoop = *maybeFusedLoop;
  // Since the target op is rewritten at the original's location, we move it to
  // the soure op's location.
  rewriter.moveOpBefore(fusedLoop, source);

  // Map control operands.
  IRMapping mapping;
  std::optional<SmallVector<Value>> fusedInductionVar =
      fusedLoop.getLoopInductionVars();
  if (fusedInductionVar) {
    if (!targetInductionVar || !sourceInductionVar)
      llvm_unreachable(
          "expected target and source loops to have induction vars");
    mapping.map(*targetInductionVar, *fusedInductionVar);
    mapping.map(*sourceInductionVar, *fusedInductionVar);
  }
  mapping.map(targetIterArgs,
              fusedLoop.getRegionIterArgs().take_front(targetIterArgs.size()));
  mapping.map(targetYieldOperands,
              fusedLoop.getYieldedValues().take_front(targetIterArgs.size()));
  mapping.map(sourceIterArgs,
              fusedLoop.getRegionIterArgs().take_back(sourceIterArgs.size()));
  mapping.map(sourceYieldOperands,
              fusedLoop.getYieldedValues().take_back(sourceIterArgs.size()));
  // Append everything except the terminator into the fused operation.
  rewriter.setInsertionPoint(
      fusedLoop.getLoopRegions().front()->front().getTerminator());
  for (Operation &op : sourceRegion->front().without_terminator())
    rewriter.clone(op, mapping);

  // TODO: Replace with corresponding interface method if added
  fuseTerminatorFn(rewriter, source, fusedLoop, mapping);

  return fusedLoop;
}
