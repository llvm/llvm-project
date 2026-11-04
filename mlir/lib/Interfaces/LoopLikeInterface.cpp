//===- LoopLikeInterface.cpp - Loop-like operations in MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LoopLikeInterface.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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
  // If the op also implements RegionBranchOpInterface, type compatibility is
  // already verified by that interface's verifier (which also provides an
  // overridable areTypesCompatible hook), so skip the check here.
  if (!isa<RegionBranchOpInterface>(op)) {
    auto yieldedValues = loopLikeOp.getYieldedValues();
    for (const auto [index, init, regionIterArg] : llvm::enumerate(
             loopLikeOp.getInits(), loopLikeOp.getRegionIterArgs())) {
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
    }
    if (loopLikeOp.getLoopResults()) {
      for (const auto [index, regionIterArg, loopResult] : llvm::enumerate(
               loopLikeOp.getRegionIterArgs(), *loopLikeOp.getLoopResults())) {
        if (regionIterArg.getType() != loopResult.getType())
          return op->emitOpError(std::to_string(index))
                 << "-th region iter_arg and " << index
                 << "-th loop result have different type: "
                 << regionIterArg.getType() << " != " << loopResult.getType();
      }
    }
  }

  // Verify that all induction variables have valid types.
  auto inductionVars = loopLikeOp.getLoopInductionVars();
  if (inductionVars.has_value()) {
    for (auto [index, inductionVar] : llvm::enumerate(*inductionVars)) {
      if (!loopLikeOp.isValidInductionVarType(inductionVar.getType()))
        return op->emitOpError(std::to_string(index))
               << "-th induction variable has invalid type: "
               << inductionVar.getType();
    }
  }

  return success();
}
