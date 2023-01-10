//===- LoopLikeInterface.cpp - Loop-like operations in MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  if (!isa<FunctionOpInterface>(parent)) {
    if (blockIsInLoop(parent->getBlock())) {
      return true;
    }
  }

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
