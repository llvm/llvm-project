//===- ReplaceOperands.h - Replacing Operands Reduction Pattern -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Reducer/Patterns/ReplaceOperands.h"
#include <cstddef>

using namespace mlir;

OperandReductionNode::OperandReductionNode(Operation *reductionOp,
                                           ArrayRef<Range> ranges)
    : startRanges(ranges), discardRanges(ranges) {

  ModuleOp moduleOp = reductionOp->getParentOfType<ModuleOp>();
  module = cast<ModuleOp>(moduleOp->clone(mapping));
  op = mapping.lookup(reductionOp);
  usedValues = DenseSet<Value>(op->operand_begin(), op->operand_end());

  size_t rangeIndex = 0;
  for (const auto &[index, operand] : enumerate(op->getOpOperands())) {
    if (rangeIndex < discardRanges.size() &&
        index == discardRanges[rangeIndex].second)
      ++rangeIndex;
    if (rangeIndex == discardRanges.size() ||
        index < discardRanges[rangeIndex].first)
      operandMap[operand.get().getType()].push_back(&operand);
  }
}

LogicalResult
ReplaceOperandsPattern::matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {

  Block *block = op->getBlock();
  // If Operation has no parent block, then return failure
  if (!block)
    return failure();

  // If operation has no operands, we don't have anything to do
  if (op->getNumOperands() == 0)
    return failure();

  // candidateValues stores suitable replacements per each type
  DenseMap<Type, SmallVector<Value, 0>> candidateValues;

  // Add block arguments first (they come first in program order)
  for (BlockArgument arg : block->getArguments())
    candidateValues[arg.getType()].push_back(arg);

  // Walk operations in the block to find remaining types
  for (auto &blockOp : block->getOperations()) {
    // Stop before reaching the current operation
    if (&blockOp == op)
      break;

    for (Value result : blockOp.getResults())
      candidateValues[result.getType()].push_back(result);
  }

  OperandReductionNode node(op, {{0, 0}});

  auto types = llvm::filter_to_vector(node.getNeededTypes(), [&](Type type) {
    return candidateValues.contains(type);
  });
  auto values = llvm::map_to_vector(types, [&](Type type) -> ArrayRef<Value> {
    return candidateValues.find(type)->second;
  });

  size_t typesLen = types.size();
  if (typesLen == 0)
    return failure();

  // We'll use a gray-code like algorithm on an arbitrary radix to iterate over
  // different combinations of values
  SmallVector<size_t, 0> idx(typesLen, 0);
  SmallVector<int8_t, 0> dir(typesLen, 1);

  bool replacementFound = false, changed = false;
  auto applyIndex = [&](size_t index) {
    for (auto *operand : node.getOperandForType(types[index])) {
      auto value = node.getMappedValue(values[index][idx[index]]);
      if (operand->get() == value)
        continue;
      operand->set(value);
      changed = true;
    }
  };

  // The caller guarantees the input module is interesting, so we skip
  // testing the initial configuration and go straight to the first
  // Gray-code neighbor.
  for (auto typeIndex : llvm::seq(typesLen))
    applyIndex(typeIndex);

  while (true) {

    if (changed) {
      auto [isInteresting, size] = tester.isInteresting(node.getModule());
      if (isInteresting == Tester::Interestingness::True) {
        replacementFound = true;
        break;
      }
    }

    ptrdiff_t next = 0;
    size_t index = 0;
    changed = false;
    bool exhausted = true;

    while (index < typesLen) {
      next = (ptrdiff_t)idx[index] + dir[index];
      if ((next >= 0) && ((size_t)next < values[index].size())) {
        idx[index] = next;
        applyIndex(index);
        exhausted = false;
        break;
      }
      dir[index] = -dir[index];
      ++index;
    }
    if (exhausted)
      break;
  }

  if (replacementFound) {
    DenseMap<Type, unsigned> typeIdx;
    typeIdx.reserve(typesLen);

    for (auto [k, v] : llvm::zip(types, idx))
      typeIdx[k] = v;

    // Instead of replacing operands inplace, we'll replace the operation
    // completely. If we don't replace the operation with a new one,  the greedy
    // driver will add the operation to the worklist again.
    auto *newOp = rewriter.clone(*op);
    for (auto &operand : newOp->getOpOperands()) {
      auto type = operand.get().getType();
      auto it = typeIdx.find(type);
      if (it != typeIdx.end())
        operand.set(candidateValues.find(type)->second[it->second]);
    }
    rewriter.replaceOp(op, newOp);
    return success();
  }

  return failure();
}
