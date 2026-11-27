//===- AnalysisManager.cpp - Analysis management --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AnalysisManager.h"

using namespace mlir;

AnalysisInstrumentation::~AnalysisInstrumentation() = default;

/// Get an analysis manager for the given operation, which must be a proper
/// descendant of the current operation represented by this analysis manager.
AnalysisManager AnalysisManager::nest(Operation *op) {
  Operation *currentOp = impl->getOperation();
  assert(currentOp->isProperAncestor(op) &&
         "expected valid descendant operation");

  // Check for the base case where the provided operation is immediately nested.
  if (currentOp == op->getParentOp())
    return nestImmediate(op);

  // Otherwise, we need to collect all ancestors up to the current operation.
  SmallVector<Operation *, 4> opAncestors;
  do {
    opAncestors.push_back(op);
    op = op->getParentOp();
  } while (op != currentOp);

  AnalysisManager result = *this;
  for (Operation *op : llvm::reverse(opAncestors))
    result = result.nestImmediate(op);
  return result;
}

/// Get an analysis manager for the given immediately nested child operation.
AnalysisManager AnalysisManager::nestImmediate(Operation *op) {
  assert(impl->getOperation() == op->getParentOp() &&
         "expected immediate child operation");

  auto [it, inserted] = impl->childAnalyses.try_emplace(op);
  if (inserted)
    it->second = std::make_unique<detail::NestedAnalysisMap>(op, impl);
  return {it->second.get()};
}

/// Invalidate any non preserved analyses.
void detail::NestedAnalysisMap::invalidate(
    const detail::PreservedAnalyses &pa) {
  // If all analyses were preserved, then there is nothing to do here.
  if (pa.isAll())
    return;

  // Invalidate the analyses for the current operation directly.
  analyses.invalidate(pa);

  // If no analyses were preserved, then just simply clear out the child
  // analysis results.
  if (pa.isNone()) {
    childAnalyses.clear();
    return;
  }

  // Otherwise, invalidate each child analysis map.
  SmallVector<NestedAnalysisMap *, 8> mapsToInvalidate(1, this);
  while (!mapsToInvalidate.empty()) {
    auto *map = mapsToInvalidate.pop_back_val();
    for (auto &analysisPair : map->childAnalyses) {
      analysisPair.second->invalidate(pa);
      if (!analysisPair.second->childAnalyses.empty())
        mapsToInvalidate.push_back(analysisPair.second.get());
    }
  }
}
