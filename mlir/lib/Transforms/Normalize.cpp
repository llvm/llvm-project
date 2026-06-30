//===- Normalize.cpp - Transforms IR into a normal form ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/DebugLog.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "normalize"

namespace {

bool isOutput(Operation *op) {
  if (!op)
    return false;
  return !isMemoryEffectFree(op) || op->hasTrait<OpTrait::IsTerminator>();
}

/// Returns a vector of output ops. An output is a op which
/// has side-effects or is terminator.
SmallVector<Operation *> collectOutputs(Operation *root) {
  SmallVector<Operation *> outputs;
  root->walk([&](Operation *op) {
    if (isOutput(op))
      outputs.push_back(op);
  });
  return outputs;
}

/// The function returns the operation that dominates all other operations in
/// the given list.
Operation *getDominateOp(const SmallVectorImpl<Operation *> &ops,
                         const DominanceInfo &domInfo) {
  if (ops.empty())
    return {};
  Operation *curDomOp = ops.front();
  for (size_t i = 1, e = ops.size(); i < e; ++i) {
    bool dominateA = domInfo.dominates(ops[i], curDomOp);
    if (dominateA) {
      LDBG() << OpWithFlags(ops[i], OpPrintingFlags().skipRegions())
             << "\ndominate\n"
             << OpWithFlags(curDomOp, OpPrintingFlags().skipRegions());
      curDomOp = ops[i];
      continue;
    }
    bool dominateB = domInfo.dominates(curDomOp, ops[i]);
    if (!dominateB) {
      LDBG() << OpWithFlags(ops[i], OpPrintingFlags().skipRegions())
             << "\nand\n"
             << OpWithFlags(curDomOp, OpPrintingFlags().skipRegions())
             << "\ndo not dominate each other";
      return {};
    }
  }
  return curDomOp;
}

/// Move used to its nearest user and recursively perform the same process on
/// the defining operations of its operands.
void reorderOutput(IRRewriter &rewriter, Operation *producer,
                   const DominanceInfo &domInfo) {
  if (!isPure(producer))
    return;
  SmallVector<Operation *> users(producer->getUsers());
  if (Operation *domOp = getDominateOp(users, domInfo)) {
    rewriter.moveOpBefore(producer, domOp);
    for (Value operand : producer->getOperands())
      if (Operation *defineOp = operand.getDefiningOp())
        reorderOutput(rewriter, defineOp, domInfo);
  }
}

/// Reorders ops by walking up the tree from each operand of an output op and
/// reducing the def-use distance. This method assumes that output ops were
/// collected top-down, otherwise the def-use chain may be broken. This method
/// is a wrapper for recursive reorderOutput().
void reorderOutputs(IRRewriter &rewriter,
                    const SmallVectorImpl<Operation *> &outputs,
                    const DominanceInfo &domInfo) {
  SmallPtrSet<Operation *, 16> visited;
  for (Operation *output : outputs) {
    for (Value operand : output->getOperands()) {
      if (Operation *defineOp = operand.getDefiningOp();
          defineOp && !visited.contains(defineOp)) {
        visited.insert(defineOp);
        reorderOutput(rewriter, defineOp, domInfo);
      }
    }
  }
}

struct NormalizePass : public impl::NormalizePassBase<NormalizePass> {
  using impl::NormalizePassBase<NormalizePass>::NormalizePassBase;
  void runOnOperation() override;
};
} // namespace

void NormalizePass::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  IRRewriter rewriter(&getContext());
  SmallVector<Operation *> outputs = collectOutputs(getOperation());
  reorderOutputs(rewriter, outputs, domInfo);
  // Since we only changed the positions of the operations, DominanceInfo and
  // PostDominanceInfo are marked as preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}
