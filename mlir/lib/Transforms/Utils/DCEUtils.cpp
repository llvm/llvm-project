//===- DCEUtils.cpp - Dead Code Elimination ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation implements method for eliminating dead code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DCEUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

void mlir::deadCodeElimination(RewriterBase &rewriter, Operation *target) {
  // Maintain a worklist of potentially dead ops.
  mlir::SetVector<Operation *> worklist;

  // Helper function that adds all defining ops of used values (operands and
  // operands of nested ops).
  auto addDefiningOpsToWorklist = [&](Operation *op) {
    op->walk([&](Operation *op) {
      for (Value v : op->getOperands())
        if (Operation *defOp = v.getDefiningOp())
          if (target->isProperAncestor(defOp))
            worklist.insert(defOp);
    });
  };

  // Helper function that erases an op.
  auto eraseOp = [&](Operation *op) {
    // Remove op and nested ops from the worklist.
    op->walk([&](Operation *op) {
      const auto *it = llvm::find(worklist, op);
      if (it != worklist.end())
        worklist.erase(it);
    });
    rewriter.eraseOp(op);
  };

  // Initial walk over the IR.
  target->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (op != target && isOpTriviallyDead(op)) {
      addDefiningOpsToWorklist(op);
      eraseOp(op);
    }
  });

  // Erase all ops that have become dead.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    addDefiningOpsToWorklist(op);
    eraseOp(op);
  }
}
