//===- Hoisting.cpp - Linalg hoisting transformations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with hoisting invariant operations
// in the context of Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-hoisting"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

using llvm::dbgs;

void mlir::linalg::hoistViewAllocOps(FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&changed](Operation *op) {
      if (!isa<AllocOp>(op) && !isa<AllocaOp>(op) && !isa<DeallocOp>(op))
        return;

      LLVM_DEBUG(DBGS() << "Candidate for hoisting: " << *op << "\n");
      auto loop = dyn_cast<scf::ForOp>(op->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *op->getParentOp() << "\n");

      // Only hoist out of immediately enclosing scf::ForOp.
      if (!loop)
        return;

      // If any operand is defined inside the loop don't hoist.
      if (llvm::any_of(op->getOperands(), [&](Value v) {
            return !loop.isDefinedOutsideOfLoop(v);
          }))
        return;

      LLVM_DEBUG(DBGS() << "All operands defined outside \n");

      // If alloc has other uses than ViewLikeOp and DeallocOp don't hoist.
      Value v;
      if (op->getNumResults() > 0) {
        assert(op->getNumResults() == 1 && "Unexpected multi-result alloc");
        v = op->getResult(0);
      }
      if (v && !llvm::all_of(v.getUses(), [&](OpOperand &operand) {
            return isa<ViewLikeOpInterface>(operand.getOwner()) ||
                   isa<DeallocOp>(operand.getOwner());
          })) {
        LLVM_DEBUG(DBGS() << "Found non view-like or dealloc use: bail\n");
        return;
      }

      // Move AllocOp before the loop.
      if (isa<AllocOp>(op) || isa<AllocaOp>(op))
        loop.moveOutOfLoop({op});
      else // Move DeallocOp outside of the loop.
        op->moveAfter(loop);
      changed = true;
    });
  }
}
