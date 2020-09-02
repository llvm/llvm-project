//===- AffineLoopInvariantCodeMotion.cpp - Code to perform loop fusion-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop invariant code motion.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "licm"

using namespace mlir;

namespace {

/// Loop invariant code motion (LICM) pass.
/// TODO: The pass is missing zero-trip tests.
/// TODO: Check for the presence of side effects before hoisting.
/// TODO: This code should be removed once the new LICM pass can handle its
///       uses.
struct LoopInvariantCodeMotion
    : public AffineLoopInvariantCodeMotionBase<LoopInvariantCodeMotion> {
  void runOnFunction() override;
  void runOnAffineForOp(AffineForOp forOp);
};
} // end anonymous namespace

static bool
checkInvarianceOfNestedIfOps(Operation *op, Value indVar,
                             SmallPtrSetImpl<Operation *> &definedOps,
                             SmallPtrSetImpl<Operation *> &opsToHoist);
static bool isOpLoopInvariant(Operation &op, Value indVar,
                              SmallPtrSetImpl<Operation *> &definedOps,
                              SmallPtrSetImpl<Operation *> &opsToHoist);

static bool
areAllOpsInTheBlockListInvariant(Region &blockList, Value indVar,
                                 SmallPtrSetImpl<Operation *> &definedOps,
                                 SmallPtrSetImpl<Operation *> &opsToHoist);

static bool isMemRefDereferencingOp(Operation &op) {
  // TODO: Support DMA Ops.
  return isa<AffineReadOpInterface, AffineWriteOpInterface>(op);
}

// Returns true if the individual op is loop invariant.
bool isOpLoopInvariant(Operation &op, Value indVar,
                       SmallPtrSetImpl<Operation *> &definedOps,
                       SmallPtrSetImpl<Operation *> &opsToHoist) {
  LLVM_DEBUG(llvm::dbgs() << "iterating on op: " << op;);

  if (isa<AffineIfOp>(op)) {
    if (!checkInvarianceOfNestedIfOps(&op, indVar, definedOps, opsToHoist)) {
      return false;
    }
  } else if (isa<AffineForOp>(op)) {
    // If the body of a predicated region has a for loop, we don't hoist the
    // 'affine.if'.
    return false;
  } else if (isa<AffineDmaStartOp, AffineDmaWaitOp>(op)) {
    // TODO: Support DMA ops.
    return false;
  } else if (!isa<ConstantOp>(op)) {
    // Register op in the set of ops defined inside the loop. This set is used
    // to prevent hoisting ops that depend on other ops defined inside the loop
    // which are themselves not being hoisted.
    definedOps.insert(&op);

    if (isMemRefDereferencingOp(op)) {
      Value memref = isa<AffineReadOpInterface>(op)
                         ? cast<AffineReadOpInterface>(op).getMemRef()
                         : cast<AffineWriteOpInterface>(op).getMemRef();
      for (auto *user : memref.getUsers()) {
        // If this memref has a user that is a DMA, give up because these
        // operations write to this memref.
        if (isa<AffineDmaStartOp, AffineDmaWaitOp>(op)) {
          return false;
        }
        // If the memref used by the load/store is used in a store elsewhere in
        // the loop nest, we do not hoist. Similarly, if the memref used in a
        // load is also being stored too, we do not hoist the load.
        if (isa<AffineWriteOpInterface>(user) ||
            (isa<AffineReadOpInterface>(user) &&
             isa<AffineWriteOpInterface>(op))) {
          if (&op != user) {
            SmallVector<AffineForOp, 8> userIVs;
            getLoopIVs(*user, &userIVs);
            // Check that userIVs don't contain the for loop around the op.
            if (llvm::is_contained(userIVs, getForInductionVarOwner(indVar))) {
              return false;
            }
          }
        }
      }
    }

    if (op.getNumOperands() == 0 && !isa<AffineYieldOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "\nNon-constant op with 0 operands\n");
      return false;
    }
    for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
      auto *operandSrc = op.getOperand(i).getDefiningOp();

      LLVM_DEBUG(
          op.getOperand(i).print(llvm::dbgs() << "\nIterating on operand\n"));

      // If the loop IV is the operand, this op isn't loop invariant.
      if (indVar == op.getOperand(i)) {
        LLVM_DEBUG(llvm::dbgs() << "\nLoop IV is the operand\n");
        return false;
      }

      if (operandSrc != nullptr) {
        LLVM_DEBUG(llvm::dbgs()
                   << *operandSrc << "\nIterating on operand src\n");

        // If the value was defined in the loop (outside of the
        // if/else region), and that operation itself wasn't meant to
        // be hoisted, then mark this operation loop dependent.
        if (definedOps.count(operandSrc) && opsToHoist.count(operandSrc) == 0) {
          return false;
        }
      }
    }
  }

  // If no operand was loop variant, mark this op for motion.
  opsToHoist.insert(&op);
  return true;
}

// Checks if all ops in a region (i.e. list of blocks) are loop invariant.
bool areAllOpsInTheBlockListInvariant(
    Region &blockList, Value indVar, SmallPtrSetImpl<Operation *> &definedOps,
    SmallPtrSetImpl<Operation *> &opsToHoist) {

  for (auto &b : blockList) {
    for (auto &op : b) {
      if (!isOpLoopInvariant(op, indVar, definedOps, opsToHoist)) {
        return false;
      }
    }
  }

  return true;
}

// Returns true if the affine.if op can be hoisted.
bool checkInvarianceOfNestedIfOps(Operation *op, Value indVar,
                                  SmallPtrSetImpl<Operation *> &definedOps,
                                  SmallPtrSetImpl<Operation *> &opsToHoist) {
  assert(isa<AffineIfOp>(op));
  auto ifOp = cast<AffineIfOp>(op);

  if (!areAllOpsInTheBlockListInvariant(ifOp.thenRegion(), indVar, definedOps,
                                        opsToHoist)) {
    return false;
  }

  if (!areAllOpsInTheBlockListInvariant(ifOp.elseRegion(), indVar, definedOps,
                                        opsToHoist)) {
    return false;
  }

  return true;
}

void LoopInvariantCodeMotion::runOnAffineForOp(AffineForOp forOp) {
  auto *loopBody = forOp.getBody();
  auto indVar = forOp.getInductionVar();

  SmallPtrSet<Operation *, 8> definedOps;
  // This is the place where hoisted instructions would reside.
  OpBuilder b(forOp.getOperation());

  SmallPtrSet<Operation *, 8> opsToHoist;
  SmallVector<Operation *, 8> opsToMove;

  for (auto &op : *loopBody) {
    // We don't hoist for loops.
    if (!isa<AffineForOp>(op)) {
      if (!isa<AffineYieldOp>(op)) {
        if (isOpLoopInvariant(op, indVar, definedOps, opsToHoist)) {
          opsToMove.push_back(&op);
        }
      }
    }
  }

  // For all instructions that we found to be invariant, place sequentially
  // right before the for loop.
  for (auto *op : opsToMove) {
    op->moveBefore(forOp);
  }

  LLVM_DEBUG(forOp.getOperation()->print(llvm::dbgs() << "Modified loop\n"));
}

void LoopInvariantCodeMotion::runOnFunction() {
  // Walk through all loops in a function in innermost-loop-first order.  This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getFunction().walk([&](AffineForOp op) {
    LLVM_DEBUG(op.getOperation()->print(llvm::dbgs() << "\nOriginal loop\n"));
    runOnAffineForOp(op);
  });
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createAffineLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}
