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
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
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
  void runOnOperation() override;
  void runOnAffineForOp(AffineForOp forOp);
};
} // namespace

static bool
checkInvarianceOfNestedIfOps(Operation *op, Value indVar, ValueRange iterArgs,
                             SmallPtrSetImpl<Operation *> &opsWithUsers,
                             SmallPtrSetImpl<Operation *> &opsToHoist);
static bool isOpLoopInvariant(Operation &op, Value indVar, ValueRange iterArgs,
                              SmallPtrSetImpl<Operation *> &opsWithUsers,
                              SmallPtrSetImpl<Operation *> &opsToHoist);

static bool
areAllOpsInTheBlockListInvariant(Region &blockList, Value indVar,
                                 ValueRange iterArgs,
                                 SmallPtrSetImpl<Operation *> &opsWithUsers,
                                 SmallPtrSetImpl<Operation *> &opsToHoist);

// Returns true if the individual op is loop invariant.
bool isOpLoopInvariant(Operation &op, Value indVar, ValueRange iterArgs,
                       SmallPtrSetImpl<Operation *> &opsWithUsers,
                       SmallPtrSetImpl<Operation *> &opsToHoist) {
  LLVM_DEBUG(llvm::dbgs() << "iterating on op: " << op;);

  if (isa<AffineIfOp>(op)) {
    if (!checkInvarianceOfNestedIfOps(&op, indVar, iterArgs, opsWithUsers,
                                      opsToHoist)) {
      return false;
    }
  } else if (auto forOp = dyn_cast<AffineForOp>(op)) {
    if (!areAllOpsInTheBlockListInvariant(forOp.getLoopBody(), indVar, iterArgs,
                                          opsWithUsers, opsToHoist)) {
      return false;
    }
  } else if (isa<AffineDmaStartOp, AffineDmaWaitOp>(op)) {
    // TODO: Support DMA ops.
    return false;
  } else if (!matchPattern(&op, m_Constant())) {
    // Register op in the set of ops that have users.
    opsWithUsers.insert(&op);
    if (isa<AffineMapAccessInterface>(op)) {
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
  }

  // Check operands.
  for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
    auto *operandSrc = op.getOperand(i).getDefiningOp();

    LLVM_DEBUG(
        op.getOperand(i).print(llvm::dbgs() << "\nIterating on operand\n"));

    // If the loop IV is the operand, this op isn't loop invariant.
    if (indVar == op.getOperand(i)) {
      LLVM_DEBUG(llvm::dbgs() << "\nLoop IV is the operand\n");
      return false;
    }

    // If the one of the iter_args is the operand, this op isn't loop invariant.
    if (llvm::is_contained(iterArgs, op.getOperand(i))) {
      LLVM_DEBUG(llvm::dbgs() << "\nOne of the iter_args is the operand\n");
      return false;
    }

    if (operandSrc != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << *operandSrc << "\nIterating on operand src\n");

      // If the value was defined in the loop (outside of the
      // if/else region), and that operation itself wasn't meant to
      // be hoisted, then mark this operation loop dependent.
      if (opsWithUsers.count(operandSrc) && opsToHoist.count(operandSrc) == 0) {
        return false;
      }
    }
  }

  // If no operand was loop variant, mark this op for motion.
  opsToHoist.insert(&op);
  return true;
}

// Checks if all ops in a region (i.e. list of blocks) are loop invariant.
bool areAllOpsInTheBlockListInvariant(
    Region &blockList, Value indVar, ValueRange iterArgs,
    SmallPtrSetImpl<Operation *> &opsWithUsers,
    SmallPtrSetImpl<Operation *> &opsToHoist) {

  for (auto &b : blockList) {
    for (auto &op : b) {
      if (!isOpLoopInvariant(op, indVar, iterArgs, opsWithUsers, opsToHoist)) {
        return false;
      }
    }
  }

  return true;
}

// Returns true if the affine.if op can be hoisted.
bool checkInvarianceOfNestedIfOps(Operation *op, Value indVar,
                                  ValueRange iterArgs,
                                  SmallPtrSetImpl<Operation *> &opsWithUsers,
                                  SmallPtrSetImpl<Operation *> &opsToHoist) {
  assert(isa<AffineIfOp>(op));
  auto ifOp = cast<AffineIfOp>(op);

  if (!areAllOpsInTheBlockListInvariant(ifOp.getThenRegion(), indVar, iterArgs,
                                        opsWithUsers, opsToHoist)) {
    return false;
  }

  if (!areAllOpsInTheBlockListInvariant(ifOp.getElseRegion(), indVar, iterArgs,
                                        opsWithUsers, opsToHoist)) {
    return false;
  }

  return true;
}

void LoopInvariantCodeMotion::runOnAffineForOp(AffineForOp forOp) {
  auto *loopBody = forOp.getBody();
  auto indVar = forOp.getInductionVar();
  ValueRange iterArgs = forOp.getRegionIterArgs();

  // This is the place where hoisted instructions would reside.
  OpBuilder b(forOp.getOperation());

  SmallPtrSet<Operation *, 8> opsToHoist;
  SmallVector<Operation *, 8> opsToMove;
  SmallPtrSet<Operation *, 8> opsWithUsers;

  for (auto &op : *loopBody) {
    // Register op in the set of ops that have users. This set is used
    // to prevent hoisting ops that depend on these ops that are
    // not being hoisted.
    if (!op.use_empty())
      opsWithUsers.insert(&op);
    if (!isa<AffineYieldOp>(op)) {
      if (isOpLoopInvariant(op, indVar, iterArgs, opsWithUsers, opsToHoist)) {
        opsToMove.push_back(&op);
      }
    }
  }

  // For all instructions that we found to be invariant, place sequentially
  // right before the for loop.
  for (auto *op : opsToMove) {
    op->moveBefore(forOp);
  }

  LLVM_DEBUG(forOp->print(llvm::dbgs() << "Modified loop\n"));
}

void LoopInvariantCodeMotion::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order.  This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation().walk([&](AffineForOp op) {
    LLVM_DEBUG(op->print(llvm::dbgs() << "\nOriginal loop\n"));
    runOnAffineForOp(op);
  });
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}
