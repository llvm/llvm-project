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

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPINVARIANTCODEMOTION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-licm"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Affine loop invariant code motion (LICM) pass.
/// TODO: The pass is missing zero tripcount tests.
/// TODO: When compared to the other standard LICM pass, this pass
/// has some special handling for affine read/write ops but such handling
/// requires aliasing to be sound, and as such this pass is unsound. In
/// addition, this handling is nothing particular to affine memory ops but would
/// apply to any memory read/write effect ops. Either aliasing should be handled
/// or this pass can be removed and the standard LICM can be used.
struct LoopInvariantCodeMotion
    : public affine::impl::AffineLoopInvariantCodeMotionBase<
          LoopInvariantCodeMotion> {
  void runOnOperation() override;
  void runOnAffineForOp(AffineForOp forOp);
};
} // namespace

static bool
checkInvarianceOfNestedIfOps(AffineIfOp ifOp, AffineForOp loop,
                             SmallPtrSetImpl<Operation *> &opsWithUsers,
                             SmallPtrSetImpl<Operation *> &opsToHoist);
static bool isOpLoopInvariant(Operation &op, AffineForOp loop,
                              SmallPtrSetImpl<Operation *> &opsWithUsers,
                              SmallPtrSetImpl<Operation *> &opsToHoist);

static bool
areAllOpsInTheBlockListInvariant(Region &blockList, AffineForOp loop,
                                 SmallPtrSetImpl<Operation *> &opsWithUsers,
                                 SmallPtrSetImpl<Operation *> &opsToHoist);

/// Returns true if `op` is invariant on `loop`.
static bool isOpLoopInvariant(Operation &op, AffineForOp loop,
                              SmallPtrSetImpl<Operation *> &opsWithUsers,
                              SmallPtrSetImpl<Operation *> &opsToHoist) {
  Value iv = loop.getInductionVar();

  if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
    if (!checkInvarianceOfNestedIfOps(ifOp, loop, opsWithUsers, opsToHoist))
      return false;
  } else if (auto forOp = dyn_cast<AffineForOp>(op)) {
    if (!areAllOpsInTheBlockListInvariant(forOp.getRegion(), loop, opsWithUsers,
                                          opsToHoist))
      return false;
  } else if (auto parOp = dyn_cast<AffineParallelOp>(op)) {
    if (!areAllOpsInTheBlockListInvariant(parOp.getRegion(), loop, opsWithUsers,
                                          opsToHoist))
      return false;
  } else if (!isMemoryEffectFree(&op) &&
             !isa<AffineReadOpInterface, AffineWriteOpInterface>(&op)) {
    // Check for side-effecting ops. Affine read/write ops are handled
    // separately below.
    return false;
  } else if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
    // Register op in the set of ops that have users.
    opsWithUsers.insert(&op);
    SmallVector<AffineForOp, 8> userIVs;
    auto read = dyn_cast<AffineReadOpInterface>(op);
    Value memref =
        read ? read.getMemRef() : cast<AffineWriteOpInterface>(op).getMemRef();
    for (auto *user : memref.getUsers()) {
      // If the memref used by the load/store is used in a store elsewhere in
      // the loop nest, we do not hoist. Similarly, if the memref used in a
      // load is also being stored too, we do not hoist the load.
      // FIXME: This is missing checking aliases.
      if (&op == user)
        continue;
      if (hasEffect<MemoryEffects::Write>(user, memref) ||
          (hasEffect<MemoryEffects::Read>(user, memref) &&
           isa<AffineWriteOpInterface>(op))) {
        userIVs.clear();
        getAffineForIVs(*user, &userIVs);
        // Check that userIVs don't contain the for loop around the op.
        if (llvm::is_contained(userIVs, loop))
          return false;
      }
    }
  }

  // Check operands.
  ValueRange iterArgs = loop.getRegionIterArgs();
  for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
    auto *operandSrc = op.getOperand(i).getDefiningOp();

    // If the loop IV is the operand, this op isn't loop invariant.
    if (iv == op.getOperand(i))
      return false;

    // If the one of the iter_args is the operand, this op isn't loop invariant.
    if (llvm::is_contained(iterArgs, op.getOperand(i)))
      return false;

    if (operandSrc) {
      // If the value was defined in the loop (outside of the if/else region),
      // and that operation itself wasn't meant to be hoisted, then mark this
      // operation loop dependent.
      if (opsWithUsers.count(operandSrc) && opsToHoist.count(operandSrc) == 0)
        return false;
    }
  }

  // If no operand was loop variant, mark this op for motion.
  opsToHoist.insert(&op);
  return true;
}

// Checks if all ops in a region (i.e. list of blocks) are loop invariant.
static bool
areAllOpsInTheBlockListInvariant(Region &blockList, AffineForOp loop,
                                 SmallPtrSetImpl<Operation *> &opsWithUsers,
                                 SmallPtrSetImpl<Operation *> &opsToHoist) {

  for (auto &b : blockList) {
    for (auto &op : b) {
      if (!isOpLoopInvariant(op, loop, opsWithUsers, opsToHoist))
        return false;
    }
  }

  return true;
}

// Returns true if the affine.if op can be hoisted.
static bool
checkInvarianceOfNestedIfOps(AffineIfOp ifOp, AffineForOp loop,
                             SmallPtrSetImpl<Operation *> &opsWithUsers,
                             SmallPtrSetImpl<Operation *> &opsToHoist) {
  if (!areAllOpsInTheBlockListInvariant(ifOp.getThenRegion(), loop,
                                        opsWithUsers, opsToHoist))
    return false;

  if (!areAllOpsInTheBlockListInvariant(ifOp.getElseRegion(), loop,
                                        opsWithUsers, opsToHoist))
    return false;

  return true;
}

void LoopInvariantCodeMotion::runOnAffineForOp(AffineForOp forOp) {
  // This is the place where hoisted instructions would reside.
  OpBuilder b(forOp.getOperation());

  SmallPtrSet<Operation *, 8> opsToHoist;
  SmallVector<Operation *, 8> opsToMove;
  SmallPtrSet<Operation *, 8> opsWithUsers;

  for (Operation &op : *forOp.getBody()) {
    // Register op in the set of ops that have users. This set is used
    // to prevent hoisting ops that depend on these ops that are
    // not being hoisted.
    if (!op.use_empty())
      opsWithUsers.insert(&op);
    if (!isa<AffineYieldOp>(op)) {
      if (isOpLoopInvariant(op, forOp, opsWithUsers, opsToHoist)) {
        opsToMove.push_back(&op);
      }
    }
  }

  // For all instructions that we found to be invariant, place sequentially
  // right before the for loop.
  for (auto *op : opsToMove) {
    op->moveBefore(forOp);
  }
}

void LoopInvariantCodeMotion::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order.  This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation().walk([&](AffineForOp op) { runOnAffineForOp(op); });
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}
