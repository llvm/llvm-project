//===- Utils.cpp ---- Utilities for affine dialect transformation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous transformation utilities for the Affine
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/// Promotes the `then` or the `else` block of `ifOp` (depending on whether
/// `elseBlock` is false or true) into `ifOp`'s containing block, and discards
/// the rest of the op.
static void promoteIfBlock(AffineIfOp ifOp, bool elseBlock) {
  if (elseBlock)
    assert(ifOp.hasElse() && "else block expected");

  Block *destBlock = ifOp.getOperation()->getBlock();
  Block *srcBlock = elseBlock ? ifOp.getElseBlock() : ifOp.getThenBlock();
  destBlock->getOperations().splice(
      Block::iterator(ifOp), srcBlock->getOperations(), srcBlock->begin(),
      std::prev(srcBlock->end()));
  ifOp.erase();
}

/// Returns the outermost affine.for/parallel op that the `ifOp` is invariant
/// on. The `ifOp` could be hoisted and placed right before such an operation.
/// This method assumes that the ifOp has been canonicalized (to be correct and
/// effective).
static Operation *getOutermostInvariantForOp(AffineIfOp ifOp) {
  // Walk up the parents past all for op that this conditional is invariant on.
  auto ifOperands = ifOp.getOperands();
  auto *res = ifOp.getOperation();
  while (!isa<FuncOp>(res->getParentOp())) {
    auto *parentOp = res->getParentOp();
    if (auto forOp = dyn_cast<AffineForOp>(parentOp)) {
      if (llvm::is_contained(ifOperands, forOp.getInductionVar()))
        break;
    } else if (auto parallelOp = dyn_cast<AffineParallelOp>(parentOp)) {
      for (auto iv : parallelOp.getIVs())
        if (llvm::is_contained(ifOperands, iv))
          break;
    } else if (!isa<AffineIfOp>(parentOp)) {
      // Won't walk up past anything other than affine.for/if ops.
      break;
    }
    // You can always hoist up past any affine.if ops.
    res = parentOp;
  }
  return res;
}

/// A helper for the mechanics of mlir::hoistAffineIfOp. Hoists `ifOp` just over
/// `hoistOverOp`. Returns the new hoisted op if any hoisting happened,
/// otherwise the same `ifOp`.
static AffineIfOp hoistAffineIfOp(AffineIfOp ifOp, Operation *hoistOverOp) {
  // No hoisting to do.
  if (hoistOverOp == ifOp)
    return ifOp;

  // Create the hoisted 'if' first. Then, clone the op we are hoisting over for
  // the else block. Then drop the else block of the original 'if' in the 'then'
  // branch while promoting its then block, and analogously drop the 'then'
  // block of the original 'if' from the 'else' branch while promoting its else
  // block.
  BlockAndValueMapping operandMap;
  OpBuilder b(hoistOverOp);
  auto hoistedIfOp = b.create<AffineIfOp>(ifOp.getLoc(), ifOp.getIntegerSet(),
                                          ifOp.getOperands(),
                                          /*elseBlock=*/true);

  // Create a clone of hoistOverOp to use for the else branch of the hoisted
  // conditional. The else block may get optimized away if empty.
  Operation *hoistOverOpClone = nullptr;
  // We use this unique name to identify/find  `ifOp`'s clone in the else
  // version.
  Identifier idForIfOp = b.getIdentifier("__mlir_if_hoisting");
  operandMap.clear();
  b.setInsertionPointAfter(hoistOverOp);
  // We'll set an attribute to identify this op in a clone of this sub-tree.
  ifOp.setAttr(idForIfOp, b.getBoolAttr(true));
  hoistOverOpClone = b.clone(*hoistOverOp, operandMap);

  // Promote the 'then' block of the original affine.if in the then version.
  promoteIfBlock(ifOp, /*elseBlock=*/false);

  // Move the then version to the hoisted if op's 'then' block.
  auto *thenBlock = hoistedIfOp.getThenBlock();
  thenBlock->getOperations().splice(thenBlock->begin(),
                                    hoistOverOp->getBlock()->getOperations(),
                                    Block::iterator(hoistOverOp));

  // Find the clone of the original affine.if op in the else version.
  AffineIfOp ifCloneInElse;
  hoistOverOpClone->walk([&](AffineIfOp ifClone) {
    if (!ifClone.getAttr(idForIfOp))
      return WalkResult::advance();
    ifCloneInElse = ifClone;
    return WalkResult::interrupt();
  });
  assert(ifCloneInElse && "if op clone should exist");
  // For the else block, promote the else block of the original 'if' if it had
  // one; otherwise, the op itself is to be erased.
  if (!ifCloneInElse.hasElse())
    ifCloneInElse.erase();
  else
    promoteIfBlock(ifCloneInElse, /*elseBlock=*/true);

  // Move the else version into the else block of the hoisted if op.
  auto *elseBlock = hoistedIfOp.getElseBlock();
  elseBlock->getOperations().splice(
      elseBlock->begin(), hoistOverOpClone->getBlock()->getOperations(),
      Block::iterator(hoistOverOpClone));

  return hoistedIfOp;
}

/// Replace affine.for with a 1-d affine.parallel and clone the former's body
/// into the latter while remapping values.
void mlir::affineParallelize(AffineForOp forOp) {
  Location loc = forOp.getLoc();
  OpBuilder outsideBuilder(forOp);
  // Creating empty 1-D affine.parallel op.
  AffineParallelOp newPloop = outsideBuilder.create<AffineParallelOp>(
      loc, llvm::None, llvm::None, forOp.getLowerBoundMap(),
      forOp.getLowerBoundOperands(), forOp.getUpperBoundMap(),
      forOp.getUpperBoundOperands());
  // Steal the body of the old affine for op and erase it.
  newPloop.region().takeBody(forOp.region());
  forOp.erase();
}

// Returns success if any hoisting happened.
LogicalResult mlir::hoistAffineIfOp(AffineIfOp ifOp, bool *folded) {
  // Bail out early if the ifOp returns a result.  TODO: Consider how to
  // properly support this case.
  if (ifOp.getNumResults() != 0)
    return failure();

  // Apply canonicalization patterns and folding - this is necessary for the
  // hoisting check to be correct (operands should be composed), and to be more
  // effective (no unused operands). Since the pattern rewriter's folding is
  // entangled with application of patterns, we may fold/end up erasing the op,
  // in which case we return with `folded` being set.
  OwningRewritePatternList patterns;
  AffineIfOp::getCanonicalizationPatterns(patterns, ifOp.getContext());
  bool erased;
  applyOpPatternsAndFold(ifOp, patterns, &erased);
  if (erased) {
    if (folded)
      *folded = true;
    return failure();
  }
  if (folded)
    *folded = false;

  // The folding above should have ensured this, but the affine.if's
  // canonicalization is missing composition of affine.applys into it.
  assert(llvm::all_of(ifOp.getOperands(),
                      [](Value v) {
                        return isTopLevelValue(v) || isForInductionVar(v);
                      }) &&
         "operands not composed");

  // We are going hoist as high as possible.
  // TODO: this could be customized in the future.
  auto *hoistOverOp = getOutermostInvariantForOp(ifOp);

  AffineIfOp hoistedIfOp = ::hoistAffineIfOp(ifOp, hoistOverOp);
  // Nothing to hoist over.
  if (hoistedIfOp == ifOp)
    return failure();

  // Canonicalize to remove dead else blocks (happens whenever an 'if' moves up
  // a sequence of affine.fors that are all perfectly nested).
  applyPatternsAndFoldGreedily(
      hoistedIfOp.getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));

  return success();
}
