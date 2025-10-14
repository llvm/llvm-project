//===- FoldAddIntoDest.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;

// Determine whether the value is defined to be zero.
static bool isDefinedAsZero(Value val) {
  if (!val)
    return false;

  // Check whether val is a constant scalar / vector splat / tensor splat float
  // or integer zero.
  if (matchPattern(val, m_AnyZeroFloat()) || matchPattern(val, m_Zero()))
    return true;

  return TypeSwitch<Operation *, bool>(val.getDefiningOp())
      .Case<linalg::FillOp, linalg::CopyOp>([&](auto op) {
        return op && op.getInputs().size() == 1 &&
               isDefinedAsZero(op.getInputs()[0]);
      })
      .Default([&](auto) { return false; });
}

/// Replace a linalg.add with one operand the single user of a contraction,
/// which has a zero-filled, "identity-mapped" destination and is dominated by
/// the `other` operand, by the contraction with `other` as its dest.
///
/// As an example, the following pseudo-code will be rewritten
///   %cst = arith.constant 0.000000e+00
///   %empty = tensor.empty()
///   %zeroed = linalg.fill ins(%cst : f32) outs(%empty : !type) -> !type
///   %C = linalg.matmul ins(%A, %B) outs(%zeroed)
///   %empty2 = tensor.empty()
///   %zeroed2 = linalg.fill ins(%cst : f32) outs(%empty2 : !type) -> !type
///   %F = linalg.matmul ins(%D, %E) outs(%zeroed2)
///   %out = linalg.add ins(%C, %F) outs(%empty)
/// to:
///   %cst = arith.constant 0.000000e+00
///   %empty = tensor.empty()
///   %zeroed = linalg.fill ins(%cst : f32) outs(%empty : !type) -> !type
///   %C = linalg.matmul ins(%A, %B) outs(%zeroed)
///   %out = linalg.matmul ins(%D, %E) outs(%C)
///
struct FoldAddIntoDest final : public OpRewritePattern<linalg::AddOp> {
  using OpRewritePattern<linalg::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    // For now, pattern only applies on tensor types (memref support is TODO).
    if (!addOp.hasPureTensorSemantics())
      return failure();

    Value dominatingOperand = nullptr;
    linalg::LinalgOp dominatedOp = nullptr;
    { // We will forget about which operand was left or right after this block.
      Value lhs = addOp.getInputs()[0];
      Value rhs = addOp.getInputs()[1];

      // Can only put one of addOp's operands in the dest/out arg of the other's
      // defining op based on suitable dominance.
      // TODO: Can be generalized to move ops around as long as that still
      //       respects use-def chains and doesn't affect side-effects.
      if (auto rhsOp = rhs.getDefiningOp<linalg::LinalgOp>()) {
        DominanceInfo domInfo(rhsOp);
        if (domInfo.properlyDominates(lhs, rhsOp)) {
          dominatingOperand = lhs;
          dominatedOp = rhsOp;
        }
      }
      if (auto lhsOp = lhs.getDefiningOp<linalg::LinalgOp>()) {
        DominanceInfo domInfo(lhsOp);
        if (domInfo.properlyDominates(rhs, lhsOp)) {
          dominatingOperand = rhs;
          dominatedOp = lhsOp;
        }
      }
      if (!dominatingOperand || !dominatedOp)
        return failure();
      // NB: As linalg.add's generalisation ignores the out argument in its
      //     region there is no need to perform checks on addOp's out argument.
    }

    // When dominated op is a contraction we know it accumulates on its out arg.
    // E.g., AddOp is not a contraction and hence ignores its out arg's value.
    // TODO: Generalize check to also pass in case of other LinalgOps that
    //       accumulate on their out arg but are not (binary) contraction ops.
    auto dominatedDestOp =
        dyn_cast<DestinationStyleOpInterface>((Operation *)dominatedOp);
    if (dominatedOp->getNumResults() != 1 ||
        !linalg::isaContractionOpInterface(dominatedOp) ||
        (!dominatedDestOp || dominatedDestOp.getNumDpsInits() != 1))
      return rewriter.notifyMatchFailure(
          dominatedOp, "expected dominated op to be single-result "
                       "destination-passing contraction");

    // To change the contraction's result, `addOp` must be its only user.
    if (!dominatedOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(
          dominatedOp,
          "expected linalg.add to be single user of contraction's result");

    // As `dominatedOp` was already accumulating on its out argument, it is only
    // safe to no longer use its current out arg when it is the additive ident.
    auto *destOperand = dominatedDestOp.getDpsInitOperand(0);
    if (!isDefinedAsZero(destOperand->get()))
      return rewriter.notifyMatchFailure(
          dominatedOp, "expected dominated op's dest to be additive zero");
    // TODO: If the other op is a contraction and has additive ident as dest, we
    // can swap the dests and achieve the proper sum, given suitable dominance.

    // As an operand to `addOp`, `dominatingOperand` has an identity affine_map.
    // Hence, we can only substitute `dominatingOperand` for the dest of the
    // contraction when dest's indexing_map corresponds to an identity map
    // w.r.t. just the dimensions of dest, i.e. is an ordered projection.
    SmallVector<AffineMap> indexMaps = dominatedOp.getIndexingMapsArray();
    int prevDimPos = -1;
    for (auto expr : indexMaps[destOperand->getOperandNumber()].getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim || prevDimPos > static_cast<int>(dim.getPosition()))
        return rewriter.notifyMatchFailure(
            dominatedOp, "expected index_map for contraction's dest to be an "
                         "ordered projection");
      prevDimPos = dim.getPosition();
    }

    // Replace the additive-ident, i.e. zero, out arg of the dominated op by the
    // dominating summand. This makes the dominated op's result the sum of both
    // of addOp's arguments - therefore we replace addOp and it uses by it.
    rewriter.modifyOpInPlace(
        dominatedOp, [&]() { dominatedOp->setOperand(2, dominatingOperand); });
    rewriter.replaceAllOpUsesWith(addOp, dominatedOp->getResult(0));
    return success();
  }
};

void linalg::populateFoldAddIntoDestPatterns(RewritePatternSet &patterns) {
  // Replace linalg.add when destination passing suffices for achieving the sum.
  patterns.add<FoldAddIntoDest>(patterns.getContext());
}
