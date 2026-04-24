//===- SimplifyAffineWithBounds.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements simplification patterns for affine.delinearize_index /
// affine.linearize_index pairs using value bounds analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "affine-simplify-with-bounds"

using namespace mlir;
using namespace mlir::affine;

/// Accumulate a single basis element into the running product expression.
/// Static values become affine constants, and dynamic values become symbols.
static void buildProductExpr(OpFoldResult basis, AffineExpr &productExpr,
                             SmallVectorImpl<Value> &operands,
                             MLIRContext *ctx) {
  if (auto val = getConstantIntValue(basis)) {
    productExpr = productExpr * getAffineConstantExpr(*val, ctx);
  } else {
    operands.push_back(cast<Value>(basis));
    productExpr = productExpr * getAffineSymbolExpr(operands.size() - 1, ctx);
  }
}

/// Try to find k consecutive elements from `lhs` (starting from tail offset)
/// whose product equals the single next element from `rhs`.
/// The product is accumulated incrementally to avoid redundant computation.
/// Returns the number of matched elements k, or std::nullopt if no match.
static std::optional<size_t> tryMatchProduct(ArrayRef<OpFoldResult> lhs,
                                             size_t lhsTailConsumed,
                                             ArrayRef<OpFoldResult> rhs,
                                             size_t rhsTailConsumed,
                                             MLIRContext *ctx) {
  // Build a Variable for the single rhs element.
  AffineExpr rhsExpr = getAffineConstantExpr(1, ctx);
  SmallVector<Value> rhsOperands;
  buildProductExpr(rhs[rhs.size() - rhsTailConsumed - 1], rhsExpr, rhsOperands,
                   ctx);
  ValueBoundsConstraintSet::Variable rhsVar(
      AffineMap::get(0, rhsOperands.size(), rhsExpr, ctx), rhsOperands);

  // Incrementally accumulate lhs product and check for equality.
  AffineExpr lhsExpr = getAffineConstantExpr(1, ctx);
  SmallVector<Value> lhsOperands;
  for (size_t k = 1; k + lhsTailConsumed <= lhs.size(); ++k) {
    buildProductExpr(lhs[lhs.size() - lhsTailConsumed - k], lhsExpr,
                     lhsOperands, ctx);
    AffineMap lhsMap = AffineMap::get(0, lhsOperands.size(), lhsExpr, ctx);
    ValueBoundsConstraintSet::Variable lhsVar(lhsMap, lhsOperands);
    FailureOr<bool> result = ValueBoundsConstraintSet::areEqual(lhsVar, rhsVar);
    if (succeeded(result) && *result)
      return k;
  }
  return std::nullopt;
}

namespace {

/// Simplify delinearize(linearize) pairs from the tail by matching groups of
/// dimensions whose basis products are equal via ValueBounds analysis.
///
/// For each step from the tail, tries:
///   1. Many-to-one: k linearize dims -> 1 delinearize dim
///   2. One-to-many: 1 linearize dim -> k delinearize dims
///
/// Matched trailing dimensions are peeled off. Unmatched prefix dimensions
/// are left as residual linearize/delinearize operations.
///
/// Example (many-to-one, D*E == Z):
///   %lin = affine.linearize_index disjoint [%a, %b, %c, %d, %e]
///              by (A, B, C, D, E)
///   %result:3 = affine.delinearize_index %lin into (X, Y, Z)
/// ->
///   %prefix_lin = affine.linearize_index disjoint [%a, %b, %c] by (A, B, C)
///   %prefix:2 = affine.delinearize_index %prefix_lin into (X, Y)
///   %tail = affine.linearize_index disjoint [%d, %e] by (D, E)
///   %result = [%prefix#0, %prefix#1, %tail]
struct SimplifyDelinearizeOfLinearizeDisjoint final
    : OpRewritePattern<AffineDelinearizeIndexOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(AffineDelinearizeIndexOp delinearizeOp,
                                PatternRewriter &rewriter) const override {
    auto linearizeOp =
        delinearizeOp.getLinearIndex().getDefiningOp<AffineLinearizeIndexOp>();
    if (!linearizeOp)
      return rewriter.notifyMatchFailure(delinearizeOp,
                                         "index doesn't come from linearize");

    if (!linearizeOp.getDisjoint())
      return rewriter.notifyMatchFailure(linearizeOp, "not disjoint");

    SmallVector<OpFoldResult> linBasis = linearizeOp.getMixedBasis();
    SmallVector<OpFoldResult> delinBasis = delinearizeOp.getMixedBasis();
    ValueRange linInputs = linearizeOp.getMultiIndex();
    MLIRContext *ctx = rewriter.getContext();

    // Track how many elements consumed from each tail.
    size_t linTailConsumed = 0;
    size_t delinTailConsumed = 0;

    // For each matched group (innermost first), record the number of
    // linearize and delinearize dimensions it spans. Many-to-one groups
    // have linCount > 1, one-to-many groups have delinCount > 1.
    SmallVector<std::pair<size_t, size_t>> matchedGroups;

    while (linTailConsumed < linBasis.size() &&
           delinTailConsumed < delinBasis.size()) {
      // Try many-to-one: k lin dims -> 1 delin dim.
      if (std::optional<size_t> k = tryMatchProduct(
              linBasis, linTailConsumed, delinBasis, delinTailConsumed, ctx)) {
        matchedGroups.emplace_back(*k, 1);
        linTailConsumed += *k;
        delinTailConsumed += 1;
        continue;
      }
      // Try one-to-many: 1 lin dim -> k delin dims.
      if (std::optional<size_t> k = tryMatchProduct(
              delinBasis, delinTailConsumed, linBasis, linTailConsumed, ctx)) {
        matchedGroups.emplace_back(1, *k);
        delinTailConsumed += *k;
        linTailConsumed += 1;
        continue;
      }
      break;
    }

    if (matchedGroups.empty())
      return rewriter.notifyMatchFailure(delinearizeOp,
                                         "no trailing dimensions matched");

    SmallVector<Value> results;

    // Build residual prefix ops for unmatched dimensions.
    if (delinTailConsumed < delinBasis.size()) {
      // Partial match: create residual linearize + delinearize for the
      // unmatched prefix.
      Value residualLinearize = AffineLinearizeIndexOp::create(
          rewriter, linearizeOp.getLoc(), linInputs.drop_back(linTailConsumed),
          ArrayRef(linBasis).drop_back(linTailConsumed),
          linearizeOp.getDisjoint());
      auto residualDelinearize = AffineDelinearizeIndexOp::create(
          rewriter, delinearizeOp.getLoc(), residualLinearize,
          ArrayRef(delinBasis).drop_back(delinTailConsumed),
          delinearizeOp.hasOuterBound());
      results.append(residualDelinearize.getResults().begin(),
                     residualDelinearize.getResults().end());
    } else if (!delinearizeOp.hasOuterBound()) {
      // All basis elements consumed, but the original delinearize has no outer
      // bound which requires special handling.
      ValueRange remainingInputs = linInputs.drop_back(linTailConsumed);
      if (remainingInputs.empty()) {
        // The outermost delinearize result is guaranteed to be zero.
        results.push_back(arith::ConstantIndexOp::create(
            rewriter, delinearizeOp.getLoc(), 0));
      } else if (remainingInputs.size() == 1) {
        // Pass through the single remaining input.
        results.push_back(remainingInputs.front());
      } else {
        // Re-linearize the remaining inputs to produce the outermost result.
        Value newLin = AffineLinearizeIndexOp::create(
            rewriter, linearizeOp.getLoc(), remainingInputs,
            ArrayRef(linBasis).drop_back(linTailConsumed),
            linearizeOp.getDisjoint());
        results.push_back(newLin);
      }
    }

    // Build results for each matched group.
    size_t linInputOffset = linInputs.size() - linTailConsumed;
    size_t linBasisOffset = linBasis.size() - linTailConsumed;
    size_t delinBasisOffset = delinBasis.size() - delinTailConsumed;
    for (auto [linCount, delinCount] : llvm::reverse(matchedGroups)) {
      if (linCount == 1 && delinCount == 1) {
        // Exact 1:1 match: pass through directly.
        results.push_back(linInputs[linInputOffset]);
      } else if (linCount > 1) {
        // Many-to-one: re-linearize the group's lin inputs.
        Value newLin = AffineLinearizeIndexOp::create(
            rewriter, linearizeOp.getLoc(),
            linInputs.slice(linInputOffset, linCount),
            ArrayRef(linBasis).slice(linBasisOffset, linCount),
            /*disjoint=*/true);
        results.push_back(newLin);
      } else {
        // One-to-many: delinearize the single lin input.
        auto newDelin = AffineDelinearizeIndexOp::create(
            rewriter, delinearizeOp.getLoc(), linInputs[linInputOffset],
            ArrayRef(delinBasis).slice(delinBasisOffset, delinCount),
            /*hasOuterBound=*/true);
        results.append(newDelin.getResults().begin(),
                       newDelin.getResults().end());
      }
      linInputOffset += linCount;
      linBasisOffset += linCount;
      delinBasisOffset += delinCount;
    }

    rewriter.replaceOp(delinearizeOp, results);
    return success();
  }
};

} // namespace

void affine::populateSimplifyAffineWithBoundsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SimplifyDelinearizeOfLinearizeDisjoint>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_SIMPLIFYAFFINEWITHBOUNDS
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

namespace {
struct SimplifyAffineWithBoundsPass
    : affine::impl::SimplifyAffineWithBoundsBase<SimplifyAffineWithBoundsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Add canonicalization patterns first so cheap exact-match cases are
    // handled without invoking value bounds analysis.
    AffineDelinearizeIndexOp::getCanonicalizationPatterns(patterns,
                                                          &getContext());
    AffineLinearizeIndexOp::getCanonicalizationPatterns(patterns,
                                                        &getContext());
    populateSimplifyAffineWithBoundsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
