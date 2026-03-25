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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "affine-simplify-with-bounds"

using namespace mlir;
using namespace mlir::affine;

/// Build a ValueBoundsConstraintSet::Variable representing the product of
/// the given basis elements. Static elements become constants in an affine
/// expression; dynamic elements become symbols.
static ValueBoundsConstraintSet::Variable
buildProductVariable(ArrayRef<OpFoldResult> bases, MLIRContext *ctx) {
  AffineExpr productExpr = getAffineConstantExpr(1, ctx);
  SmallVector<Value> operands;
  for (OpFoldResult basis : bases) {
    if (auto attr = dyn_cast<Attribute>(basis)) {
      int64_t val = cast<IntegerAttr>(attr).getInt();
      productExpr = productExpr * getAffineConstantExpr(val, ctx);
    } else {
      Value val = cast<Value>(basis);
      operands.push_back(val);
      productExpr = productExpr * getAffineSymbolExpr(operands.size() - 1, ctx);
    }
  }
  AffineMap productMap = AffineMap::get(0, operands.size(), productExpr, ctx);
  return ValueBoundsConstraintSet::Variable(productMap, operands);
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
  auto rhsVar =
      buildProductVariable(rhs.slice(rhs.size() - rhsTailConsumed - 1, 1), ctx);

  AffineExpr productExpr = getAffineConstantExpr(1, ctx);
  SmallVector<Value> operands;

  for (size_t k = 1; k + lhsTailConsumed <= lhs.size(); ++k) {
    OpFoldResult basis = lhs[lhs.size() - lhsTailConsumed - k];
    if (auto attr = dyn_cast<Attribute>(basis)) {
      int64_t val = cast<IntegerAttr>(attr).getInt();
      productExpr = productExpr * getAffineConstantExpr(val, ctx);
    } else {
      operands.push_back(cast<Value>(basis));
      productExpr = productExpr * getAffineSymbolExpr(operands.size() - 1, ctx);
    }

    AffineMap productMap = AffineMap::get(0, operands.size(), productExpr, ctx);
    ValueBoundsConstraintSet::Variable lhsVar(productMap, operands);
    FailureOr<bool> result = ValueBoundsConstraintSet::areEqual(lhsVar, rhsVar);
    if (succeeded(result) && *result)
      return k;
  }
  return std::nullopt;
}

namespace {

/// Simplify delinearize(linearize) pairs from the tail by matching multiple
/// linearize dimensions whose product equals a single delinearize dimension
/// (many-to-one).
///
/// Scans from the rightmost basis elements. For each trailing delinearize
/// dimension, accumulates consecutive linearize dimension products until an
/// equal product is found via ValueBounds. Matched trailing dimensions are
/// peeled off, and residual ops are created for unmatched prefixes.
///
/// Example:
///   %lin = affine.linearize_index disjoint [%a, %b, %c, %d, %e]
///              by (A, B, C, D, E)
///   %result:3 = affine.delinearize_index %lin into (X, Y, Z)
///
/// If D*E == Z but neither C, B*C, nor A*B*C equals Y, scanning stops
/// and the unmatched prefix is left as residual ops:
///   %prefix_lin = affine.linearize_index disjoint [%a, %b, %c] by (A, B, C)
///   %prefix:2 = affine.delinearize_index %prefix_lin into (X, Y)
///   %tail = affine.linearize_index disjoint [%d, %e] by (D, E)
///   %result = [%prefix#0, %prefix#1, %tail]
struct SimplifyDelinearizeOfLinearizeDisjointManyToOneTail final
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

    // For each matched delinearize dimension (innermost first), store the
    // number of linearize dimensions that map to it.
    SmallVector<size_t> groupLinCounts;

    while (linTailConsumed < linBasis.size() &&
           delinTailConsumed < delinBasis.size()) {
      // Try matching k linearize dimensions to one delinearize dimension.
      std::optional<size_t> k = tryMatchProduct(
          linBasis, linTailConsumed, delinBasis, delinTailConsumed, ctx);
      if (!k)
        break;
      groupLinCounts.push_back(*k);
      linTailConsumed += *k;
      delinTailConsumed += 1;
    }

    if (delinTailConsumed == 0)
      return rewriter.notifyMatchFailure(delinearizeOp,
                                         "no trailing dimensions matched");

    SmallVector<Value> results;
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

    // Produce one result per matched group. If the group size is 1,
    // the input passes through directly. Otherwise, a smaller linearize is
    // created over just that group's basis elements.
    size_t inputMatchStart = linInputs.size() - linTailConsumed;
    size_t basisMatchStart = linBasis.size() - linTailConsumed;
    size_t offset = 0;
    for (size_t count : llvm::reverse(groupLinCounts)) {
      if (count == 1) {
        results.push_back(linInputs[inputMatchStart + offset]);
      } else {
        Value newLin = AffineLinearizeIndexOp::create(
            rewriter, linearizeOp.getLoc(),
            linInputs.slice(inputMatchStart + offset, count),
            ArrayRef(linBasis).slice(basisMatchStart + offset, count),
            /*disjoint=*/true);
        results.push_back(newLin);
      }
      offset += count;
    }

    rewriter.replaceOp(delinearizeOp, results);
    return success();
  }
};

/// Simplify delinearize(linearize) pairs from the tail by matching a single
/// linearize dimension whose basis equals the product of multiple delinearize
/// dimensions (one-to-many).
///
/// Scans from the rightmost basis elements. For each trailing linearize
/// dimension, accumulates consecutive delinearize dimension products until an
/// equal product is found via ValueBounds. Matched trailing dimensions are
/// peeled off, and residual ops are created for unmatched prefixes.
///
/// Example:
///   %lin = affine.linearize_index disjoint [%a, %b, %c] by (A, B, C)
///   %result:5 = affine.delinearize_index %lin into (X, Y, Z, W, V)
///
/// If C == W*V but neither Z, Y*Z, nor X*Y*Z equals B, scanning stops
/// and the unmatched prefix is left as residual ops:
///   %prefix_lin = affine.linearize_index disjoint [%a, %b] by (A, B)
///   %prefix:3 = affine.delinearize_index %prefix_lin into (X, Y, Z)
///   %tail:2 = affine.delinearize_index %c into (W, V)
///   %result = [%prefix#0, %prefix#1, %prefix#2, %tail#0, %tail#1]
struct SimplifyDelinearizeOfLinearizeDisjointOneToManyTail final
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

    // For each matched linearize dimension (innermost first), store the
    // number of delinearize dimensions it expands to.
    SmallVector<size_t> groupDelinCounts;

    while (linTailConsumed < linBasis.size() &&
           delinTailConsumed < delinBasis.size()) {
      // Try matching k delinearize dimensions to one linearize dimension.
      std::optional<size_t> k = tryMatchProduct(delinBasis, delinTailConsumed,
                                                linBasis, linTailConsumed, ctx);
      if (!k)
        break;
      groupDelinCounts.push_back(*k);
      delinTailConsumed += *k;
      linTailConsumed += 1;
    }

    if (linTailConsumed == 0)
      return rewriter.notifyMatchFailure(delinearizeOp,
                                         "no trailing dimensions matched");

    SmallVector<Value> results;

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

    // Produce results for each matched group. If the group size is 1, the
    // input passes through directly. Otherwise, a smaller delinearize is
    // created over just that group's basis elements.
    size_t linMatchStart = linInputs.size() - linTailConsumed;
    size_t delinMatchStart = delinBasis.size() - delinTailConsumed;
    size_t inputOffset = 0;
    size_t delinOffset = 0;
    for (size_t count : llvm::reverse(groupDelinCounts)) {
      if (count == 1) {
        results.push_back(linInputs[linMatchStart + inputOffset]);
      } else {
        auto newDelin = AffineDelinearizeIndexOp::create(
            rewriter, delinearizeOp.getLoc(),
            linInputs[linMatchStart + inputOffset],
            ArrayRef(delinBasis).slice(delinMatchStart + delinOffset, count),
            /*hasOuterBound=*/true);
        results.append(newDelin.getResults().begin(),
                       newDelin.getResults().end());
      }
      inputOffset += 1;
      delinOffset += count;
    }

    rewriter.replaceOp(delinearizeOp, results);
    return success();
  }
};

} // namespace

void affine::populateSimplifyAffineWithBoundsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SimplifyDelinearizeOfLinearizeDisjointManyToOneTail,
               SimplifyDelinearizeOfLinearizeDisjointOneToManyTail>(
      patterns.getContext());
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
