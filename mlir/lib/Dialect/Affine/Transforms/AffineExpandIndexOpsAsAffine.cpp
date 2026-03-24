//===- AffineExpandIndexOpsAsAffine.cpp - Expand index ops to apply pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to expand affine index ops into one or more more
// fundamental operations.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEEXPANDINDEXOPSASAFFINE
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Create a constant splat of the given type with the given integer value.
static Value createTypedConstant(OpBuilder &b, Location loc, Type type,
                                 int64_t value) {
  if (auto vecTy = dyn_cast<VectorType>(type))
    return arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(vecTy, b.getIndexAttr(value)));
  return arith::ConstantIndexOp::create(b, loc, value);
}

/// Materialize an OpFoldResult (which represents a scalar index or constant)
/// as a Value matching the given target type. For vector target types, scalar
/// constants are splatted. Returns failure for dynamic basis with vector types
/// since that requires vector.broadcast which is not available here.
static FailureOr<Value> materializeBasis(OpBuilder &b, Location loc,
                                         OpFoldResult ofr, Type targetType) {
  std::optional<int64_t> cst = getConstantIntValue(ofr);
  if (cst)
    return createTypedConstant(b, loc, targetType, *cst);
  // Dynamic scalar basis value. For scalar target types, return as-is.
  if (isa<IndexType>(targetType))
    return getValueOrCreateConstantIndexOp(b, loc, ofr);
  // Dynamic scalar basis with vector target type -- would need
  // vector.broadcast, bail out.
  return failure();
}

/// Lowers `affine.delinearize_index` into a sequence of division and remainder
/// operations.
struct LowerDelinearizeIndexOps
    : public OpRewritePattern<AffineDelinearizeIndexOp> {
  using OpRewritePattern<AffineDelinearizeIndexOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineDelinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    // For scalar types, use the existing affine lowering path.
    if (isa<IndexType>(op.getLinearIndex().getType())) {
      FailureOr<SmallVector<Value>> multiIndex =
          delinearizeIndex(rewriter, op->getLoc(), op.getLinearIndex(),
                           op.getEffectiveBasis(), /*hasOuterBound=*/false);
      if (failed(multiIndex))
        return failure();
      rewriter.replaceOp(op, *multiIndex);
      return success();
    }

    // Vector lowering: emit arith div/rem ops (which work element-wise on
    // vectors).
    Location loc = op.getLoc();
    Value linearIndex = op.getLinearIndex();
    Type type = linearIndex.getType();
    SmallVector<OpFoldResult> basis = op.getEffectiveBasis();

    // Compute cumulative products of basis from the right. These serve as
    // divisors: for basis (B0, B1, B2), the divisors are (B1*B2, B2).
    SmallVector<Value> divisors;
    Value cumulativeProd = createTypedConstant(rewriter, loc, type, 1);
    for (OpFoldResult basisElem : llvm::reverse(basis)) {
      FailureOr<Value> basisVal =
          materializeBasis(rewriter, loc, basisElem, type);
      if (failed(basisVal))
        return failure();
      cumulativeProd =
          arith::MulIOp::create(rewriter, loc, cumulativeProd, *basisVal);
      divisors.push_back(cumulativeProd);
    }

    // Emit div/mod pairs from the most-significant dimension to the least.
    SmallVector<Value> results;
    results.reserve(divisors.size() + 1);
    Value residual = linearIndex;
    for (Value divisor : llvm::reverse(divisors)) {
      Value quotient = arith::DivSIOp::create(rewriter, loc, residual, divisor);
      Value product = arith::MulIOp::create(rewriter, loc, quotient, divisor);
      Value remainder = arith::SubIOp::create(rewriter, loc, residual, product);
      results.push_back(quotient);
      residual = remainder;
    }
    results.push_back(residual);
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Lowers `affine.linearize_index` into a sequence of multiplications and
/// additions.
struct LowerLinearizeIndexOps final : OpRewritePattern<AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    // Should be folded away, included here for safety.
    if (op.getMultiIndex().empty()) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
      return success();
    }

    // For scalar types, use the existing affine lowering path.
    if (isa<IndexType>(op.getLinearIndex().getType())) {
      SmallVector<OpFoldResult> multiIndex =
          getAsOpFoldResult(op.getMultiIndex());
      OpFoldResult linearIndex =
          linearizeIndex(rewriter, op.getLoc(), multiIndex, op.getMixedBasis());
      Value linearIndexValue =
          getValueOrCreateConstantIntOp(rewriter, op.getLoc(), linearIndex);
      rewriter.replaceOp(op, linearIndexValue);
      return success();
    }

    // Vector lowering: emit arith ops (which work element-wise on vectors).
    //
    // linearize_index [i0, i1, ..., iN-1] by (B0, B1, ..., BN-1)
    // = i0 * stride_0 + i1 * stride_1 + ... + iN-1
    // where stride_k = B_{k+1} * B_{k+2} * ... * B_{N-1}
    //
    // We compute from the back: result = iN-1, stride = 1, then:
    //   stride *= B_{k}, result += i_k * stride
    Location loc = op.getLoc();
    Type type = op.getLinearIndex().getType();
    SmallVector<OpFoldResult> effectiveBasis = op.getEffectiveBasis();
    ValueRange indices = op.getMultiIndex();

    // effectiveBasis drops the outer bound. For indices [i0, i1, ..., iN-1]:
    //   no outer bound:  effectiveBasis = [B1, B2, ..., BN-1] (N-1 elems)
    //   has outer bound: effectiveBasis = [B0, B1, ..., BN-1] (N elems,
    //                    but B0 is advisory, dropped by getEffectiveBasis)
    //
    // Computation: result = iN-1 + BN-1 * (iN-2 + BN-2 * (... + B1 * i0))
    // Or equivalently, accumulate from back:
    //   result = iN-1
    //   stride = 1
    //   for k = numBasis-1 downto 0:
    //     stride *= effectiveBasis[k]
    //     result += indices[k] * stride
    //
    // This works because effectiveBasis[k] is the "size" of dimension k+1,
    // and indices[k] is paired with the product of all sizes after it.
    Value result = indices.back();
    Value stride = createTypedConstant(rewriter, loc, type, 1);

    for (int i = static_cast<int>(effectiveBasis.size()) - 1; i >= 0; --i) {
      FailureOr<Value> basisVal =
          materializeBasis(rewriter, loc, effectiveBasis[i], type);
      if (failed(basisVal))
        return failure();
      stride = arith::MulIOp::create(rewriter, loc, stride, *basisVal);
      Value term = arith::MulIOp::create(rewriter, loc, indices[i], stride);
      result = arith::AddIOp::create(rewriter, loc, term, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ExpandAffineIndexOpsAsAffinePass
    : public affine::impl::AffineExpandIndexOpsAsAffineBase<
          ExpandAffineIndexOpsAsAffinePass> {
public:
  ExpandAffineIndexOpsAsAffinePass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateAffineExpandIndexOpsAsAffinePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::affine::populateAffineExpandIndexOpsAsAffinePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<LowerDelinearizeIndexOps, LowerLinearizeIndexOps>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::affine::createAffineExpandIndexOpsAsAffinePass() {
  return std::make_unique<ExpandAffineIndexOpsAsAffinePass>();
}
