//===- AffineExpandIndexOps.cpp - Affine expand index ops pass ------------===//
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

#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEEXPANDINDEXOPS
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

/// Given a basis (in static and dynamic components), return the sequence of
/// suffix products of the basis, including the product of the entire basis,
/// which must **not** contain an outer bound.
///
/// If excess dynamic values are provided, the values at the beginning
/// will be ignored. This allows for dropping the outer bound without
/// needing to manipulate the dynamic value array. `knownPositive`
/// indicases that the values being used to compute the strides are known
/// to be non-negative.
static SmallVector<Value> computeStrides(Location loc, RewriterBase &rewriter,
                                         ValueRange dynamicBasis,
                                         ArrayRef<int64_t> staticBasis,
                                         bool knownNonNegative) {
  if (staticBasis.empty())
    return {};

  SmallVector<Value> result;
  result.reserve(staticBasis.size());
  size_t dynamicIndex = dynamicBasis.size();
  Value dynamicPart = nullptr;
  int64_t staticPart = 1;
  // The products of the strides can't have overflow by definition of
  // affine.*_index.
  arith::IntegerOverflowFlags ovflags = arith::IntegerOverflowFlags::nsw;
  if (knownNonNegative)
    ovflags = ovflags | arith::IntegerOverflowFlags::nuw;
  for (int64_t elem : llvm::reverse(staticBasis)) {
    if (ShapedType::isDynamic(elem)) {
      // Note: basis elements and their products are, definitionally,
      // non-negative, so `nuw` is justified.
      if (dynamicPart)
        dynamicPart =
            arith::MulIOp::create(rewriter, loc, dynamicPart,
                                  dynamicBasis[dynamicIndex - 1], ovflags);
      else
        dynamicPart = dynamicBasis[dynamicIndex - 1];
      --dynamicIndex;
    } else {
      staticPart *= elem;
    }

    if (dynamicPart && staticPart == 1) {
      result.push_back(dynamicPart);
    } else {
      Value stride =
          rewriter.createOrFold<arith::ConstantIndexOp>(loc, staticPart);
      if (dynamicPart)
        stride =
            arith::MulIOp::create(rewriter, loc, dynamicPart, stride, ovflags);
      result.push_back(stride);
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

LogicalResult
affine::lowerAffineDelinearizeIndexOp(RewriterBase &rewriter,
                                      AffineDelinearizeIndexOp op) {
  Location loc = op.getLoc();
  Value linearIdx = op.getLinearIndex();
  unsigned numResults = op.getNumResults();
  ArrayRef<int64_t> staticBasis = op.getStaticBasis();
  if (numResults == staticBasis.size())
    staticBasis = staticBasis.drop_front();

  if (numResults == 1) {
    rewriter.replaceOp(op, linearIdx);
    return success();
  }

  SmallVector<Value> results;
  results.reserve(numResults);
  SmallVector<Value> strides =
      computeStrides(loc, rewriter, op.getDynamicBasis(), staticBasis,
                     /*knownNonNegative=*/true);

  Value zero = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);

  Value initialPart =
      arith::FloorDivSIOp::create(rewriter, loc, linearIdx, strides.front());
  results.push_back(initialPart);

  auto emitModTerm = [&](Value stride) -> Value {
    Value remainder = arith::RemSIOp::create(rewriter, loc, linearIdx, stride);
    Value remainderNegative = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, remainder, zero);
    // If the correction is relevant, this term is <= stride, which is known
    // to be positive in `index`. Otherwise, while 2 * stride might overflow,
    // this branch won't be taken, so the risk of `poison` is fine.
    Value corrected = arith::AddIOp::create(rewriter, loc, remainder, stride,
                                            arith::IntegerOverflowFlags::nsw);
    Value mod = arith::SelectOp::create(rewriter, loc, remainderNegative,
                                        corrected, remainder);
    return mod;
  };

  // Generate all the intermediate parts
  for (size_t i = 0, e = strides.size() - 1; i < e; ++i) {
    Value thisStride = strides[i];
    Value nextStride = strides[i + 1];
    Value modulus = emitModTerm(thisStride);
    // We know both inputs are positive, so floorDiv == div.
    // This could potentially be a divui, but it's not clear if that would
    // cause issues.
    Value divided = arith::DivSIOp::create(rewriter, loc, modulus, nextStride);
    results.push_back(divided);
  }

  results.push_back(emitModTerm(strides.back()));

  rewriter.replaceOp(op, results);
  return success();
}

LogicalResult affine::lowerAffineLinearizeIndexOp(RewriterBase &rewriter,
                                                  AffineLinearizeIndexOp op) {
  // Should be folded away, included here for safety.
  if (op.getMultiIndex().empty()) {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }

  Location loc = op.getLoc();
  ValueRange multiIndex = op.getMultiIndex();
  size_t numIndexes = multiIndex.size();
  ArrayRef<int64_t> staticBasis = op.getStaticBasis();
  if (numIndexes == staticBasis.size())
    staticBasis = staticBasis.drop_front();

  SmallVector<Value> strides =
      computeStrides(loc, rewriter, op.getDynamicBasis(), staticBasis,
                     /*knownNonNegative=*/op.getDisjoint());
  SmallVector<std::pair<Value, int64_t>> scaledValues;
  scaledValues.reserve(numIndexes);

  // Note: strides doesn't contain a value for the final element (stride 1)
  // and everything else lines up. We use the "mutable" accessor so we can get
  // our hands on an `OpOperand&` for the loop invariant counting function.
  for (auto [stride, idxOp] :
       llvm::zip_equal(strides, llvm::drop_end(op.getMultiIndexMutable()))) {
    Value scaledIdx = arith::MulIOp::create(rewriter, loc, idxOp.get(), stride,
                                            arith::IntegerOverflowFlags::nsw);
    int64_t numHoistableLoops = numEnclosingInvariantLoops(idxOp);
    scaledValues.emplace_back(scaledIdx, numHoistableLoops);
  }
  scaledValues.emplace_back(
      multiIndex.back(),
      numEnclosingInvariantLoops(op.getMultiIndexMutable()[numIndexes - 1]));

  // Sort by how many enclosing loops there are, ties implicitly broken by
  // size of the stride.
  llvm::stable_sort(scaledValues,
                    [&](auto l, auto r) { return l.second > r.second; });

  Value result = scaledValues.front().first;
  for (auto [scaledValue, numHoistableLoops] : llvm::drop_begin(scaledValues)) {
    std::ignore = numHoistableLoops;
    result = arith::AddIOp::create(rewriter, loc, result, scaledValue,
                                   arith::IntegerOverflowFlags::nsw);
  }
  rewriter.replaceOp(op, result);
  return success();
}

namespace {
struct LowerDelinearizeIndexOps
    : public OpRewritePattern<AffineDelinearizeIndexOp> {
  using OpRewritePattern<AffineDelinearizeIndexOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineDelinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    return affine::lowerAffineDelinearizeIndexOp(rewriter, op);
  }
};

struct LowerLinearizeIndexOps final : OpRewritePattern<AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    return affine::lowerAffineLinearizeIndexOp(rewriter, op);
  }
};

class ExpandAffineIndexOpsPass
    : public affine::impl::AffineExpandIndexOpsBase<ExpandAffineIndexOpsPass> {
public:
  ExpandAffineIndexOpsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateAffineExpandIndexOpsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::affine::populateAffineExpandIndexOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<LowerDelinearizeIndexOps, LowerLinearizeIndexOps>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::affine::createAffineExpandIndexOpsPass() {
  return std::make_unique<ExpandAffineIndexOpsPass>();
}
