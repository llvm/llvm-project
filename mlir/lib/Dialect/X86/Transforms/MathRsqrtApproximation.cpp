//===- MathRsqrtApproximation.cpp - Approximate math.rsqrt with AVX -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an approximation of `math.rsqrt` using the AVX
// `vrsqrtps` instruction refined by one Newton-Raphson step.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

// Expands and unrolls n-D vector operands into multiple fixed size 1-D vectors
// and calls the compute function with 1-D vector operands. Stitches back all
// results into the original n-D vector result.
//
// Examples: vectorWidth = 8
//   - vector<4x8xf32> unrolled 4 times
//   - vector<16xf32> expanded to vector<2x8xf32> and unrolled 2 times
//   - vector<4x16xf32> expanded to vector<4x2x8xf32> and unrolled 4*2 times
//
// It is the caller's responsibility to verify that the inner dimension is
// divisible by the vectorWidth, and that all operands have the same vector
// shape.
static Value
handleMultidimensionalVectors(ImplicitLocOpBuilder &builder,
                              ValueRange operands, int64_t vectorWidth,
                              llvm::function_ref<Value(ValueRange)> compute) {
  assert(!operands.empty() && "operands must be not empty");
  assert(vectorWidth > 0 && "vector width must be larger than 0");

  VectorType inputType = cast<VectorType>(operands[0].getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // If input shape matches target vector width, we can just call the
  // user-provided compute function with the operands.
  if (inputShape == llvm::ArrayRef(vectorWidth))
    return compute(operands);

  // Check if the inner dimension has to be expanded, or we can directly iterate
  // over the outer dimensions of the vector.
  int64_t innerDim = inputShape.back();
  int64_t expansionDim = innerDim / vectorWidth;
  assert((innerDim % vectorWidth == 0) && "invalid inner dimension size");

  // Maybe expand operands to the higher rank vector shape that we'll use to
  // iterate over and extract one dimensional vectors.
  SmallVector<int64_t> expandedShape(inputShape);
  SmallVector<Value> expandedOperands(operands);

  if (expansionDim > 1) {
    // Expand shape from [..., innerDim] to [..., expansionDim, vectorWidth].
    expandedShape.insert(expandedShape.end() - 1, expansionDim);
    expandedShape.back() = vectorWidth;

    for (unsigned i = 0; i < operands.size(); ++i) {
      auto operand = operands[i];
      auto eltType = cast<VectorType>(operand.getType()).getElementType();
      auto expandedType = VectorType::get(expandedShape, eltType);
      expandedOperands[i] =
          vector::ShapeCastOp::create(builder, expandedType, operand);
    }
  }

  // Iterate over all outer dimensions of the compute shape vector type.
  auto iterationDims = ArrayRef<int64_t>(expandedShape).drop_back();
  int64_t maxIndex = computeMaxLinearIndex(iterationDims);
  auto strides = computeStrides(iterationDims);

  // Compute results for each one dimensional vector.
  SmallVector<Value> results(maxIndex);

  for (int64_t i = 0; i < maxIndex; ++i) {
    auto offsets = delinearize(i, strides);

    SmallVector<Value> extracted(expandedOperands.size());
    for (const auto &tuple : llvm::enumerate(expandedOperands))
      extracted[tuple.index()] =
          vector::ExtractOp::create(builder, tuple.value(), offsets);

    results[i] = compute(extracted);
  }

  // Stitch results together into one large vector.
  Type resultEltType = cast<VectorType>(results[0].getType()).getElementType();
  Type resultExpandedType = VectorType::get(expandedShape, resultEltType);
  Value result = arith::ConstantOp::create(
      builder, resultExpandedType, builder.getZeroAttr(resultExpandedType));

  for (int64_t i = 0; i < maxIndex; ++i)
    result = vector::InsertOp::create(builder, results[i], result,
                                      delinearize(i, strides));

  // Reshape back to the original vector shape.
  return vector::ShapeCastOp::create(
      builder, VectorType::get(inputShape, resultEltType), result);
}

namespace {
struct RsqrtApproximation : public OpRewritePattern<math::RsqrtOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(math::RsqrtOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
RsqrtApproximation::matchAndRewrite(math::RsqrtOp op,
                                    PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  // Only support already-vectorized rsqrt's.
  auto vectorType = dyn_cast<VectorType>(op.getOperand().getType());
  if (!vectorType || vectorType.getRank() == 0 ||
      vectorType.getShape().back() % 8 != 0)
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcastF32FromBits = [&](uint32_t bits) -> Value {
    Value i32Value = arith::ConstantOp::create(
        builder, builder.getI32IntegerAttr(static_cast<int32_t>(bits)));
    Value f32Value =
        arith::BitcastOp::create(builder, builder.getF32Type(), i32Value);
    return vector::BroadcastOp::create(builder, vectorType, f32Value);
  };
  auto bcastF32 = [&](float value) -> Value {
    Value f32Value =
        arith::ConstantOp::create(builder, builder.getF32FloatAttr(value));
    return vector::BroadcastOp::create(builder, vectorType, f32Value);
  };

  Value cstPosInf = bcastF32FromBits(0x7f800000u);
  Value cstOnePointFive = bcastF32(1.5f);
  Value cstNegHalf = bcastF32(-0.5f);
  Value cstMinNormPos = bcastF32FromBits(0x00800000u);

  Value negHalf = arith::MulFOp::create(builder, op.getOperand(), cstNegHalf);

  // Select only the inverse sqrt of positive normals (denormals are
  // flushed to zero).
  Value ltMinMask = arith::CmpFOp::create(builder, arith::CmpFPredicate::OLT,
                                          op.getOperand(), cstMinNormPos);
  Value infMask = arith::CmpFOp::create(builder, arith::CmpFPredicate::OEQ,
                                        op.getOperand(), cstPosInf);
  Value notNormalFiniteMask = arith::OrIOp::create(builder, ltMinMask, infMask);

  // Compute an approximate result.
  Value yApprox = handleMultidimensionalVectors(
      builder, op->getOperands(), 8, [&builder](ValueRange operands) -> Value {
        return x86::avx::RsqrtOp::create(builder, operands);
      });

  // Do a single step of Newton-Raphson iteration to improve the approximation.
  // This uses the formula y_{n+1} = y_n * (1.5 - y_n * (0.5 * x) * y_n).
  // It is essential to evaluate the inner term like this because forming
  // y_n^2 may over- or underflow.
  Value inner = arith::MulFOp::create(builder, negHalf, yApprox);
  Value fma = math::FmaOp::create(builder, yApprox, inner, cstOnePointFive);
  Value yNewton = arith::MulFOp::create(builder, yApprox, fma);

  // Select the result of the Newton-Raphson step for positive normal arguments.
  // For other arguments, choose the output of the intrinsic. This will
  // return rsqrt(+inf) = 0, rsqrt(x) = NaN if x < 0, and rsqrt(x) = +inf if
  // x is zero or a positive denormalized float (equivalent to flushing positive
  // denormalized inputs to zero).
  Value res =
      arith::SelectOp::create(builder, notNormalFiniteMask, yApprox, yNewton);
  rewriter.replaceOp(op, res);

  return success();
}

void mlir::x86::populateMathRsqrtApproximationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<RsqrtApproximation>(patterns.getContext());
}
