//===- ArithToAMDGPU.cpp - Arith to AMDGPU dialect conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToAMDGPU/ArithToAMDGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAMDGPUCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ArithToAMDGPUConversionPass final
    : impl::ArithToAMDGPUConversionPassBase<ArithToAMDGPUConversionPass> {
  using impl::ArithToAMDGPUConversionPassBase<
      ArithToAMDGPUConversionPass>::ArithToAMDGPUConversionPassBase;

  void runOnOperation() override;
};

struct ExtFOnFloat8RewritePattern final : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match(arith::ExtFOp op) const override;
  void rewrite(arith::ExtFOp op, PatternRewriter &rewriter) const override;
};

struct TruncFToFloat8RewritePattern final : OpRewritePattern<arith::TruncFOp> {
  bool saturateFP8 = false;
  TruncFToFloat8RewritePattern(MLIRContext *ctx, bool saturateFP8)
      : OpRewritePattern::OpRewritePattern(ctx), saturateFP8(saturateFP8) {}

  LogicalResult match(arith::TruncFOp op) const override;
  void rewrite(arith::TruncFOp op, PatternRewriter &rewriter) const override;
};
} // end namespace

static Value castF32To(Type elementType, Value f32, Location loc,
                       PatternRewriter &rewriter) {
  if (elementType.isF32())
    return f32;
  if (elementType.getIntOrFloatBitWidth() < 32)
    return rewriter.create<arith::TruncFOp>(loc, elementType, f32);
  if (elementType.getIntOrFloatBitWidth() > 32)
    return rewriter.create<arith::ExtFOp>(loc, elementType, f32);
  llvm_unreachable("The only 32-bit float type is f32");
}

LogicalResult ExtFOnFloat8RewritePattern::match(arith::ExtFOp op) const {
  Type inType = op.getIn().getType();
  if (auto inVecType = inType.dyn_cast<VectorType>()) {
    if (inVecType.isScalable())
      return failure();
    if (inVecType.getShape().size() > 1)
      // Multi-dimensional vectors are currently unsupported.
      return failure();
    inType = inVecType.getElementType();
  }
  return success(inType.isFloat8E5M2FNUZ() || inType.isFloat8E4M3FNUZ());
}

void ExtFOnFloat8RewritePattern::rewrite(arith::ExtFOp op,
                                         PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  if (!in.getType().isa<VectorType>()) {
    Value asFloat = rewriter.create<amdgpu::ExtPackedFp8Op>(
        loc, rewriter.getF32Type(), in, 0);
    Value result = castF32To(outElemType, asFloat, loc, rewriter);
    return rewriter.replaceOp(op, result);
  }
  VectorType inType = in.getType().cast<VectorType>();
  int64_t numElements = inType.getNumElements();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  Value result =
      rewriter.createOrFold<vector::SplatOp>(loc, op.getOut().getType(), zero);
  if (inType.getShape().empty()) {
    Value scalarIn =
        rewriter.create<vector::ExtractOp>(loc, in, ArrayRef<int64_t>{});
    // Recurse to send the 0-D vector case to the 1-D vector case
    Value scalarExt =
        rewriter.create<arith::ExtFOp>(loc, outElemType, scalarIn);
    result = rewriter.create<vector::InsertOp>(loc, scalarExt, zero,
                                               ArrayRef<int64_t>{});
    return rewriter.replaceOp(op, result);
  }
  for (int64_t i = 0; i < numElements; i += 4) {
    int64_t elemsThisOp = std::min(numElements, i + 4) - i;
    Value inSlice = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, in, i, elemsThisOp, 1);
    for (int64_t j = 0; j < elemsThisOp; ++j) {
      Value asFloat = rewriter.create<amdgpu::ExtPackedFp8Op>(
          loc, rewriter.getF32Type(), inSlice, j);
      Value asType = castF32To(outElemType, asFloat, loc, rewriter);
      result = rewriter.create<vector::InsertOp>(loc, asType, result, i + j);
    }
  }
  rewriter.replaceOp(op, result);
}

static Value castToF32(Value value, Location loc, PatternRewriter &rewriter) {
  Type type = value.getType();
  if (type.isF32())
    return value;
  if (type.getIntOrFloatBitWidth() < 32)
    return rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), value);
  if (type.getIntOrFloatBitWidth() > 32)
    return rewriter.create<arith::TruncFOp>(loc, rewriter.getF32Type(), value);
  llvm_unreachable("The only 32-bit float type is f32");
}

// If `in` is a finite value, clamp it between the maximum and minimum values
// of `outElemType` so that subsequent conversion instructions don't
// overflow those out-of-range values to NaN. These semantics are commonly
// used in machine-learning contexts where failure to clamp would lead to
// excessive NaN production.
static Value clampInput(PatternRewriter &rewriter, Location loc,
                        Type outElemType, Value source) {
  Type sourceType = source.getType();
  const llvm::fltSemantics &sourceSem =
      cast<FloatType>(getElementTypeOrSelf(sourceType)).getFloatSemantics();
  const llvm::fltSemantics &targetSem =
      cast<FloatType>(outElemType).getFloatSemantics();

  APFloat min = APFloat::getLargest(targetSem, /*Negative=*/true);
  APFloat max = APFloat::getLargest(targetSem, /*Negative=*/false);
  bool ignoredLosesInfo = false;
  // We can ignore conversion failures here because this conversion promotes
  // from a smaller type to a larger one - ex. there can be no loss of precision
  // when casting fp8 to f16.
  (void)min.convert(sourceSem, APFloat::rmNearestTiesToEven, &ignoredLosesInfo);
  (void)max.convert(sourceSem, APFloat::rmNearestTiesToEven, &ignoredLosesInfo);

  Value minCst = createScalarOrSplatConstant(rewriter, loc, sourceType, min);
  Value maxCst = createScalarOrSplatConstant(rewriter, loc, sourceType, max);

  Value inf = createScalarOrSplatConstant(
      rewriter, loc, sourceType,
      APFloat::getInf(sourceSem, /*Negative=*/false));
  Value negInf = createScalarOrSplatConstant(
      rewriter, loc, sourceType, APFloat::getInf(sourceSem, /*Negative=*/true));
  Value isInf = rewriter.createOrFold<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, source, inf);
  Value isNegInf = rewriter.createOrFold<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, source, negInf);
  Value isNan = rewriter.createOrFold<arith::CmpFOp>(
      loc, arith::CmpFPredicate::UNO, source, source);
  Value isNonFinite = rewriter.create<arith::OrIOp>(
      loc, rewriter.create<arith::OrIOp>(loc, isInf, isNegInf), isNan);

  Value clampedBelow = rewriter.create<arith::MaximumFOp>(loc, source, minCst);
  Value clamped = rewriter.create<arith::MinimumFOp>(loc, clampedBelow, maxCst);
  Value res =
      rewriter.create<arith::SelectOp>(loc, isNonFinite, source, clamped);
  return res;
}

LogicalResult TruncFToFloat8RewritePattern::match(arith::TruncFOp op) const {
  // Only supporting default rounding mode as of now.
  if (op.getRoundingmodeAttr())
    return failure();
  Type outType = op.getOut().getType();
  if (auto outVecType = outType.dyn_cast<VectorType>()) {
    if (outVecType.isScalable())
      return failure();
    if (outVecType.getShape().size() > 1)
      // Multi-dimensional vectors are currently unsupported.
      return failure();
    outType = outVecType.getElementType();
  }
  auto inType = dyn_cast<FloatType>(getElementTypeOrSelf(op.getIn().getType()));
  if (inType && inType.getWidth() <= 8 && saturateFP8)
    // Conversion between 8-bit floats is not supported with truncation enabled.
    return failure();
  return success(outType.isFloat8E5M2FNUZ() || outType.isFloat8E4M3FNUZ());
}

void TruncFToFloat8RewritePattern::rewrite(arith::TruncFOp op,
                                           PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  if (saturateFP8)
    in = clampInput(rewriter, loc, outElemType, in);
  VectorType truncResType = VectorType::get(4, outElemType);
  if (!in.getType().isa<VectorType>()) {
    Value asFloat = castToF32(in, loc, rewriter);
    Value asF8s = rewriter.create<amdgpu::PackedTrunc2xFp8Op>(
        loc, truncResType, asFloat, /*sourceB=*/nullptr, 0,
        /*existing=*/nullptr);
    Value result = rewriter.create<vector::ExtractOp>(loc, asF8s, 0);
    return rewriter.replaceOp(op, result);
  }
  VectorType outType = op.getOut().getType().cast<VectorType>();
  int64_t numElements = outType.getNumElements();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  Value result = rewriter.createOrFold<vector::SplatOp>(loc, outType, zero);
  if (outType.getShape().empty()) {
    Value scalarIn =
        rewriter.create<vector::ExtractOp>(loc, in, ArrayRef<int64_t>{});
    // Recurse to send the 0-D vector case to the 1-D vector case
    Value scalarTrunc =
        rewriter.create<arith::TruncFOp>(loc, outElemType, scalarIn);
    result = rewriter.create<vector::InsertOp>(loc, scalarTrunc, zero,
                                               ArrayRef<int64_t>{});
    return rewriter.replaceOp(op, result);
  }

  for (int64_t i = 0; i < numElements; i += 4) {
    int64_t elemsThisOp = std::min(numElements, i + 4) - i;
    Value thisResult = nullptr;
    for (int64_t j = 0; j < elemsThisOp; j += 2) {
      Value elemA = rewriter.create<vector::ExtractOp>(loc, in, i + j);
      Value asFloatA = castToF32(elemA, loc, rewriter);
      Value asFloatB = nullptr;
      if (j + 1 < elemsThisOp) {
        Value elemB = rewriter.create<vector::ExtractOp>(loc, in, i + j + 1);
        asFloatB = castToF32(elemB, loc, rewriter);
      }
      thisResult = rewriter.create<amdgpu::PackedTrunc2xFp8Op>(
          loc, truncResType, asFloatA, asFloatB, j / 2, thisResult);
    }
    if (elemsThisOp < 4)
      thisResult = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, thisResult, 0, elemsThisOp, 1);
    result = rewriter.create<vector::InsertStridedSliceOp>(loc, thisResult,
                                                           result, i, 1);
  }
  rewriter.replaceOp(op, result);
}

void mlir::arith::populateArithToAMDGPUConversionPatterns(
    RewritePatternSet &patterns, bool saturateFP8TruncF) {
  patterns.add<ExtFOnFloat8RewritePattern>(patterns.getContext());
  patterns.add<TruncFToFloat8RewritePattern>(patterns.getContext(),
                                             saturateFP8TruncF);
}

void ArithToAMDGPUConversionPass::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(op->getContext());
  arith::populateArithToAMDGPUConversionPatterns(patterns, saturateFP8Truncf);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    return signalPassFailure();
}
