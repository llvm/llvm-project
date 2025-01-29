//===- ArithToAMDGPU.cpp - Arith to AMDGPU dialect conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToAMDGPU/ArithToAMDGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAMDGPUCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::amdgpu;

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
  TruncFToFloat8RewritePattern(MLIRContext *ctx, bool saturateFP8,
                               Chipset chipset)
      : OpRewritePattern::OpRewritePattern(ctx), saturateFP8(saturateFP8),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult match(arith::TruncFOp op) const override;
  void rewrite(arith::TruncFOp op, PatternRewriter &rewriter) const override;
};

struct TruncfToFloat16RewritePattern final
    : public OpRewritePattern<arith::TruncFOp> {

  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

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
  if (auto inVecType = dyn_cast<VectorType>(inType)) {
    if (inVecType.isScalable())
      return failure();
    inType = inVecType.getElementType();
  }
  return success(isa<Float8E5M2FNUZType, Float8E4M3FNUZType>(inType));
}

void ExtFOnFloat8RewritePattern::rewrite(arith::ExtFOp op,
                                         PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  auto inType = dyn_cast<VectorType>(in.getType());
  if (!inType) {
    Value asFloat = rewriter.create<amdgpu::ExtPackedFp8Op>(
        loc, rewriter.getF32Type(), in, 0);
    Value result = castF32To(outElemType, asFloat, loc, rewriter);
    return rewriter.replaceOp(op, result);
  }
  int64_t numElements = inType.getNumElements();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  if (inType.getShape().empty()) {
    Value scalarIn =
        rewriter.create<vector::ExtractOp>(loc, in, ArrayRef<int64_t>{});
    // Recurse to send the 0-D vector case to the 1-D vector case
    Value scalarExt =
        rewriter.create<arith::ExtFOp>(loc, outElemType, scalarIn);
    Value result = rewriter.create<vector::InsertOp>(loc, scalarExt, zero,
                                                     ArrayRef<int64_t>{});
    return rewriter.replaceOp(op, result);
  }

  VectorType outType = cast<VectorType>(op.getOut().getType());
  VectorType flatTy = VectorType::get(SmallVector<int64_t>{numElements},
                                      outType.getElementType());
  Value result = rewriter.createOrFold<vector::SplatOp>(loc, flatTy, zero);

  if (inType.getRank() > 1) {
    inType = VectorType::get(SmallVector<int64_t>{numElements},
                             inType.getElementType());
    in = rewriter.create<vector::ShapeCastOp>(loc, inType, in);
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

  if (inType.getRank() != outType.getRank()) {
    result = rewriter.create<vector::ShapeCastOp>(loc, outType, result);
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
  if (auto outVecType = dyn_cast<VectorType>(outType)) {
    if (outVecType.isScalable())
      return failure();
    outType = outVecType.getElementType();
  }
  auto inType = dyn_cast<FloatType>(getElementTypeOrSelf(op.getIn().getType()));
  if (inType && inType.getWidth() <= 8 && saturateFP8)
    // Conversion between 8-bit floats is not supported with truncation enabled.
    return failure();
  return success(isa<Float8E5M2FNUZType, Float8E4M3FNUZType>(outType));
}

void TruncFToFloat8RewritePattern::rewrite(arith::TruncFOp op,
                                           PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  if (saturateFP8)
    in = clampInput(rewriter, loc, outElemType, in);
  auto inVectorTy = dyn_cast<VectorType>(in.getType());
  VectorType truncResType = VectorType::get(4, outElemType);
  if (!inVectorTy) {
    Value asFloat = castToF32(in, loc, rewriter);
    Value asF8s = rewriter.create<amdgpu::PackedTrunc2xFp8Op>(
        loc, truncResType, asFloat, /*sourceB=*/nullptr, 0,
        /*existing=*/nullptr);
    Value result = rewriter.create<vector::ExtractOp>(loc, asF8s, 0);
    return rewriter.replaceOp(op, result);
  }
  VectorType outType = cast<VectorType>(op.getOut().getType());
  int64_t numElements = outType.getNumElements();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  if (outType.getShape().empty()) {
    Value scalarIn =
        rewriter.create<vector::ExtractOp>(loc, in, ArrayRef<int64_t>{});
    // Recurse to send the 0-D vector case to the 1-D vector case
    Value scalarTrunc =
        rewriter.create<arith::TruncFOp>(loc, outElemType, scalarIn);
    Value result = rewriter.create<vector::InsertOp>(loc, scalarTrunc, zero,
                                                     ArrayRef<int64_t>{});
    return rewriter.replaceOp(op, result);
  }

  VectorType flatTy = VectorType::get(SmallVector<int64_t>{numElements},
                                      outType.getElementType());
  Value result = rewriter.createOrFold<vector::SplatOp>(loc, flatTy, zero);

  if (inVectorTy.getRank() > 1) {
    inVectorTy = VectorType::get(SmallVector<int64_t>{numElements},
                                 inVectorTy.getElementType());
    in = rewriter.create<vector::ShapeCastOp>(loc, inVectorTy, in);
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

  if (inVectorTy.getRank() != outType.getRank()) {
    result = rewriter.create<vector::ShapeCastOp>(loc, outType, result);
  }

  rewriter.replaceOp(op, result);
}

LogicalResult TruncfToFloat16RewritePattern::match(arith::TruncFOp op) const {
  Type outType = op.getOut().getType();
  Type inputType = getElementTypeOrSelf(op.getIn());
  if (auto outVecType = dyn_cast<VectorType>(outType)) {
    if (outVecType.isScalable())
      return failure();
    outType = outVecType.getElementType();
  }
  return success(outType.isF16() && inputType.isF32());
}

void TruncfToFloat16RewritePattern::rewrite(arith::TruncFOp op,
                                            PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  VectorType truncResType = VectorType::get(2, outElemType);
  auto inVectorTy = dyn_cast<VectorType>(in.getType());

  // Handle the case where input type is not a vector type
  if (!inVectorTy) {
    auto sourceB = rewriter.create<LLVM::PoisonOp>(loc, rewriter.getF32Type());
    Value asF16s =
        rewriter.create<ROCDL::CvtPkRtz>(loc, truncResType, in, sourceB);
    Value result = rewriter.create<vector::ExtractOp>(loc, asF16s, 0);
    return rewriter.replaceOp(op, result);
  }
  VectorType outType = cast<VectorType>(op.getOut().getType());
  int64_t numElements = outType.getNumElements();
  Value zero = rewriter.createOrFold<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  Value result = rewriter.createOrFold<vector::SplatOp>(loc, outType, zero);

  if (inVectorTy.getRank() > 1) {
    inVectorTy = VectorType::get(SmallVector<int64_t>{numElements},
                                 inVectorTy.getElementType());
    in = rewriter.create<vector::ShapeCastOp>(loc, inVectorTy, in);
  }

  // Handle the vector case. We also handle the (uncommon) case where the vector
  // length is odd
  for (int64_t i = 0; i < numElements; i += 2) {
    int64_t elemsThisOp = std::min(numElements, i + 2) - i;
    Value thisResult = nullptr;
    Value elemA = rewriter.create<vector::ExtractOp>(loc, in, i);
    Value elemB = rewriter.create<LLVM::PoisonOp>(loc, rewriter.getF32Type());

    if (elemsThisOp == 2) {
      elemB = rewriter.create<vector::ExtractOp>(loc, in, i + 1);
    }

    thisResult =
        rewriter.create<ROCDL::CvtPkRtz>(loc, truncResType, elemA, elemB);
    // Place back the truncated result into the possibly larger vector. If we
    // are operating on a size 2 vector, these operations should be folded away
    thisResult = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, thisResult, 0, elemsThisOp, 1);
    result = rewriter.create<vector::InsertStridedSliceOp>(loc, thisResult,
                                                           result, i, 1);
  }

  if (inVectorTy.getRank() != outType.getRank()) {
    result = rewriter.create<vector::ShapeCastOp>(loc, outType, result);
  }

  rewriter.replaceOp(op, result);
}

void mlir::arith::populateArithToAMDGPUConversionPatterns(
    RewritePatternSet &patterns, bool convertFP8Arithmetic,
    bool saturateFP8Truncf, bool allowPackedF16Rtz, Chipset chipset) {

  if (convertFP8Arithmetic) {
    patterns.add<ExtFOnFloat8RewritePattern>(patterns.getContext());
    patterns.add<TruncFToFloat8RewritePattern>(patterns.getContext(),
                                               saturateFP8Truncf, chipset);
  }
  if (allowPackedF16Rtz)
    patterns.add<TruncfToFloat16RewritePattern>(patterns.getContext());
}

void ArithToAMDGPUConversionPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(op->getContext());
  FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
  if (failed(maybeChipset)) {
    emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
    return signalPassFailure();
  }

  bool convertFP8Arithmetic =
      maybeChipset->majorVersion == 9 && *maybeChipset >= Chipset(9, 4, 0);
  arith::populateArithToAMDGPUConversionPatterns(
      patterns, convertFP8Arithmetic, saturateFP8Truncf, allowPackedF16Rtz,
      *maybeChipset);
  if (failed(applyPatternsGreedily(op, std::move(patterns))))
    return signalPassFailure();
}
