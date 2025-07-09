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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
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
// Define commonly used chipsets versions for convenience.
constexpr Chipset kGfx942 = Chipset(9, 4, 2);
constexpr Chipset kGfx950 = Chipset(9, 5, 0);

struct ArithToAMDGPUConversionPass final
    : impl::ArithToAMDGPUConversionPassBase<ArithToAMDGPUConversionPass> {
  using impl::ArithToAMDGPUConversionPassBase<
      ArithToAMDGPUConversionPass>::ArithToAMDGPUConversionPassBase;

  void runOnOperation() override;
};

struct ExtFOnFloat8RewritePattern final : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  Chipset chipset;
  ExtFOnFloat8RewritePattern(MLIRContext *ctx, Chipset chipset)
      : OpRewritePattern::OpRewritePattern(ctx), chipset(chipset) {}

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override;
};

struct TruncFToFloat8RewritePattern final : OpRewritePattern<arith::TruncFOp> {
  bool saturateFP8 = false;
  TruncFToFloat8RewritePattern(MLIRContext *ctx, bool saturateFP8,
                               Chipset chipset)
      : OpRewritePattern::OpRewritePattern(ctx), saturateFP8(saturateFP8),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override;
};

struct TruncfToFloat16RewritePattern final
    : public OpRewritePattern<arith::TruncFOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override;
};

struct ScalingExtFRewritePattern final
    : OpRewritePattern<arith::ScalingExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  ScalingExtFRewritePattern(MLIRContext *ctx)
      : OpRewritePattern::OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(arith::ScalingExtFOp op,
                                PatternRewriter &rewriter) const override;
};

struct ScalingTruncFRewritePattern final
    : OpRewritePattern<arith::ScalingTruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  ScalingTruncFRewritePattern(MLIRContext *ctx)
      : OpRewritePattern::OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(arith::ScalingTruncFOp op,
                                PatternRewriter &rewriter) const override;
};

} // end namespace

static bool isSupportedF8(Type elementType, Chipset chipset) {
  if (chipset == kGfx942)
    return isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(elementType);
  if (hasOcpFp8(chipset))
    return isa<Float8E4M3FNType, Float8E5M2Type>(elementType);
  return false;
}

static Value castF32To(Type desType, Value f32, Location loc,
                       PatternRewriter &rewriter) {
  Type elementType = getElementTypeOrSelf(desType);
  if (elementType.isF32())
    return f32;
  if (elementType.getIntOrFloatBitWidth() < 32)
    return rewriter.create<arith::TruncFOp>(loc, desType, f32);
  if (elementType.getIntOrFloatBitWidth() > 32)
    return rewriter.create<arith::ExtFOp>(loc, desType, f32);
  llvm_unreachable("The only 32-bit float type is f32");
}

LogicalResult
ExtFOnFloat8RewritePattern::matchAndRewrite(arith::ExtFOp op,
                                            PatternRewriter &rewriter) const {
  Type inType = op.getIn().getType();
  auto inVecType = dyn_cast<VectorType>(inType);
  if (inVecType) {
    if (inVecType.isScalable())
      return failure();
    inType = inVecType.getElementType();
  }
  if (!isSupportedF8(inType, chipset))
    return failure();

  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  VectorType extResType = VectorType::get(2, rewriter.getF32Type());
  if (!inVecType) {
    Value asFloat = rewriter.create<amdgpu::ExtPackedFp8Op>(
        loc, rewriter.getF32Type(), in, 0);
    Value result = castF32To(outElemType, asFloat, loc, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
  int64_t numElements = inVecType.getNumElements();

  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  VectorType outType = cast<VectorType>(op.getOut().getType());

  if (inVecType.getShape().empty()) {
    Value zerodSplat =
        rewriter.createOrFold<vector::BroadcastOp>(loc, outType, zero);
    Value scalarIn =
        rewriter.create<vector::ExtractOp>(loc, in, ArrayRef<int64_t>{});
    Value scalarExt =
        rewriter.create<arith::ExtFOp>(loc, outElemType, scalarIn);
    Value result = rewriter.create<vector::InsertOp>(loc, scalarExt, zerodSplat,
                                                     ArrayRef<int64_t>{});
    rewriter.replaceOp(op, result);
    return success();
  }

  VectorType flatTy = VectorType::get(SmallVector<int64_t>{numElements},
                                      outType.getElementType());
  Value result = rewriter.createOrFold<vector::BroadcastOp>(loc, flatTy, zero);

  if (inVecType.getRank() > 1) {
    inVecType = VectorType::get(SmallVector<int64_t>{numElements},
                                inVecType.getElementType());
    in = rewriter.create<vector::ShapeCastOp>(loc, inVecType, in);
  }

  for (int64_t i = 0; i < numElements; i += 4) {
    int64_t elemsThisOp = std::min(numElements, i + 4) - i;
    Value inSlice = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, in, i, elemsThisOp, 1);
    for (int64_t j = 0; j < elemsThisOp; j += 2) {
      if (i + j + 1 < numElements) { // Convert two 8-bit elements
        Value asFloats = rewriter.create<amdgpu::ExtPackedFp8Op>(
            loc, extResType, inSlice, j / 2);
        Type desType = VectorType::get(2, outElemType);
        Value asType = castF32To(desType, asFloats, loc, rewriter);
        result = rewriter.create<vector::InsertStridedSliceOp>(
            loc, asType, result, i + j, 1);
      } else { // Convert a 8-bit element
        Value asFloat = rewriter.create<amdgpu::ExtPackedFp8Op>(
            loc, rewriter.getF32Type(), inSlice, j / 2 * 2);
        Value asType = castF32To(outElemType, asFloat, loc, rewriter);
        result = rewriter.create<vector::InsertOp>(loc, asType, result, i + j);
      }
    }
  }

  if (inVecType.getRank() != outType.getRank()) {
    result = rewriter.create<vector::ShapeCastOp>(loc, outType, result);
  }

  rewriter.replaceOp(op, result);
  return success();
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

LogicalResult
TruncFToFloat8RewritePattern::matchAndRewrite(arith::TruncFOp op,
                                              PatternRewriter &rewriter) const {
  // Only supporting default rounding mode as of now.
  if (op.getRoundingmodeAttr())
    return failure();
  Type outType = op.getOut().getType();
  auto outVecType = dyn_cast<VectorType>(outType);
  if (outVecType) {
    if (outVecType.isScalable())
      return failure();
    outType = outVecType.getElementType();
  }
  auto inType = dyn_cast<FloatType>(getElementTypeOrSelf(op.getIn().getType()));
  if (inType && inType.getWidth() <= 8 && saturateFP8)
    // Conversion between 8-bit floats is not supported with truncation enabled.
    return failure();

  if (!isSupportedF8(outType, chipset))
    return failure();

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
    rewriter.replaceOp(op, result);
    return success();
  }

  int64_t numElements = outVecType.getNumElements();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  if (outVecType.getShape().empty()) {
    Value scalarIn =
        rewriter.create<vector::ExtractOp>(loc, in, ArrayRef<int64_t>{});
    // Recurse to send the 0-D vector case to the 1-D vector case
    Value scalarTrunc =
        rewriter.create<arith::TruncFOp>(loc, outElemType, scalarIn);
    Value result = rewriter.create<vector::InsertOp>(loc, scalarTrunc, zero,
                                                     ArrayRef<int64_t>{});
    rewriter.replaceOp(op, result);
    return success();
  }

  VectorType flatTy = VectorType::get(SmallVector<int64_t>{numElements},
                                      outVecType.getElementType());
  Value result = rewriter.createOrFold<vector::BroadcastOp>(loc, flatTy, zero);

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

  if (inVectorTy.getRank() != outVecType.getRank()) {
    result = rewriter.create<vector::ShapeCastOp>(loc, outVecType, result);
  }

  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult TruncfToFloat16RewritePattern::matchAndRewrite(
    arith::TruncFOp op, PatternRewriter &rewriter) const {
  Type outType = op.getOut().getType();
  Type inputType = getElementTypeOrSelf(op.getIn());
  auto outVecType = dyn_cast<VectorType>(outType);
  if (outVecType) {
    if (outVecType.isScalable())
      return failure();
    outType = outVecType.getElementType();
  }
  if (!(outType.isF16() && inputType.isF32()))
    return failure();

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
    rewriter.replaceOp(op, result);
    return success();
  }
  int64_t numElements = outVecType.getNumElements();
  Value zero = rewriter.createOrFold<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  Value result =
      rewriter.createOrFold<vector::BroadcastOp>(loc, outVecType, zero);

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

  if (inVectorTy.getRank() != outVecType.getRank()) {
    result = rewriter.create<vector::ShapeCastOp>(loc, outVecType, result);
  }

  rewriter.replaceOp(op, result);
  return success();
}

/// Get the broadcasted / splatted value for a chain of ops.
static Value getOriginalVectorValue(Value value) {
  Value current = value;
  while (Operation *definingOp = current.getDefiningOp()) {
    bool skipOp = llvm::TypeSwitch<Operation *, bool>(definingOp)
                      .Case<vector::ShapeCastOp>([&current](auto op) {
                        current = op.getSource();
                        return true;
                      })
                      .Case<vector::BroadcastOp>([&current](auto op) {
                        current = op.getSource();
                        return false;
                      })
                      .Case<vector::SplatOp>([&current](auto op) {
                        current = op.getInput();
                        return false;
                      })
                      .Default([](Operation *) { return false; });

    if (!skipOp) {
      break;
    }
  }
  return current;
}

LogicalResult
ScalingExtFRewritePattern::matchAndRewrite(arith::ScalingExtFOp op,
                                           PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  constexpr int64_t opWidth = 2;

  Value in = op.getIn();
  Value scale = op.getScale();
  Value out = op.getOut();

  Type f32 = rewriter.getF32Type();
  Type inType = getElementTypeOrSelf(in);
  Type scaleType = getElementTypeOrSelf(scale);
  Type outType = getElementTypeOrSelf(out);

  VectorType outVecType = dyn_cast<VectorType>(out.getType());
  VectorType scaleVecType = dyn_cast<VectorType>(scale.getType());

  if (outVecType && outVecType.isScalable())
    return failure();

  Type scaleF32Type =
      scaleVecType ? VectorType::get(scaleVecType.getShape(), f32) : f32;
  if (scaleType.getIntOrFloatBitWidth() < 32)
    scale = rewriter.create<arith::ExtFOp>(loc, scaleF32Type, scale);
  else if (scaleType.getIntOrFloatBitWidth() > 32)
    scale = rewriter.create<arith::TruncFOp>(loc, scaleF32Type, scale);

  VectorType extScaleResultType = VectorType::get(opWidth, outType);

  if (!outVecType) {
    Value inCast = rewriter.create<vector::BroadcastOp>(
        loc, VectorType::get(1, inType), in);
    // TODO: replace this with non-packed ScaledExtOp
    Value scaleExt = rewriter.create<amdgpu::ScaledExtPackedOp>(
        loc, extScaleResultType, inCast, scale, 0);
    scaleExt = rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, scaleExt, 0);
    return success();
  }

  VectorType inVecType = cast<VectorType>(in.getType());
  Value origScale = getOriginalVectorValue(op.getScale());

  ArrayRef<int64_t> inShape = inVecType.getShape();
  SmallVector<int64_t> originalScaleShape;
  if (auto origScaleVecType = dyn_cast<VectorType>(origScale.getType()))
    llvm::append_range(originalScaleShape, origScaleVecType.getShape());

  originalScaleShape.insert(originalScaleShape.end(),
                            inShape.size() - originalScaleShape.size(), 1);

  auto maybeRatio = computeShapeRatio(inShape, originalScaleShape);
  assert(maybeRatio &&
         "failed to derive block size from broadcast or splat operation");

  SmallVector<int64_t> ratio =
      maybeRatio.value_or(SmallVector<int64_t>(inShape.size(), 1));

  int64_t blockSize = computeProduct(ratio);

  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outType, rewriter.getFloatAttr(outType, 0.0));
  Value result =
      rewriter.createOrFold<vector::BroadcastOp>(loc, outVecType, zero);

  for (SmallVector<int64_t> offsets : StaticTileOffsetRange(inShape, ratio)) {
    SmallVector<int64_t> strides(offsets.size(), 1);
    Value block = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, in, offsets, ratio, strides);
    VectorType block1DType = VectorType::get(blockSize, inType);
    Value block1D =
        rewriter.create<vector::ShapeCastOp>(loc, block1DType, block);
    Value uniformScale =
        rewriter.create<vector::ExtractOp>(loc, scale, offsets);

    VectorType blockResultType = VectorType::get(blockSize, outType);
    Value blockResult =
        rewriter.createOrFold<vector::BroadcastOp>(loc, blockResultType, zero);

    for (int64_t i = 0, sliceWidth = std::min(opWidth, blockSize - i);
         i < blockSize;
         i += sliceWidth, sliceWidth = std::min(opWidth, blockSize - i)) {
      Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, block1D, i, sliceWidth, 1);
      // TODO: replace this with non-packed ScaledExtOp for sliceWidth == 1
      Value scaleExt = rewriter.create<amdgpu::ScaledExtPackedOp>(
          loc, extScaleResultType, slice, uniformScale, 0);
      if (sliceWidth != opWidth)
        scaleExt = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, scaleExt, 0, sliceWidth, 1);
      blockResult = rewriter.create<vector::InsertStridedSliceOp>(
          loc, scaleExt, blockResult, i, 1);
    }

    VectorType resultType = VectorType::get(ratio, outType);
    Value cast =
        rewriter.create<vector::ShapeCastOp>(loc, resultType, blockResult);
    result = rewriter.create<vector::InsertStridedSliceOp>(loc, cast, result,
                                                           offsets, strides);
  }

  rewriter.replaceOp(op, result);

  return success();
}

LogicalResult
ScalingTruncFRewritePattern::matchAndRewrite(arith::ScalingTruncFOp op,
                                             PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  constexpr int64_t opWidth = 2;

  Value in = op.getIn();
  Value scale = op.getScale();
  Value out = op.getOut();

  Type f32 = rewriter.getF32Type();
  Type inType = getElementTypeOrSelf(in);
  Type scaleType = getElementTypeOrSelf(scale);
  Type outType = getElementTypeOrSelf(out);

  VectorType outVecType = dyn_cast<VectorType>(out.getType());
  VectorType scaleVecType = dyn_cast<VectorType>(scale.getType());

  if (outVecType && outVecType.isScalable())
    return failure();

  Type scaleF32Type =
      scaleVecType ? VectorType::get(scaleVecType.getShape(), f32) : f32;
  if (scaleType.getIntOrFloatBitWidth() < 32)
    scale = rewriter.create<arith::ExtFOp>(loc, scaleF32Type, scale);
  else if (scaleType.getIntOrFloatBitWidth() > 32)
    scale = rewriter.create<arith::TruncFOp>(loc, scaleF32Type, scale);

  Value zero = rewriter.create<arith::ConstantOp>(
      loc, outType, rewriter.getFloatAttr(outType, 0.0));
  unsigned numPackedElem = 32 / outType.getIntOrFloatBitWidth();
  VectorType truncScaleResultType = VectorType::get(numPackedElem, outType);

  if (!outVecType) {
    Type inVecType = VectorType::get(1, inType);
    Value inCast = rewriter.create<vector::BroadcastOp>(loc, inVecType, in);
    // TODO: replace this with non-packed ScaledTruncOp
    Value scaleTrunc = rewriter.create<amdgpu::PackedScaledTruncOp>(
        loc, truncScaleResultType, inCast, scale, 0, /*existing=*/nullptr);
    scaleTrunc =
        rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, scaleTrunc, 0);
    return success();
  }

  VectorType inVecType = cast<VectorType>(in.getType());
  Value origScale = getOriginalVectorValue(op.getScale());

  ArrayRef<int64_t> inShape = inVecType.getShape();
  SmallVector<int64_t> originalScaleShape;
  if (auto origScaleVecType = dyn_cast<VectorType>(origScale.getType()))
    llvm::append_range(originalScaleShape, origScaleVecType.getShape());

  originalScaleShape.insert(originalScaleShape.end(),
                            inShape.size() - originalScaleShape.size(), 1);

  auto maybeRatio = computeShapeRatio(inShape, originalScaleShape);
  assert(maybeRatio &&
         "failed to derive block size from broadcast or splat operation");

  SmallVector<int64_t> ratio =
      maybeRatio.value_or(SmallVector<int64_t>(inShape.size(), 1));

  int64_t blockSize = computeProduct(ratio);

  Value result =
      rewriter.createOrFold<vector::BroadcastOp>(loc, outVecType, zero);

  for (SmallVector<int64_t> offsets : StaticTileOffsetRange(inShape, ratio)) {
    SmallVector<int64_t> strides(offsets.size(), 1);
    Value block = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, in, offsets, ratio, strides);
    VectorType block1DType = VectorType::get(blockSize, inType);
    Value block1D =
        rewriter.create<vector::ShapeCastOp>(loc, block1DType, block);
    Value uniformScale =
        rewriter.create<vector::ExtractOp>(loc, scale, offsets);

    VectorType blockResultType = VectorType::get(blockSize, outType);
    Value blockResult =
        rewriter.createOrFold<vector::BroadcastOp>(loc, blockResultType, zero);

    for (int64_t i = 0, sliceWidth = std::min(opWidth, blockSize - i);
         i < blockSize;
         i += sliceWidth, sliceWidth = std::min(opWidth, blockSize - i)) {
      Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, block1D, i, sliceWidth, 1);
      // TODO: replace this with non-packed ScaledTruncOp for sliceWidth == 1
      Value scaleTrunc = rewriter.create<amdgpu::PackedScaledTruncOp>(
          loc, truncScaleResultType, slice, uniformScale, 0,
          /*existing=*/nullptr);
      int64_t packedWidth =
          cast<VectorType>(scaleTrunc.getType()).getNumElements();
      if (packedWidth != opWidth)
        scaleTrunc = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, scaleTrunc, 0, sliceWidth, 1);
      blockResult = rewriter.create<vector::InsertStridedSliceOp>(
          loc, scaleTrunc, blockResult, i, 1);
    }

    VectorType resultType = VectorType::get(ratio, outType);
    Value cast =
        rewriter.create<vector::ShapeCastOp>(loc, resultType, blockResult);
    result = rewriter.create<vector::InsertStridedSliceOp>(loc, cast, result,
                                                           offsets, strides);
  }

  rewriter.replaceOp(op, result);

  return success();
}

void mlir::arith::populateArithToAMDGPUConversionPatterns(
    RewritePatternSet &patterns, bool convertFP8Arithmetic,
    bool saturateFP8Truncf, bool allowPackedF16Rtz, Chipset chipset) {

  if (convertFP8Arithmetic) {
    patterns.add<ExtFOnFloat8RewritePattern>(patterns.getContext(), chipset);
    patterns.add<TruncFToFloat8RewritePattern>(patterns.getContext(),
                                               saturateFP8Truncf, chipset);
  }
  if (allowPackedF16Rtz)
    patterns.add<TruncfToFloat16RewritePattern>(patterns.getContext());

  if (chipset >= kGfx950) {
    patterns.add<ScalingExtFRewritePattern>(patterns.getContext());
    patterns.add<ScalingTruncFRewritePattern>(patterns.getContext());
  }
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
      *maybeChipset == kGfx942 || hasOcpFp8(*maybeChipset);
  arith::populateArithToAMDGPUConversionPatterns(
      patterns, convertFP8Arithmetic, saturateFP8Truncf, allowPackedF16Rtz,
      *maybeChipset);
  if (failed(applyPatternsGreedily(op, std::move(patterns))))
    return signalPassFailure();
}
