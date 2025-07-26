//===- TosaCanonicalizations.cpp - Canonicalization patterns & folders ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// TOSA canonicalization patterns and folders.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

#include <functional>

using namespace mlir;
using namespace mlir::tosa;

//===----------------------------------------------------------------------===//
// Operator Canonicalizers.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Tensor Data Engine Operators.
//===----------------------------------------------------------------------===//

// Check that the zero point of the tensor and padding operations are aligned.
bool checkMatchingPadConstAndZp(Value padConst, Value zp) {
  // Check that padConst is a constant value and a scalar tensor
  DenseElementsAttr padConstAttr;
  if (!matchPattern(padConst, m_Constant(&padConstAttr)) ||
      (padConstAttr.size() != 1)) {
    return false;
  }

  // Check that floating point pad is zero
  if (auto padConstFpAttr = mlir::dyn_cast<DenseFPElementsAttr>(padConstAttr)) {
    float padConstVal = (*padConstFpAttr.begin()).convertToFloat();
    return padConstVal == 0.0f;
  }

  // Check that the zp and padConst align for the integer (quantized) case
  if (auto padConstIntAttr =
          mlir::dyn_cast<DenseIntElementsAttr>(padConstAttr)) {
    DenseIntElementsAttr zpAttr;
    // Check that zp is a constant value and a scalar tensor
    if (!matchPattern(zp, m_Constant(&zpAttr)) || (padConstAttr.size() != 1)) {
      return false;
    }

    // Check equality
    int64_t zpVal = (*zpAttr.begin()).getSExtValue();
    int64_t padConstVal = (*padConstIntAttr.begin()).getSExtValue();
    return zpVal == padConstVal;
  }

  // Bail-out on unsupported type
  return false;
}

namespace {
template <typename OpTy>
struct PoolPadFoldAdaptor;

template <>
struct PoolPadFoldAdaptor<tosa::AvgPool2dOp> {
  using OpTy = tosa::AvgPool2dOp;
  static bool checkKernelCompliance(OpTy op, const ArrayRef<int64_t> newPad) {
    const llvm::ArrayRef<int64_t> kernel = op.getKernel();
    if (newPad[2] >= kernel[1] || newPad[3] >= kernel[1] ||
        newPad[0] >= kernel[0] || newPad[1] >= kernel[0])
      return false;
    return true;
  }
  static bool checkPadConstCompliance(OpTy op, Value padConst) {
    return checkMatchingPadConstAndZp(padConst, op.getInputZp());
  }
  static void replaceOpWithNewPad(PatternRewriter &rewriter, OpTy op,
                                  Value padInput, ArrayRef<int64_t> newPad) {
    rewriter.replaceOpWithNewOp<tosa::AvgPool2dOp>(
        op, op.getType(), padInput, op.getInputZp(), op.getOutputZp(),
        op.getKernel(), op.getStride(), rewriter.getDenseI64ArrayAttr(newPad),
        op.getAccType());
  }
};

template <>
struct PoolPadFoldAdaptor<tosa::MaxPool2dOp> {
  using OpTy = tosa::MaxPool2dOp;
  static bool checkKernelCompliance(OpTy op, const ArrayRef<int64_t> newPad) {
    const llvm::ArrayRef<int64_t> kernel = op.getKernel();
    if (newPad[2] >= kernel[1] || newPad[3] >= kernel[1] ||
        newPad[0] >= kernel[0] || newPad[1] >= kernel[0])
      return false;
    return true;
  }
  static bool checkPadConstCompliance(OpTy, Value padConst) {
    // Check that padConst is a constant value and a scalar tensor
    DenseElementsAttr padConstAttr;
    if (!matchPattern(padConst, m_Constant(&padConstAttr)) ||
        padConstAttr.size() != 1) {
      return false;
    }

    // Pad needs to be in the minimum value to be able to merge
    if (auto padConstFpAttr =
            mlir::dyn_cast<DenseFPElementsAttr>(padConstAttr)) {
      const APFloat padConstVal = *padConstFpAttr.begin();
      const APFloat lowestVal =
          APFloat::getLargest(padConstVal.getSemantics(), true);
      return padConstVal == lowestVal;
    } else if (auto padConstIntAttr =
                   mlir::dyn_cast<DenseIntElementsAttr>(padConstAttr)) {
      const APInt padConstVal = *padConstIntAttr.begin();
      const unsigned int bitWidth = padConstVal.getBitWidth();
      const APInt lowestVal =
          padConstIntAttr.getElementType().isUnsignedInteger()
              ? APInt::getZero(bitWidth)
              : APInt::getSignedMinValue(bitWidth);
      return padConstVal == lowestVal;
    }

    // Bail-out on unsupported type
    return false;
  }
  static void replaceOpWithNewPad(PatternRewriter &rewriter, OpTy op,
                                  Value padInput, ArrayRef<int64_t> newPad) {
    rewriter.replaceOpWithNewOp<tosa::MaxPool2dOp>(
        op, op.getType(), padInput, op.getKernel(), op.getStride(),
        rewriter.getDenseI64ArrayAttr(newPad), op.getNanMode());
  }
};

template <typename OpTy>
struct ConvPadFoldAdaptor {
  static bool checkKernelCompliance(OpTy, const ArrayRef<int64_t>) {
    return true;
  }
  static bool checkPadConstCompliance(OpTy op, Value padConst) {
    return checkMatchingPadConstAndZp(padConst, op.getInputZp());
  }
  static void replaceOpWithNewPad(PatternRewriter &rewriter, OpTy op,
                                  Value padInput, ArrayRef<int64_t> newPad) {
    rewriter.replaceOpWithNewOp<OpTy>(
        op, op.getResult().getType(), padInput, op.getWeight(), op.getBias(),
        op.getInputZp(), op.getWeightZp(), newPad, op.getStrideAttr(),
        op.getDilationAttr(), op.getAccType(), op.getLocalBound());
  }
};

// Pattern attempts to fold a `tosa.pad` operator to a following tensor
// operation like `tosa.conv2d` by merging the padding associated with the
// pad operator directly to the implicit padding of the tensor operation.
// This helps eliminate the explicit padding operator if unused.
template <typename OpTy, typename AdaptorTy>
struct FoldPadToTensorOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy tensorOp,
                                PatternRewriter &rewriter) const override {
    // Check producer is a tosa::PadOp
    auto padOp = tensorOp.getInput().template getDefiningOp<tosa::PadOp>();
    if (!padOp)
      return rewriter.notifyMatchFailure(tensorOp,
                                         "Producer must be a tosa::PadOp.");

    // Validate that tensor operation has sane padding
    const std::vector<int64_t> &tensorOpPad = tensorOp.getPad().vec();
    if (tensorOpPad.size() != 4) // pad_top, pad_bottom, pad_left, pad_right
      return rewriter.notifyMatchFailure(
          tensorOp, "Tensor operation padding shall have 4 elements.");

    // Validate tosa::PadOp padding
    DenseIntElementsAttr padOpPadding;
    if (!matchPattern(padOp.getPadding(), m_Constant(&padOpPadding))) {
      return rewriter.notifyMatchFailure(
          tensorOp,
          "The `padding` input specified on the tosa::PadOp must be constant.");
    }
    // N_before, N_after, H_before, H_after, W_before, W_after, C_before,
    // C_after
    if (padOpPadding.size() != 8)
      return rewriter.notifyMatchFailure(tensorOp,
                                         "Pad padding should have 8 elements.");
    int64_t padNBefore = (*(padOpPadding.begin() + 0)).getLimitedValue();
    int64_t padNAfter = (*(padOpPadding.begin() + 1)).getLimitedValue();
    int64_t padHBefore = (*(padOpPadding.begin() + 2)).getLimitedValue();
    int64_t padHAfter = (*(padOpPadding.begin() + 3)).getLimitedValue();
    int64_t padWBefore = (*(padOpPadding.begin() + 4)).getLimitedValue();
    int64_t padWAfter = (*(padOpPadding.begin() + 5)).getLimitedValue();
    int64_t padCBefore = (*(padOpPadding.begin() + 6)).getLimitedValue();
    int64_t padCAfter = (*(padOpPadding.begin() + 7)).getLimitedValue();

    if (padNBefore != 0 || padNAfter != 0 || padCBefore != 0 || padCAfter != 0)
      return rewriter.notifyMatchFailure(
          tensorOp, "Folding padding in N or C dimensions is not supported.");

    // Fold padding from Pad into the tensor operation
    // 4 elements - pad_top, pad_bottom, pad_left, pad_right
    SmallVector<int64_t> foldedPad(tensorOpPad.size());
    foldedPad[0] = padHBefore + tensorOpPad[0];
    foldedPad[1] = padHAfter + tensorOpPad[1];
    foldedPad[2] = padWBefore + tensorOpPad[2];
    foldedPad[3] = padWAfter + tensorOpPad[3];

    // Check kernel related restrictions
    if (!AdaptorTy::checkKernelCompliance(tensorOp, foldedPad)) {
      return rewriter.notifyMatchFailure(
          tensorOp, "Padding size not aligned with kernel restrictions.");
    }

    // Check padding constant restrictions
    if (!AdaptorTy::checkPadConstCompliance(tensorOp, padOp.getPadConst())) {
      return rewriter.notifyMatchFailure(
          tensorOp,
          "Padding constant is not aligned with operator zero-point.");
    }

    // Check that padding doesn't grow more than 8K level (8192) for now
    if (llvm::any_of(foldedPad, [](int64_t padVal) { return padVal > 8192; })) {
      return rewriter.notifyMatchFailure(
          tensorOp, "Padding size more than the 8K level limit.");
    }

    // Create operator
    AdaptorTy::replaceOpWithNewPad(rewriter, tensorOp, padOp.getInput1(),
                                   foldedPad);

    return success();
  }
};
} // namespace

void AvgPool2dOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<FoldPadToTensorOp<tosa::AvgPool2dOp,
                                PoolPadFoldAdaptor<tosa::AvgPool2dOp>>>(
      context);
}

void Conv2DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<
      FoldPadToTensorOp<tosa::Conv2DOp, ConvPadFoldAdaptor<tosa::Conv2DOp>>>(
      context);
}

void DepthwiseConv2DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<FoldPadToTensorOp<tosa::DepthwiseConv2DOp,
                                ConvPadFoldAdaptor<tosa::DepthwiseConv2DOp>>>(
      context);
}

struct MaxPool2dIsNoOp : public OpRewritePattern<tosa::MaxPool2dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MaxPool2dOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Value output = op.getOutput();
    ShapedType inputType = llvm::cast<ShapedType>(input.getType());
    ShapedType outputType = llvm::cast<ShapedType>(output.getType());

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
      return failure();
    }

    // If the output and input shapes are 1x1, then this is a no op.
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (outputShape[1] != 1 || outputShape[2] != 1) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    if (inputShape[1] != 1 || inputShape[2] != 1) {
      return failure();
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

void MaxPool2dOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<MaxPool2dIsNoOp,
              FoldPadToTensorOp<tosa::MaxPool2dOp,
                                PoolPadFoldAdaptor<tosa::MaxPool2dOp>>>(
      context);
}

//===----------------------------------------------------------------------===//
// Data Layout / Memory Reinterpretation.
//===----------------------------------------------------------------------===//

struct ConcatOptimization : public OpRewritePattern<tosa::ConcatOp> {
  using OpRewritePattern<tosa::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput1().size() != 1)
      return failure();
    if (op.getInput1().front().getType() != op.getType()) {
      rewriter
          .replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                              op.getInput1().front())
          .getResult();
      return success();
    }

    rewriter.replaceOp(op, op.getInput1().front());
    return success();
  }
};

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<ConcatOptimization>(context);
}

LogicalResult SelectOp::canonicalize(SelectOp op, PatternRewriter &rewriter) {
  auto notOp = op.getInput1().getDefiningOp<tosa::LogicalNotOp>();
  if (!notOp)
    return failure();
  rewriter.modifyOpInPlace(op, [&]() {
    op.getOperation()->setOperands(
        {notOp.getInput1(), op.getOnFalse(), op.getOnTrue()});
  });
  return success();
}

struct ConsolidateTransposeOptimization
    : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Input is also TransposeOp - transpose(transpose(A)).
    auto innerTranspose =
        transposeOp.getInput1().getDefiningOp<tosa::TransposeOp>();
    if (!innerTranspose)
      return rewriter.notifyMatchFailure(transposeOp,
                                         "input must be transpose operation");

    const llvm::ArrayRef<int32_t> transposePerms = transposeOp.getPerms();
    const llvm::ArrayRef<int32_t> innerTransposePerms =
        innerTranspose.getPerms();

    if (transposePerms.size() != innerTransposePerms.size())
      return rewriter.notifyMatchFailure(
          transposeOp,
          "transpose and inner transpose perms sizes must be equal");
    if (transposePerms.empty())
      return rewriter.notifyMatchFailure(
          transposeOp, "transpose perms sizes must be positive");

    // Consolidate transposes into one transpose.
    SmallVector<int32_t> perms(transposePerms.size());
    for (int i = 0, s = transposePerms.size(); i < s; ++i)
      perms[i] = innerTransposePerms[transposePerms[i]];

    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
        transposeOp, transposeOp.getResult().getType(),
        innerTranspose.getInput1(), rewriter.getDenseI32ArrayAttr(perms));

    return success();
  }
};

// Determines the case when tosa.transpose is a tosa.reshape operation.
struct TransposeIsReshape : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput1().getDefiningOp<tosa::TransposeOp>())
      return rewriter.notifyMatchFailure(
          op, "Src is from transpose, can compose transposes");

    Value result = op.getResult();
    for (Operation *subop : result.getUsers()) {
      if (isa_and_nonnull<tosa::TransposeOp>(subop))
        return rewriter.notifyMatchFailure(
            op, "Dest is used by transpose, can compose transposes");
    }

    auto input = op.getInput1();
    auto inputTy = llvm::cast<ShapedType>(input.getType());
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "Unranked input.");

    int64_t numDynDims = 0;
    for (int i = 0; i < inputTy.getRank(); ++i)
      if (inputTy.isDynamicDim(i))
        numDynDims++;

    if (numDynDims > 1)
      return rewriter.notifyMatchFailure(op, "Has more than one dynamic dim.");

    const llvm::ArrayRef<int32_t> permValues = op.getPerms();

    SmallVector<int64_t> nonZeroPerms;
    nonZeroPerms.reserve(permValues.size());
    for (auto idx : permValues) {
      auto sz = inputTy.getDimSize(idx);
      if (sz != 1)
        nonZeroPerms.push_back(idx);
    }

    for (int i = 1, s = nonZeroPerms.size(); i < s; ++i)
      if (nonZeroPerms[i - 1] > nonZeroPerms[i])
        return rewriter.notifyMatchFailure(op,
                                           "Transpose changes memory layout.");

    SmallVector<int64_t> newShape;
    newShape.reserve(inputTy.getRank());
    for (int i = 0, s = inputTy.getRank(); i < s; ++i)
      newShape.push_back(inputTy.getDimSize(permValues[i]));

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, op.getType(), op.getInput1(),
        getTosaConstShape(rewriter, op.getLoc(), newShape));
    return success();
  }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<ConsolidateTransposeOptimization, TransposeIsReshape>(context);
}

struct ClampIsNoOp : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
    auto inputElementType = inputType.getElementType();

    if (!inputType.hasStaticShape()) {
      return failure();
    }

    if (isa<FloatType>(inputElementType)) {
      // Unlike integer types, floating point types can represent infinity.
      auto minClamp =
          llvm::cast<mlir::FloatAttr>(op.getMinValAttr()).getValue();
      auto maxClamp =
          llvm::cast<mlir::FloatAttr>(op.getMaxValAttr()).getValue();
      bool isMin = minClamp.isNegInfinity();
      bool isMax = maxClamp.isInfinity();

      if (isMin && isMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    if (inputElementType.isUnsignedInteger()) {
      int64_t minClamp =
          llvm::cast<mlir::IntegerAttr>(op.getMinValAttr()).getUInt();
      int64_t maxClamp =
          llvm::cast<mlir::IntegerAttr>(op.getMaxValAttr()).getUInt();

      int64_t intMin =
          APInt::getMinValue(inputElementType.getIntOrFloatBitWidth())
              .getZExtValue();
      int64_t intMax =
          APInt::getMaxValue(inputElementType.getIntOrFloatBitWidth())
              .getZExtValue();

      if (minClamp <= intMin && maxClamp >= intMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    if (llvm::isa<IntegerType>(inputElementType)) {
      int64_t minClamp =
          llvm::cast<mlir::IntegerAttr>(op.getMinValAttr()).getInt();
      int64_t maxClamp =
          llvm::cast<mlir::IntegerAttr>(op.getMaxValAttr()).getInt();

      int64_t intMin =
          APInt::getSignedMinValue(inputElementType.getIntOrFloatBitWidth())
              .getSExtValue();
      int64_t intMax =
          APInt::getSignedMaxValue(inputElementType.getIntOrFloatBitWidth())
              .getSExtValue();

      if (minClamp <= intMin && maxClamp >= intMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    return failure();
  }
};

// Attempts the following transformation:
//
// For integers a, b, a', and b' such that [a, b] ∩ [a', b'] ≠ ∅ and input
// tensor X the following identity holds:
//
// CLAMP(CLAMP(X, a, b), a', b') = CLAMP(X, max(a, a'),  min(b, b'))
//
// subject to the following valid NaN propagation semantics:
// --------------------------------------------
// | OUTER CLAMP | INNER CLAMP  | RESULT MODE |
// |-------------|--------------|-------------|
// | PROPAGATE   | PROPAGATE    | PROPAGATE   |
// | PROPAGATE   | IGNORE       | IGNORE      |
// | IGNORE      | PROPAGATE    | INVALID     |
// | IGNORE      | IGNORE       | IGNORE      |
// |------------------------------------------|

struct ClampClampOptimization : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  // Helper structure to describe the range of a clamp operation.
  template <typename T>
  struct ClampRange {
    ClampRange(const T &start, const T &end) : start(start), end(end) {}
    T start;
    T end;

    // Helper function to determine if two Clamp ranges intersect.
    bool intersects(const ClampRange<T> &otherRange) {
      return start < otherRange.end && otherRange.start < end;
    }
  };

  LogicalResult matchAndRewrite(tosa::ClampOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();

    // Check the input to the CLAMP op is itself a CLAMP.
    auto clampOp = dyn_cast_if_present<tosa::ClampOp>(input.getDefiningOp());
    if (!clampOp)
      return failure();

    // Check we have a valid NaN propagation combination.
    const auto opNanMode = op.getNanMode();
    const auto clampNanMode = clampOp.getNanMode();
    if (opNanMode == "IGNORE" && clampNanMode == "PROPAGATE")
      return failure();

    auto maxValAttr = op.getMaxValAttr();
    auto minValAttr = op.getMinValAttr();
    auto clampOpMaxValAttr = clampOp.getMaxValAttr();
    auto clampOpMinValAttr = clampOp.getMinValAttr();

    auto inputEType = llvm::cast<ShapedType>(input.getType()).getElementType();
    if (auto quantType =
            llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputEType)) {
      inputEType = quantType.getStorageType();
    }

    Attribute newMinValAttr, newMaxValAttr;
    if (mlir::isa<FloatType>(inputEType)) {
      auto floatMaxValAttr = cast<mlir::FloatAttr>(maxValAttr);
      auto floatMinValAttr = cast<mlir::FloatAttr>(minValAttr);
      auto clampOpFloatMaxValAttr = cast<mlir::FloatAttr>(clampOpMaxValAttr);
      auto clampOpFloatMinValAttr = cast<mlir::FloatAttr>(clampOpMinValAttr);

      // Check we have intersecting ranges.
      const auto opMinFloat = floatMinValAttr.getValue();
      const auto opMaxFloat = floatMaxValAttr.getValue();
      const auto clampOpMinFloat = clampOpFloatMinValAttr.getValue();
      const auto clampOpMaxFloat = clampOpFloatMaxValAttr.getValue();
      ClampRange<APFloat> opRangeFloatRange(opMinFloat, opMaxFloat);
      ClampRange<APFloat> clampRangeFloatRange(clampOpMinFloat,
                                               clampOpMaxFloat);
      if (!opRangeFloatRange.intersects(clampRangeFloatRange))
        return failure();

      // Run the transformation.
      auto newMinVal = std::max(opMinFloat, clampOpMinFloat);
      auto newMaxVal = std::min(opMaxFloat, clampOpMaxFloat);
      newMinValAttr = rewriter.getFloatAttr(inputEType, newMinVal);
      newMaxValAttr = rewriter.getFloatAttr(inputEType, newMaxVal);
    } else {
      assert(mlir::isa<IntegerType>(inputEType));
      auto intMaxValAttr = cast<mlir::IntegerAttr>(maxValAttr);
      auto intMinValAttr = cast<mlir::IntegerAttr>(minValAttr);
      auto clampOpIntMaxValAttr = cast<mlir::IntegerAttr>(clampOpMaxValAttr);
      auto clampOpIntMinValAttr = cast<mlir::IntegerAttr>(clampOpMinValAttr);

      if (inputEType.isUnsignedInteger()) {
        // Check we have intersecting ranges.
        const auto opMinInt = intMinValAttr.getUInt();
        const auto opMaxInt = intMaxValAttr.getUInt();
        const auto clampOpMinInt = clampOpIntMinValAttr.getUInt();
        const auto clampOpMaxInt = clampOpIntMaxValAttr.getUInt();
        ClampRange<std::uint64_t> opRangeIntRange(opMinInt, opMaxInt);
        ClampRange<std::uint64_t> clampRangeIntRange(clampOpMinInt,
                                                     clampOpMaxInt);
        if (!opRangeIntRange.intersects(clampRangeIntRange))
          return failure();

        // Run the transformation.
        auto newMinVal = std::max(opMinInt, clampOpMinInt);
        auto newMaxVal = std::min(opMaxInt, clampOpMaxInt);
        newMinValAttr = rewriter.getIntegerAttr(inputEType, newMinVal);
        newMaxValAttr = rewriter.getIntegerAttr(inputEType, newMaxVal);
      } else {
        // Check we have intersecting ranges.
        const auto opMinInt = intMinValAttr.getInt();
        const auto opMaxInt = intMaxValAttr.getInt();
        const auto clampOpMinInt = clampOpIntMinValAttr.getInt();
        const auto clampOpMaxInt = clampOpIntMaxValAttr.getInt();
        ClampRange<std::int64_t> opRangeIntRange(opMinInt, opMaxInt);
        ClampRange<std::int64_t> clampRangeIntRange(clampOpMinInt,
                                                    clampOpMaxInt);
        if (!opRangeIntRange.intersects(clampRangeIntRange))
          return failure();

        // Run the transformation.
        auto newMinVal = std::max(opMinInt, clampOpMinInt);
        auto newMaxVal = std::min(opMaxInt, clampOpMaxInt);
        newMinValAttr = rewriter.getIntegerAttr(inputEType, newMinVal);
        newMaxValAttr = rewriter.getIntegerAttr(inputEType, newMaxVal);
      }
    }

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(
        op, op.getType(), clampOp.getInput(), newMinValAttr, newMaxValAttr,
        rewriter.getStringAttr((opNanMode != clampNanMode) ? "IGNORE"
                                                           : opNanMode));
    return success();
  }
};

void ClampOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<ClampIsNoOp>(context);
  results.add<ClampClampOptimization>(context);
}

struct ConcatSliceOptimization : public OpRewritePattern<tosa::SliceOp> {
  using OpRewritePattern<tosa::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    Value sliceInput = sliceOp.getInput1();
    auto concatOp = sliceInput.getDefiningOp<tosa::ConcatOp>();
    if (!concatOp)
      return rewriter.notifyMatchFailure(
          sliceOp, "slice input must be concat operation");

    OperandRange inputs = concatOp.getInput1();
    auto concatType = dyn_cast<RankedTensorType>(concatOp.getType());
    if (!concatType || !concatType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          sliceOp, "slice input must be a static ranked tensor");
    int32_t axis = concatOp.getAxis();

    DenseElementsAttr startElems;
    DenseElementsAttr sizeElems;

    if (!matchPattern(sliceOp.getStart(), m_Constant(&startElems)))
      return rewriter.notifyMatchFailure(
          sliceOp, "start of slice must be a static ranked shape");

    if (!matchPattern(sliceOp.getSize(), m_Constant(&sizeElems)))
      return rewriter.notifyMatchFailure(
          sliceOp, "size of slice must be a static ranked shape");

    llvm::SmallVector<int64_t> sliceStarts =
        llvm::to_vector(startElems.getValues<int64_t>());
    llvm::SmallVector<int64_t> sliceSizes =
        llvm::to_vector(sizeElems.getValues<int64_t>());

    // Validate slice on the concatenated axis. Slicing along this
    // axis should span only one of the inputs to the concatenate
    // operation.
    std::optional<Value> replaceWithSlice;
    for (auto input : inputs) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType || !inputType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            sliceOp, "concat input must be a static ranked tensor");

      if (sliceStarts[axis] >= 0 && (sliceStarts[axis] + sliceSizes[axis]) <=
                                        inputType.getDimSize(axis)) {
        auto start_op =
            getTosaConstShape(rewriter, sliceOp.getLoc(), sliceStarts);
        auto size_op =
            getTosaConstShape(rewriter, sliceOp.getLoc(), sliceSizes);
        replaceWithSlice =
            rewriter
                .create<tosa::SliceOp>(sliceOp.getLoc(), sliceOp.getType(),
                                       input, start_op, size_op)
                .getResult();
        break;
      }
      sliceStarts[axis] -= inputType.getDimSize(axis);
    }

    if (!replaceWithSlice)
      return rewriter.notifyMatchFailure(
          sliceOp, "corresponding concat input not found for slice");

    rewriter.replaceOp(sliceOp, replaceWithSlice.value());
    return success();
  }
};

struct PadSliceOptimization : public OpRewritePattern<tosa::SliceOp> {
  using OpRewritePattern<tosa::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    Value sliceInput = sliceOp.getInput1();

    // Check if producer is a PadOp
    auto padOp = sliceInput.getDefiningOp<tosa::PadOp>();
    if (!padOp)
      return rewriter.notifyMatchFailure(sliceOp,
                                         "slice input must be a pad operation");

    // Check PadOp has a single consumer
    if (!padOp->hasOneUse())
      return rewriter.notifyMatchFailure(sliceOp,
                                         "pad shall have a single consumer");

    // Check input is statically ranked
    auto inputTy = dyn_cast<RankedTensorType>(padOp.getInput1().getType());
    auto padTy = dyn_cast<RankedTensorType>(padOp.getType());
    if (!inputTy || !padTy || !inputTy.hasRank())
      return rewriter.notifyMatchFailure(sliceOp,
                                         "slice input must be a ranked tensor");

    // Validate and extract tosa::PadOp padding
    DenseIntElementsAttr paddingElems;
    if (!matchPattern(padOp.getPadding(), m_Constant(&paddingElems))) {
      return rewriter.notifyMatchFailure(
          sliceOp,
          "`padding` input specified on the tosa::PadOp must be constant.");
    }
    llvm::SmallVector<int64_t> padPaddings =
        llvm::to_vector(paddingElems.getValues<int64_t>());

    // Extract slice parameters
    DenseElementsAttr startElems;
    if (!matchPattern(sliceOp.getStart(), m_Constant(&startElems)))
      return rewriter.notifyMatchFailure(
          sliceOp, "start of slice must be a static ranked shape");
    llvm::SmallVector<int64_t> sliceStarts =
        llvm::to_vector(startElems.getValues<int64_t>());

    DenseElementsAttr sizeElems;
    if (!matchPattern(sliceOp.getSize(), m_Constant(&sizeElems)))
      return rewriter.notifyMatchFailure(
          sliceOp, "size of slice must be a static ranked shape");
    llvm::SmallVector<int64_t> sliceSizes =
        llvm::to_vector(sizeElems.getValues<int64_t>());

    // Check if dynamic dimensions are sliced
    const int64_t rank = inputTy.getRank();
    if (llvm::any_of(llvm::seq<int64_t>(0, rank), [&](int64_t i) {
          const bool isDimDynamic = inputTy.isDynamicDim(i);
          const bool isDimSliced =
              (sliceStarts[i] != 0) || (sliceSizes[i] != -1);

          return isDimDynamic && isDimSliced;
        })) {
      return rewriter.notifyMatchFailure(
          sliceOp, "axis that are sliced shall be statically known.");
    }

    // Update the parameters
    llvm::SmallVector<int64_t> newSliceStarts(rank, 0);
    llvm::SmallVector<int64_t> newPadPaddings(2 * rank, 0);
    llvm::SmallVector<int64_t> newPadShape(rank, ShapedType::kDynamic);
    bool updated = false;

    for (int64_t i = 0; i < rank; ++i) {
      const int64_t padLo = padPaddings[i * 2];
      const int64_t padHi = padPaddings[i * 2 + 1];
      const int64_t sliceStart = sliceStarts[i];
      const int64_t sliceSize = sliceSizes[i];
      const int64_t sliceEnd = sliceStart + sliceSize;

      // If dimension is dynamic pass-through
      if (inputTy.isDynamicDim(i)) {
        newPadPaddings[i * 2] = padLo;
        newPadPaddings[i * 2 + 1] = padHi;
        newSliceStarts[i] = sliceStart;
        continue;
      }

      // Handle static dimensions
      const int64_t dimSize = inputTy.getShape()[i];
      const int64_t dimTotal = padLo + dimSize + padHi;

      // Check slice within bounds
      if (sliceStart < 0 || sliceEnd > dimTotal)
        return rewriter.notifyMatchFailure(sliceOp, "slice is out-of-bounds");

      // Compute updated slice start parameter
      const int64_t newSliceStart = std::max<int64_t>(sliceStart - padLo, 0);
      newSliceStarts[i] = newSliceStart;
      updated |= newSliceStart != sliceStart;

      // Compute updated pad parameters
      const int64_t newPadLo = std::max<int64_t>(padLo - sliceStart, 0);
      const int64_t newPadHi =
          std::max<int64_t>(sliceEnd - (padLo + dimSize), 0);
      newPadPaddings[i * 2] = newPadLo;
      newPadPaddings[i * 2 + 1] = newPadHi;
      updated |= (newPadLo != padLo) || (newPadHi != padHi);

      // Calculate new pad output shape
      newPadShape[i] =
          newPadPaddings[i * 2] + dimSize + newPadPaddings[i * 2 + 1];
    }

    // Check that we actually need to proceed with the rewrite
    if (!updated)
      return rewriter.notifyMatchFailure(
          sliceOp, "terminate condition; nothing to rewrite");

    // Create a PadOp with updated padding
    auto newPaddingsOp =
        getTosaConstShape(rewriter, sliceOp.getLoc(), newPadPaddings);
    auto newPadTy =
        RankedTensorType::get(newPadShape, inputTy.getElementType());
    auto newPadOp = tosa::PadOp::create(rewriter, padOp.getLoc(), newPadTy,
                                        padOp.getInput1(), newPaddingsOp,
                                        padOp.getPadConst());

    // Update SliceOp and point to new PadOp
    auto newStartOp =
        getTosaConstShape(rewriter, sliceOp.getLoc(), newSliceStarts);
    rewriter.replaceOpWithNewOp<tosa::SliceOp>(sliceOp, sliceOp.getType(),
                                               newPadOp.getResult(), newStartOp,
                                               sliceOp.getSize());

    return success();
  }
};

// Update size operand of tosa.slice if size has dynamic dims but corresponding
// output dim is static
struct SliceDynamicSizeCanonicalization
    : public OpRewritePattern<tosa::SliceOp> {
  using OpRewritePattern<tosa::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    ShapedType resultType = cast<ShapedType>(sliceOp.getType());

    ElementsAttr sizeElems;
    if (!matchPattern(sliceOp.getSize(), m_Constant(&sizeElems))) {
      return rewriter.notifyMatchFailure(
          sliceOp, "size of slice must be a static ranked shape");
    }

    llvm::SmallVector<int64_t> sliceSizes =
        llvm::to_vector(sizeElems.getValues<int64_t>());

    bool replaceSliceSize{false};
    // if size op has -1 indicating dynamic shape but corresponding dim on the
    // output is statically known, update size to match with known output dim
    // shape
    for (const auto &[index, size] : llvm::enumerate(sliceSizes)) {
      if (size == -1 && !resultType.isDynamicDim(index)) {
        sliceSizes[index] = resultType.getDimSize(index);
        replaceSliceSize = true;
      }
    }

    if (!replaceSliceSize) {
      return rewriter.notifyMatchFailure(
          sliceOp, "no dimension of size of slice is dynamic that resolves "
                   "to static output shape");
    }

    auto size_op = getTosaConstShape(rewriter, sliceOp.getLoc(), sliceSizes);
    auto newSliceOp =
        tosa::SliceOp::create(rewriter, sliceOp.getLoc(), sliceOp.getType(),
                              sliceOp.getInput1(), sliceOp.getStart(), size_op);

    rewriter.replaceOp(sliceOp, newSliceOp.getResult());
    return success();
  }
};

void SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<ConcatSliceOptimization, PadSliceOptimization,
              SliceDynamicSizeCanonicalization>(context);
}

//===----------------------------------------------------------------------===//
// Operator Folders.
//===----------------------------------------------------------------------===//

template <typename IntFolder, typename FloatFolder>
DenseElementsAttr binaryFolder(DenseElementsAttr lhs, DenseElementsAttr rhs,
                               RankedTensorType returnTy) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    auto lETy = llvm::cast<ShapedType>(lhs.getType()).getElementType();
    auto rETy = llvm::cast<ShapedType>(rhs.getType()).getElementType();
    if (lETy != rETy)
      return {};

    if (llvm::isa<IntegerType>(lETy)) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();
      auto result = IntFolder()(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }

    if (llvm::isa<FloatType>(lETy)) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      auto result = FloatFolder()(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }
  }

  return {};
}

static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  if (llvm::isa<IntegerType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  return false;
}

static bool isSplatOne(Type elemType, DenseElementsAttr val, int64_t shift) {
  if (llvm::isa<FloatType>(elemType))
    return val && val.isSplat() &&
           val.getSplatValue<APFloat>().isExactlyValue(1.0);
  if (llvm::isa<IntegerType>(elemType)) {
    const int64_t shifted = 1LL << shift;
    return val && val.isSplat() &&
           val.getSplatValue<APInt>().getSExtValue() == shifted;
  }
  return false;
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(getInput2().getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};

  // Cannot create an ElementsAttr from non-int/float/index types
  if (!lhsTy.getElementType().isIntOrIndexOrFloat() ||
      !rhsTy.getElementType().isIntOrIndexOrFloat())
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (lhsTy == resultTy && isSplatZero(resultETy, rhsAttr))
    return getInput1();
  if (rhsTy == resultTy && isSplatZero(resultETy, lhsAttr))
    return getInput2();

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<std::plus<APInt>, std::plus<APFloat>>(lhsAttr, rhsAttr,
                                                            resultTy);
}

OpFoldResult ArgMaxOp::fold(FoldAdaptor adaptor) {
  auto inputTy = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto outputTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
      !outputTy.hasStaticShape())
    return {};

  if (inputTy.getDimSize(getAxis()) == 1)
    return DenseElementsAttr::get(outputTy, 0);

  return {};
}

OpFoldResult IntDivOp::fold(FoldAdaptor adaptor) {
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(getInput2().getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};
  if (lhsTy != rhsTy)
    return {};

  // IntDivOp inputs must be integer type, no need to check for quantized type
  auto resultETy = resultTy.getElementType();
  auto lhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());
  if (lhsAttr && lhsAttr.isSplat()) {
    if (llvm::isa<IntegerType>(resultETy) &&
        lhsAttr.getSplatValue<APInt>().isZero())
      return lhsAttr;
  }

  if (rhsAttr && rhsAttr.isSplat()) {
    if (llvm::isa<IntegerType>(resultETy) &&
        rhsAttr.getSplatValue<APInt>().isOne())
      return getInput1();
  }

  if (rhsAttr && lhsAttr && rhsAttr.isSplat() && lhsAttr.isSplat() &&
      llvm::isa<IntegerType>(resultETy)) {
    APInt l = lhsAttr.getSplatValue<APInt>();
    APInt r = rhsAttr.getSplatValue<APInt>();
    if (!r.isZero()) {
      APInt result = l.sdiv(r);
      return DenseElementsAttr::get(resultTy, result);
    }
  }

  return {};
}

namespace {
// calculate lhs * rhs >> shift according to TOSA Spec
// return nullopt if result is not in range of int32_t when shift > 0
std::optional<APInt> mulInt(APInt lhs, APInt rhs, int32_t shift,
                            unsigned bitwidth) {
  APInt result = lhs.sext(64) * rhs.sext(64);

  if (shift > 0) {
    auto round = APInt(64, 1) << (shift - 1);
    result += round;
    result.ashrInPlace(shift);
    // REQUIRE(product >= minimum_s<i32_t>() && product <= maximum_s<i32_t>())
    if (!(result.getSExtValue() >= INT32_MIN &&
          result.getSExtValue() <= INT32_MAX)) {
      // REQUIRE failed
      return std::nullopt;
    }
  }

  return result.trunc(bitwidth);
}

DenseElementsAttr mulBinaryFolder(DenseElementsAttr lhs, DenseElementsAttr rhs,
                                  RankedTensorType ty, int32_t shift) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    if (llvm::isa<IntegerType>(ty.getElementType())) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();

      if (shift == 0) {
        return DenseElementsAttr::get(ty, l * r);
      }

      auto bitwidth = ty.getElementType().getIntOrFloatBitWidth();
      const std::optional<APInt> result = mulInt(l, r, shift, bitwidth);
      if (!result)
        return {};
      return DenseElementsAttr::get(ty, result.value());
    }

    if (llvm::isa<FloatType>(ty.getElementType())) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      APFloat result = l * r;
      return DenseElementsAttr::get(ty, result);
    }
  }

  return {};
}
} // namespace

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = getInput1();
  auto rhs = getInput2();
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  // Result right shift on i32_t data type only. For simplification, synthesize
  // a zero shift for other data type.
  int32_t shift = 0;
  if (resultETy.isInteger(32)) {
    ElementsAttr shift_elem;
    if (getShift().getImpl()) {
      if (!matchPattern(getShift(), m_Constant(&shift_elem)))
        // cannot be folded when the shift value is unknown.
        return {};
      shift = shift_elem.getValues<IntegerAttr>()[0].getInt();
    }
  }

  if (rhsTy == resultTy) {
    if (isSplatZero(resultETy, lhsAttr))
      return lhsAttr.resizeSplat(resultTy);
    if (isSplatOne(resultETy, lhsAttr, shift))
      return rhs;
  }
  if (lhsTy == resultTy) {
    if (isSplatZero(resultETy, rhsAttr))
      return rhsAttr.resizeSplat(resultTy);
    if (isSplatOne(resultETy, rhsAttr, shift))
      return lhs;
  }

  return mulBinaryFolder(lhsAttr, rhsAttr, resultTy, shift);
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(getInput2().getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};

  // Cannot create an ElementsAttr from non-int/float/index types
  if (!lhsTy.getElementType().isIntOrIndexOrFloat() ||
      !rhsTy.getElementType().isIntOrIndexOrFloat())
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (lhsTy == resultTy && isSplatZero(resultETy, rhsAttr))
    return getInput1();

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<std::minus<APInt>, std::minus<APFloat>>(lhsAttr, rhsAttr,
                                                              resultTy);
}

namespace {
template <typename Cmp>
struct ComparisonFold {
  ComparisonFold() = default;
  APInt operator()(const APInt &l, const APInt &r) {
    return APInt(1, Cmp()(l, r));
  }

  APInt operator()(const APFloat &l, const APFloat &r) {
    return APInt(1, Cmp()(l, r));
  }
};

struct APIntFoldGreater {
  APIntFoldGreater() = default;
  APInt operator()(const APInt &l, const APInt &r) {
    return APInt(1, l.sgt(r));
  }
};

struct APIntFoldGreaterEqual {
  APIntFoldGreaterEqual() = default;
  APInt operator()(const APInt &l, const APInt &r) {
    return APInt(1, l.sge(r));
  }
};
} // namespace

OpFoldResult GreaterOp::fold(FoldAdaptor adaptor) {
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  auto lhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<APIntFoldGreater, ComparisonFold<std::greater<APFloat>>>(
      lhsAttr, rhsAttr, resultTy);
}

OpFoldResult GreaterEqualOp::fold(FoldAdaptor adaptor) {
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  auto lhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<APIntFoldGreaterEqual,
                      ComparisonFold<std::greater_equal<APFloat>>>(
      lhsAttr, rhsAttr, resultTy);
}

OpFoldResult EqualOp::fold(FoldAdaptor adaptor) {
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  auto lhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());
  Value lhs = getInput1();
  Value rhs = getInput2();
  auto lhsTy = llvm::cast<ShapedType>(lhs.getType());

  // If we are comparing an integer value to itself it is always true. We can
  // not do this with float due to float values.
  if (llvm::isa<IntegerType>(lhsTy.getElementType()) && resultTy &&
      resultTy.hasStaticShape() && lhs == rhs) {
    return DenseElementsAttr::get(resultTy, true);
  }

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<ComparisonFold<std::equal_to<APInt>>,
                      ComparisonFold<std::equal_to<APFloat>>>(lhsAttr, rhsAttr,
                                                              resultTy);
}

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getType())
    return getInput();

  auto operand = llvm::dyn_cast_if_present<ElementsAttr>(adaptor.getInput());
  if (!operand)
    return {};

  auto inTy = llvm::cast<ShapedType>(getInput().getType());
  auto outTy = llvm::cast<ShapedType>(getType());
  auto inETy = inTy.getElementType();
  auto outETy = outTy.getElementType();

  if (operand.isSplat()) {
    if (llvm::isa<FloatType>(inETy) && llvm::isa<FloatType>(outETy)) {
      bool overflow;
      auto splatVal = operand.getSplatValue<APFloat>();
      auto &semantics = llvm::cast<FloatType>(outETy).getFloatSemantics();
      splatVal.convert(semantics, llvm::RoundingMode::NearestTiesToEven,
                       &overflow);
      return SplatElementsAttr::get(outTy, splatVal);
    }

    if (llvm::isa<IntegerType>(inETy) && llvm::isa<FloatType>(outETy)) {
      auto unsign = llvm::cast<IntegerType>(inETy).isUnsignedInteger();
      APFloat splatVal(llvm::cast<FloatType>(outETy).getFloatSemantics());
      splatVal.convertFromAPInt(operand.getSplatValue<APInt>(), !unsign,
                                llvm::RoundingMode::NearestTiesToEven);
      return SplatElementsAttr::get(outTy, splatVal);
    }

    if (llvm::isa<FloatType>(inETy) && llvm::isa<IntegerType>(outETy)) {
      auto unsign = llvm::cast<IntegerType>(outETy).isUnsignedInteger();
      auto intVal = APSInt(
          llvm::cast<IntegerType>(outETy).getIntOrFloatBitWidth(), unsign);
      auto floatVal = operand.getSplatValue<APFloat>();
      bool exact;
      floatVal.convertToInteger(intVal, llvm::RoundingMode::NearestTiesToEven,
                                &exact);
      return SplatElementsAttr::get(outTy, intVal);
    }

    if (llvm::isa<IntegerType>(inETy) && llvm::isa<IntegerType>(outETy)) {
      const auto inIntType = llvm::cast<IntegerType>(inETy);
      auto unsignIn = inIntType.isUnsignedInteger();
      bool trunc =
          inETy.getIntOrFloatBitWidth() > outETy.getIntOrFloatBitWidth();
      auto intVal = operand.getSplatValue<APInt>();
      auto bitwidth = outETy.getIntOrFloatBitWidth();

      if (trunc) {
        intVal = intVal.trunc(bitwidth);
        // i1 types are boolean in TOSA
      } else if (unsignIn || inIntType.isInteger(1)) {
        intVal = intVal.zext(bitwidth);
      } else {
        intVal = intVal.sext(bitwidth);
      }

      return SplatElementsAttr::get(outTy, intVal);
    }
  }

  return {};
}

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) { return getValuesAttr(); }

OpFoldResult ConstShapeOp::fold(FoldAdaptor adaptor) { return getValuesAttr(); }

#define REDUCE_FOLDER(OP)                                                      \
  OpFoldResult OP::fold(FoldAdaptor adaptor) {                                 \
    ShapedType inputTy = llvm::cast<ShapedType>(getInput().getType());         \
    if (!inputTy.hasRank())                                                    \
      return {};                                                               \
    if (inputTy != getType())                                                  \
      return {};                                                               \
    if (inputTy.getRank() == 0 || inputTy.getDimSize(getAxis()) == 1)          \
      return getInput();                                                       \
    return {};                                                                 \
  }

REDUCE_FOLDER(ReduceAllOp)
REDUCE_FOLDER(ReduceAnyOp)
REDUCE_FOLDER(ReduceMaxOp)
REDUCE_FOLDER(ReduceMinOp)
REDUCE_FOLDER(ReduceProductOp)
REDUCE_FOLDER(ReduceSumOp)
#undef REDUCE_FOLDER

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto inputTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto outputTy = llvm::dyn_cast<RankedTensorType>(getType());

  if (!inputTy || !outputTy)
    return {};

  // Fold when the input and output types are the same. This is only safe when
  // there is at most 1 dynamic dimension. For 2 or more dynamic dimensions,
  // there may still be a productive reshape.
  if (inputTy == outputTy && inputTy.getNumDynamicDims() < 2)
    return getInput1();

  // reshape(reshape(x)) -> reshape(x)
  if (auto reshapeOp = llvm::dyn_cast_if_present<tosa::ReshapeOp>(
          getInput1().getDefiningOp())) {
    getInput1Mutable().assign(reshapeOp.getInput1());
    return getResult();
  }

  // Cannot create an ElementsAttr from non-int/float/index types
  if (!inputTy.getElementType().isIntOrIndexOrFloat())
    return {};

  // reshape(const(x)) -> const(reshape-attr(x))
  if (auto operand =
          llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1())) {
    // Constants must have static shape.
    if (!outputTy.hasStaticShape())
      return {};

    // Okay to duplicate splat constants.
    if (operand.isSplat())
      return SplatElementsAttr::get(outputTy,
                                    operand.getSplatValue<Attribute>());

    // Don't duplicate other constants.
    if (!getInput1().hasOneUse())
      return {};

    llvm::SmallVector<int64_t> shapeVec;
    if (!tosa::getConstShapeValues(getShape().getDefiningOp(), shapeVec))
      return {};

    return operand.reshape(
        llvm::cast<ShapedType>(operand.getType()).clone(shapeVec));
  }

  return {};
}

OpFoldResult PadOp::fold(FoldAdaptor adaptor) {
  // If the pad is all zeros we can fold this operation away.
  if (adaptor.getPadding() && getInput1().getType() == getType()) {
    auto densePad = llvm::dyn_cast<DenseElementsAttr>(adaptor.getPadding());
    if (densePad && densePad.isSplat() &&
        densePad.getSplatValue<APInt>().isZero()) {
      return getInput1();
    }
  }

  return {};
}

// Fold away cases where a tosa.resize operation returns a copy
// of the input image.
OpFoldResult ResizeOp::fold(FoldAdaptor adaptor) {
  auto scaleAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getScale());
  auto offsetAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getOffset());
  auto borderAttr =
      llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getBorder());
  if (!scaleAttr || !offsetAttr || !borderAttr) {
    return {};
  }

  auto scale = tosa::convertFromIntAttr(scaleAttr, /* rank = */ 4);
  auto offset = tosa::convertFromIntAttr(offsetAttr, /* rank = */ 2);
  auto border = tosa::convertFromIntAttr(borderAttr, /* rank = */ 2);
  if (scale.size() != 4 || offset.size() != 2 || border.size() != 2) {
    return {};
  }

  // Check unit scaling.
  if (scale[0] != scale[1] || scale[2] != scale[3]) {
    return {};
  }

  // There should be no offset.
  if (offset[0] != 0 || offset[1] != 0) {
    return {};
  }

  // There should be no border.
  if (border[0] != 0 || border[1] != 0) {
    return {};
  }

  auto input = getInput();
  auto inputTy = llvm::cast<RankedTensorType>(input.getType());
  auto resultTy = llvm::cast<RankedTensorType>(getType());
  if (inputTy != resultTy)
    return {};

  return input;
}

OpFoldResult ReverseOp::fold(FoldAdaptor adaptor) {
  auto operand = getInput1();
  auto operandTy = llvm::cast<ShapedType>(operand.getType());
  auto axis = getAxis();
  auto operandAttr =
      llvm::dyn_cast_if_present<SplatElementsAttr>(adaptor.getInput1());
  if (operandAttr)
    return operandAttr;

  // If the dim-length is 1, tosa.reverse is a no-op.
  if (operandTy.hasRank() &&
      (operandTy.getRank() == 0 || operandTy.getDimSize(axis) == 1))
    return operand;

  return {};
}

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  auto inputTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto outputTy = llvm::dyn_cast<RankedTensorType>(getType());

  if (!inputTy || !outputTy)
    return {};

  if (inputTy == outputTy && inputTy.hasStaticShape())
    return getInput1();

  if (!adaptor.getInput1())
    return {};

  // Cannot create an ElementsAttr from non-int/float/index types
  if (!inputTy.getElementType().isIntOrIndexOrFloat() ||
      !outputTy.getElementType().isIntOrIndexOrFloat())
    return {};

  auto operand = llvm::cast<ElementsAttr>(adaptor.getInput1());
  if (operand.isSplat() && outputTy.hasStaticShape()) {
    return SplatElementsAttr::get(outputTy, operand.getSplatValue<Attribute>());
  }

  if (inputTy.hasStaticShape() && outputTy.hasStaticShape() &&
      outputTy.getNumElements() == 1) {
    DenseElementsAttr startElems;
    if (!matchPattern(getStart(), m_Constant(&startElems)))
      return {};

    llvm::SmallVector<uint64_t> indices =
        llvm::to_vector(startElems.getValues<uint64_t>());
    auto value = operand.getValues<Attribute>()[indices];
    return SplatElementsAttr::get(outputTy, value);
  }

  return {};
}

OpFoldResult tosa::SelectOp::fold(FoldAdaptor adaptor) {
  if (getOnTrue() == getOnFalse())
    return getOnTrue();

  auto predicate =
      llvm::dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getInput1());
  if (!predicate)
    return {};

  if (!predicate.isSplat())
    return {};
  return predicate.getSplatValue<APInt>().getBoolValue() ? getOnTrue()
                                                         : getOnFalse();
}

OpFoldResult TileOp::fold(FoldAdaptor adaptor) {
  if (getInput1().getType() == getType()) {
    if (auto multiples = llvm::dyn_cast_if_present<DenseElementsAttr>(
            adaptor.getMultiples())) {
      if (multiples.isSplat() &&
          multiples.getSplatValue<APInt>().getSExtValue() == 1)
        return getInput1();
      if (auto int_array_attr =
              llvm::dyn_cast<DenseIntElementsAttr>(multiples)) {
        if (llvm::all_of(int_array_attr.getValues<APInt>(),
                         [](APInt v) { return v.getSExtValue() == 1; }))
          return getInput1();
      }
    }
  }
  return {};
}

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto resultTy = llvm::cast<ShapedType>(getType());

  // Transposing splat values just means reshaping.
  if (auto input =
          llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1())) {
    if (input.isSplat() && resultTy.hasStaticShape() &&
        input.getType().getElementType() == resultTy.getElementType())
      return input.reshape(resultTy);
  }

  // Transpose is not the identity transpose.
  const llvm::ArrayRef<int32_t> perms = getPerms();

  if (!llvm::equal(llvm::seq<int32_t>(0, perms.size()), perms))
    return {};

  return getInput1();
}

OpFoldResult tosa::LogOp::fold(FoldAdaptor adaptor) {
  auto input = getInput1();
  // Element-wise log(exp(x)) = x
  if (auto op = input.getDefiningOp<tosa::ExpOp>()) {
    return op.getInput1();
  }

  return {};
}

OpFoldResult tosa::ExpOp::fold(FoldAdaptor adaptor) {
  auto input = getInput1();
  // Element-wise exp(log(x)) = x
  if (auto op = input.getDefiningOp<tosa::LogOp>()) {
    return op.getInput1();
  }

  return {};
}

OpFoldResult tosa::NegateOp::fold(FoldAdaptor adaptor) {
  // Element-wise negate(negate(x)) = x
  // iff all zero points are constant 0
  auto definingOp = getInput1().getDefiningOp<tosa::NegateOp>();
  if (!definingOp) {
    // defining op of input1 is not a negate, cannot fold
    return {};
  }

  if (FailureOr<int64_t> maybeIZp = getInput1ZeroPoint();
      failed(maybeIZp) || *maybeIZp != 0) {
    // input1 zero point is not constant 0, cannot fold
    return {};
  }
  if (FailureOr<int64_t> maybeOZp = getOutputZeroPoint();
      failed(maybeOZp) || *maybeOZp != 0) {
    // output zero point is not constant 0, cannot fold
    return {};
  }
  if (FailureOr<int64_t> maybeIZp = definingOp.getInput1ZeroPoint();
      failed(maybeIZp) || *maybeIZp != 0) {
    // definingOp's input1 zero point is not constant 0, cannot fold
    return {};
  }
  if (FailureOr<int64_t> maybeOZp = definingOp.getOutputZeroPoint();
      failed(maybeOZp) || *maybeOZp != 0) {
    // definingOp's output zero point is not constant 0, cannot fold
    return {};
  }

  return definingOp.getInput1();
}

OpFoldResult tosa::AbsOp::fold(FoldAdaptor adaptor) {
  auto input = getInput1();
  // Element-wise abs(abs(x)) = abs(x)
  if (auto op = input.getDefiningOp<tosa::AbsOp>()) {
    return input;
  }

  return {};
}

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  // Fold consecutive concats on the same axis into a single op.
  // Keep track of the operands so we are able to construct a new concat
  // later. Conservatively assume that we double the number of operands when
  // folding
  SmallVector<Value, 8> concatOperands;
  concatOperands.reserve(2 * getNumOperands());

  // Find all operands that are foldable concats
  bool foundFoldableConcat = false;
  for (Value operand : getOperands()) {
    concatOperands.emplace_back(operand);

    auto producer = dyn_cast_or_null<ConcatOp>(operand.getDefiningOp());
    if (!producer)
      continue;

    // Not foldable if axes are not the same
    if (getAxis() != producer.getAxis())
      continue;

    // Replace the original operand with all incoming operands
    foundFoldableConcat = true;
    concatOperands.pop_back();
    llvm::append_range(concatOperands, producer->getOperands());
  }

  if (!foundFoldableConcat)
    return {};

  getOperation()->setOperands(concatOperands);
  return getResult();
}

OpFoldResult tosa::ReciprocalOp::fold(FoldAdaptor adaptor) {
  auto input = adaptor.getInput1();

  auto inputAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(input);
  // Fold splat inputs only.
  if (!inputAttr || !inputAttr.isSplat())
    return {};

  auto shapeType = llvm::cast<ShapedType>(getType());
  if (auto floatType = llvm::dyn_cast<FloatType>(inputAttr.getElementType())) {
    auto floatVal = inputAttr.getSplatValue<APFloat>();
    return DenseElementsAttr::get(shapeType,
                                  ReciprocalOp::calcOneElement(floatVal));
  }

  return {};
}
