//===- RewriteAsConstant.cpp - Patterns to rewrite tensor ops as constants ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Rewrite tensor.generate with arith.constant if the yielded value is a
/// constant and the tensor type is static.
struct GenerateToConstant : public OpRewritePattern<GenerateOp> {
  using OpRewritePattern<GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenerateOp generateOp,
                                PatternRewriter &rewriter) const override {
    auto tensorType =
        llvm::cast<RankedTensorType>(generateOp.getResult().getType());
    if (!tensorType.hasStaticShape())
      return failure();
    auto terminatorOp =
        cast<tensor::YieldOp>(generateOp.getBody().front().getTerminator());
    Attribute attr;
    if (!matchPattern(terminatorOp.getValue(), m_Constant(&attr)))
      return failure();
    Operation *constantOp =
        rewriter.getContext()
            ->getLoadedDialect<TensorDialect>()
            ->materializeConstant(rewriter,
                                  DenseElementsAttr::get(tensorType, attr),
                                  tensorType, generateOp->getLoc());
    if (!constantOp)
      return failure();
    rewriter.replaceOp(generateOp, constantOp->getResults());
    return success();
  }
};

/// Transform a linear index from one indexing space to another given:
///
/// - the shape of the source indexing space,
/// - the strides of the target indexing space,
/// - a linear index into the source indexing space.
///
/// This function is logically a sequence of linearize/delinearize over
/// different bases but avoids allocating intermediate SmallVectors.
int64_t transformIndexSpace(ArrayRef<int64_t> inputShape,
                            ArrayRef<int64_t> outputStrides,
                            int64_t srcLinearIndex) {
  assert(inputShape.size() == outputStrides.size());

  int64_t dstLinearIndex = 0;

  for (int64_t dim = inputShape.size() - 1; dim >= 0; --dim) {
    // Compute the index into the current dimension of the source tensor.
    // `quotient` is the remaining linear index after accounting for the
    // current dimension.
    //
    // `remainder` is the index into the source tensor for the current
    // dimension.
    auto [quotient, remainder] = std::div(srcLinearIndex, inputShape[dim]);

    srcLinearIndex = quotient;

    // Add the contribution of the current dimension to the output using the
    // permutation map.
    dstLinearIndex += outputStrides[dim] * remainder;
  }

  return dstLinearIndex;
}

template <typename ElemType, typename AttrType>
Value constantFoldPadOp(PatternRewriter &rewriter, Location loc,
                        DenseElementsAttr input, AttrType padValue,
                        ArrayRef<int64_t> padLow, ArrayRef<int64_t> padHigh) {
  auto inputValues = input.tryGetValues<ElemType>();
  if (failed(inputValues))
    return nullptr;

  auto oldShape = input.getType().getShape();

  // Compute the output shape of the new value.
  auto newShape =
      llvm::map_to_vector(llvm::zip(oldShape, padLow, padHigh),
                          [](std::tuple<int64_t, int64_t, int64_t> pack) {
                            auto [old, low, high] = pack;
                            return old + low + high;
                          });

  int64_t outputSize = computeProduct(newShape);

  // Fully initialize the vector with the padding value.
  // The non-padded area will then be copied.
  SmallVector<ElemType> values(outputSize, padValue.getValue());

  // Strides for input and output are used to transform between the indexing
  // space of the input and output tensors.
  SmallVector<int64_t> outputStrides = computeStrides(newShape);

  // The contribution of the low padding to the offset in the output tensor.
  // This is the starting position of the source tensor within the padding
  // tensor.
  int64_t startingOffset = linearize(padLow, outputStrides);

  // Copy values from the input tensor to the corresponding sub-region
  // of the output tensor.
  for (auto [inputIndex, inputValue] : llvm::enumerate(*inputValues)) {
    auto outputIndex = transformIndexSpace(oldShape, outputStrides, inputIndex);
    values[outputIndex + startingOffset] = inputValue;
  }

  // Create an attribute for the folded value.
  auto newType = input.getType().clone(newShape);
  auto newAttr = DenseElementsAttr::get(newType, values);

  Operation *constantOp =
      rewriter.getContext()
          ->getLoadedDialect<TensorDialect>()
          ->materializeConstant(rewriter, newAttr, newType, loc);

  return constantOp ? constantOp->getResult(0) : nullptr;
}

struct PadOpToConstant final : public OpRewritePattern<PadOp> {

  PadOpToConstant(MLIRContext *context, const ControlFoldFn &controlFn,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<PadOp>(context, benefit), controlFn{controlFn} {}

  LogicalResult matchAndRewrite(PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    if (padTensorOp.getNofold())
      return rewriter.notifyMatchFailure(
          padTensorOp, "refusing to fold nofold pad operation");

    TypedValue<RankedTensorType> input = padTensorOp.getSource();
    RankedTensorType resultType = padTensorOp.getResult().getType();

    DenseElementsAttr inputAttr = nullptr;
    if (!matchPattern(input, m_Constant(&inputAttr)))
      return failure();

    Value paddingValue = padTensorOp.getConstantPaddingValue();

    // Extract the constant value used for padding or bail out.
    Attribute paddingAttr = nullptr;
    if (!paddingValue || !matchPattern(paddingValue, m_Constant(&paddingAttr)))
      return rewriter.notifyMatchFailure(padTensorOp,
                                         "unable to get constant value");

    // Try to extract the constant values of the low and high padding.
    auto lowPad = getConstantIntValues(padTensorOp.getMixedLowPad());
    auto highPad = getConstantIntValues(padTensorOp.getMixedHighPad());

    // If the padding cannot be extracted, bail out.
    if (!lowPad || !highPad)
      return rewriter.notifyMatchFailure(padTensorOp,
                                         "unable to extract constant padding");

    // We have a potential candidate, consult the control function to
    // determine if the op should fold.
    if (!controlFn(&padTensorOp.getSourceMutable()))
      return rewriter.notifyMatchFailure(padTensorOp,
                                         "not folding due to cost function");

    Location loc = padTensorOp.getLoc();

    // Try constant folding the supported cases of integer and float values.
    Value newOp =
        llvm::TypeSwitch<Attribute, Value>(paddingAttr)
            .Case([&](FloatAttr floatAttr) {
              return constantFoldPadOp<llvm::APFloat>(
                  rewriter, loc, inputAttr, floatAttr, *lowPad, *highPad);
            })
            .Case([&](IntegerAttr integerAttr) {
              return constantFoldPadOp<llvm::APInt>(
                  rewriter, loc, inputAttr, integerAttr, *lowPad, *highPad);
            })
            .Default(Value());

    if (!newOp)
      return rewriter.notifyMatchFailure(padTensorOp,
                                         "tensor type not supported");

    if (newOp.getType() != resultType)
      newOp = rewriter.create<tensor::CastOp>(loc, resultType, newOp);

    rewriter.replaceOp(padTensorOp, newOp);
    return success();
  }

private:
  ControlFoldFn controlFn;
};

} // namespace

void mlir::tensor::populateRewriteAsConstantPatterns(
    RewritePatternSet &patterns, const ControlFoldFn &controlFn) {
  patterns.add<GenerateToConstant>(patterns.getContext());

  patterns.add<PadOpToConstant>(patterns.getContext(), controlFn);
}
