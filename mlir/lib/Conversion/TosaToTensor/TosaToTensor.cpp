//===- TosaToTensor.cpp - Lowering Tosa to Tensor Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>

using namespace mlir;
using namespace tosa;

// Compute the dimension size of the result tensor corresponding to the
// placeholder value set to -1 in the 'new_shape' attribute of a 'tosa.reshape'
// op. Argument 'newShape' is expected to contain exactly one value set to -1.
static Value getReshapePlaceholderDimSize(OpBuilder &builder, Location loc,
                                          TypedValue<TensorType> input,
                                          ArrayRef<int64_t> newShape) {
  // Calculate the product of all dimensions in the new shape. We expect to have
  // exactly one size set to -1, so we can discard this component by just
  // negating the final product.
  auto newSizeValue = -std::accumulate(newShape.begin(), newShape.end(), 1,
                                       std::multiplies<int64_t>());
  assert(newSizeValue > 0);
  auto newSize = builder.create<arith::ConstantIndexOp>(loc, newSizeValue);

  // Calculate input tensor size. Traverse static dimensions first to fold as
  // many results as possible.
  auto inputShape = input.getType().getShape();
  int64_t staticInputSize = 1;
  for (auto size : inputShape)
    if (size != ShapedType::kDynamic)
      staticInputSize *= size;
 
  // Continue calculating input tensor size by multiplying dynamic dimensions.
  Value inputSize = builder.create<arith::ConstantIndexOp>(loc, staticInputSize);
  for (auto [index, size] : llvm::enumerate(inputShape)) {
    if (size == ShapedType::kDynamic) {
      auto dimSize = builder.create<tensor::DimOp>(loc, input, index);
      inputSize = builder.create<arith::MulIOp>(loc, inputSize, dimSize);
    }
  }

  // Calculate placeholder size, which will be folded if input tensor was
  // statically shaped.
  return builder.createOrFold<arith::DivUIOp>(loc, inputSize, newSize);
}

namespace {

class ReshapeConverterCollapseExpand
    : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = reshape.getLoc();
    auto input = reshape.getInput1();
    auto newShape = reshape.getNewShape();

    // Create list of values for new shape
    SmallVector<Value> newShapeList(reshape.getNewShape().size());
    for (auto [index, size] : llvm::enumerate(reshape.getNewShape())) {
      newShapeList[index] = size == -1 ?
          getReshapePlaceholderDimSize(rewriter, loc, input, newShape) :
          rewriter.create<arith::ConstantIndexOp>(loc, size);
    }

    // Reshape tensor
    auto newShapeTensor = rewriter.createOrFold<tensor::FromElementsOp>(
        loc, newShapeList);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        reshape, reshape.getResult().getType(), input, newShapeTensor);
    return success();
  }
};

class SliceConverter : public OpConversionPattern<tosa::SliceOp> {
public:
  using OpConversionPattern<tosa::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::SliceOp sliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = sliceOp.getLoc();
    Value input = adaptor.getInput();
    ShapedType resultType = cast<ShapedType>(sliceOp.getType());
    if (llvm::isa<UnrankedTensorType>(resultType))
      return failure();
    SmallVector<int64_t> strides, sizes;
    ArrayRef<int64_t> starts = sliceOp.getStart();
    strides.resize(cast<ShapedType>(sliceOp.getType()).getRank(), 1);

    SmallVector<Value> dynSizes;
    for (const auto &i : llvm::enumerate(sliceOp.getSize())) {
      int64_t size = i.value();
      size_t index = i.index();
      sizes.push_back(size == -1 ? ShapedType::kDynamic : size);
      if (!ShapedType::isDynamic(sizes.back()))
        continue;

      auto dim = rewriter.create<tensor::DimOp>(loc, input, index);
      auto offset = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(starts[index]));
      dynSizes.push_back(rewriter.create<arith::SubIOp>(loc, dim, offset));
    }

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), input, ValueRange({}), dynSizes,
        ValueRange({}), rewriter.getDenseI64ArrayAttr(starts),
        rewriter.getDenseI64ArrayAttr(sizes),
        rewriter.getDenseI64ArrayAttr(strides));

    rewriter.replaceOp(sliceOp, newSliceOp.getResult());
    return success();
  }
};

class PadConverter : public OpRewritePattern<tosa::PadOp> {
public:
  using OpRewritePattern<tosa::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    auto loc = padOp.getLoc();
    auto input = padOp.getInput1();
    auto padding = padOp.getPadding();

    ShapedType inputTy = cast<ShapedType>(input.getType());
    Type elementTy = inputTy.getElementType();
    int64_t rank = inputTy.getRank();

    // Setup the default constantAttr.

    Value padConstant;

    if (padOp.getPadConst()) {
      padConstant = rewriter.createOrFold<tensor::ExtractOp>(
          loc, padOp.getPadConst(), ValueRange({}));
    } else {
      TypedAttr constantAttr;
      if (isa<FloatType>(elementTy)) {
        constantAttr = rewriter.getFloatAttr(elementTy, 0.0);
      } else if (isa<IntegerType>(elementTy) && !padOp.getQuantizationInfo()) {
        constantAttr = rewriter.getIntegerAttr(elementTy, 0);
      } else if (isa<IntegerType>(elementTy) && padOp.getQuantizationInfo()) {
        int64_t value = padOp.getQuantizationInfo()->getInputZp();
        constantAttr = rewriter.getIntegerAttr(elementTy, value);
      }
      if (constantAttr)
        padConstant = rewriter.create<arith::ConstantOp>(loc, constantAttr);
    }

    if (!padConstant) {
      return rewriter.notifyMatchFailure(
          padOp, "tosa.pad was unable to determine the pad constant value.");
    }

    Value lowIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value highIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    SmallVector<OpFoldResult, 3> lowValues;
    SmallVector<OpFoldResult, 3> highValues;

    lowValues.reserve(rank);
    highValues.reserve(rank);

    for (int i = 0; i < rank; i++) {
      Value inputIndex = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value lowVal = rewriter.createOrFold<tensor::ExtractOp>(
          loc, padding, ValueRange({inputIndex, lowIndex}));
      Value highVal = rewriter.createOrFold<tensor::ExtractOp>(
          loc, padding, ValueRange({inputIndex, highIndex}));

      lowVal = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), lowVal);
      highVal = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), highVal);

      lowValues.push_back(lowVal);
      highValues.push_back(highVal);
    }

    auto newPadOp = rewriter.create<tensor::PadOp>(
        loc, padOp.getType(), input, lowValues, highValues, padConstant);

    rewriter.replaceOp(padOp, newPadOp.getResult());
    return success();
  }
};

struct ConcatConverter : public OpConversionPattern<tosa::ConcatOp> {
  using OpConversionPattern<tosa::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getType());

    Location loc = op.getLoc();
    int axis = op.getAxis();
    Value axisValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(axis));
    int64_t rank = resultType.getRank();

    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, op.getLoc(), adaptor.getOperands()[0]);

    // Pre-compute the offsets along the axis dimension.
    // The axisOffsets will be of size rank + 1, where the last value
    // will hold the total size of the tensor along the 'axis' dimension.
    SmallVector<OpFoldResult> axisOffsets;
    axisOffsets.push_back(rewriter.getIndexAttr(0));
    axisOffsets.push_back(sizes[axis]);

    for (auto arg : adaptor.getOperands().drop_front()) {
      auto size = rewriter.createOrFold<tensor::DimOp>(loc, arg, axisValue);
      auto currentOffset =
          getValueOrCreateConstantIndexOp(rewriter, loc, axisOffsets.back());
      auto total =
          rewriter.createOrFold<arith::AddIOp>(loc, currentOffset, size);
      axisOffsets.push_back(getAsOpFoldResult(total));
    }
    sizes[axis] = axisOffsets.back();

    // Compute the dynamic sizes of the tensor.empty operation.
    // This is based off of the specified result type of the tosa.concat
    // operation, since we don't want to change the result type of the operation
    // during the conversion.
    SmallVector<Value> dynDims;
    for (int64_t i = 0; i < rank; ++i) {
      if (resultType.isDynamicDim(i)) {
        dynDims.push_back(
            getValueOrCreateConstantIndexOp(rewriter, loc, sizes[i]));
      }
    }

    Value result = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), dynDims);

    for (auto [arg, offset] : llvm::zip(adaptor.getOperands(), axisOffsets)) {
      auto sizes = tensor::getMixedSizes(rewriter, op.getLoc(), arg);
      offsets[axis] = offset;
      result = rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, arg, result, offsets, sizes, strides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToTensorConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<SliceConverter, PadConverter, ConcatConverter>(
      patterns->getContext());

  patterns->add<ReshapeConverterCollapseExpand>(patterns->getContext());
}
