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

using namespace mlir;
using namespace tosa;

static bool findIntermediateShape(ArrayRef<int64_t> lhsShape,
                                  ArrayRef<int64_t> rhsShape,
                                  SmallVector<int64_t> &intermediateShape,
                                  bool isDynamic) {
  if (isDynamic) {
    // TODO (natashaknk): Make dynamic intermediate shape not always be rank-1
    intermediateShape = {ShapedType::kDynamic};
    return true;
  }

  if (lhsShape.empty() || rhsShape.empty()) {
    intermediateShape = {};
    return true;
  }

  unsigned currLhsDim = 0, currRhsDim = 0;
  while (currLhsDim < lhsShape.size() && currRhsDim < rhsShape.size()) {
    int64_t rhsSize = rhsShape[currRhsDim];
    int64_t lhsSize = lhsShape[currLhsDim];
    while (lhsSize != rhsSize && currLhsDim < lhsShape.size() &&
           currRhsDim < rhsShape.size()) {
      if (lhsSize < rhsSize) {
        currLhsDim++;
        if (currLhsDim < lhsShape.size()) {
          lhsSize *= lhsShape[currLhsDim];
        }
      } else {
        currRhsDim++;
        if (currRhsDim < rhsShape.size()) {
          rhsSize *= rhsShape[currRhsDim];
        }
      }
    }
    if (lhsSize == rhsSize) {
      intermediateShape.push_back(lhsSize);
    }
    currRhsDim++;
    currLhsDim++;
  }

  // If the iterators didn't reach the end and their leftover dimensions are not
  // equal to 1 an intermediate shape was not found.
  while (currLhsDim < lhsShape.size()) {
    if (lhsShape[currLhsDim++] != 1) {
      return false;
    }
  }

  while (currRhsDim < rhsShape.size()) {
    if (rhsShape[currRhsDim++] != 1) {
      return false;
    }
  }

  return true;
}

static bool createReassociationMapsForCollapse(
    PatternRewriter &rewriter, ArrayRef<int64_t> srcShape,
    ArrayRef<int64_t> dstShape,
    SmallVector<ReassociationExprs, 4> &reassociationMap, bool isDynamic) {

  // If the shape is dynamic, create a map for collapsing into one dimension.
  if (isDynamic) {
    SmallVector<AffineExpr, 2> exprs;
    for (int i = 0, s = srcShape.size(); i < s; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    reassociationMap = {exprs};
    return true;
  }

  if (dstShape.empty()) {
    reassociationMap = {};
    return true;
  }

  reassociationMap.resize(dstShape.size());
  unsigned currSrcDim = 0, currDstDim = 0;
  while (currSrcDim < srcShape.size() && currDstDim < dstShape.size()) {
    int64_t dstSize = dstShape[currDstDim];
    int64_t srcSize = srcShape[currSrcDim];
    while (srcSize < dstSize && currSrcDim < srcShape.size()) {
      reassociationMap[currDstDim].push_back(
          rewriter.getAffineDimExpr(currSrcDim++));
      srcSize *= srcShape[currSrcDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currDstDim].push_back(
          rewriter.getAffineDimExpr(currSrcDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currDstDim == dstShape.size() - 1 || dstShape[currDstDim + 1] != 1) {
        while (currSrcDim < srcShape.size() && srcShape[currSrcDim] == 1) {
          reassociationMap[currDstDim].push_back(
              rewriter.getAffineDimExpr(currSrcDim++));
        }
      }
    }
    currDstDim++;
  }

  // If both iterators didn't reach the end, we have leftover dimentions which
  // implies that we have a mismatch in shape.
  return currSrcDim == srcShape.size() && currDstDim == dstShape.size();
}

namespace {
Value createCollapse(ConversionPatternRewriter &rewriter, Location loc,
                     ShapedType resultTy, Value operand) {
  ShapedType operandTy = cast<ShapedType>(operand.getType());
  if (resultTy == operandTy)
    return operand;

  bool isDynamic = !operandTy.hasStaticShape();

  if (isDynamic && resultTy.getRank() != 1) {
    (void)rewriter.notifyMatchFailure(
        loc, "Cannot collapse dynamic dims to more than one dimension");
    return {};
  }

  SmallVector<ReassociationExprs, 4> reassociationMap;
  if (!createReassociationMapsForCollapse(rewriter, operandTy.getShape(),
                                          resultTy.getShape(),
                                          reassociationMap, isDynamic)) {
    (void)rewriter.notifyMatchFailure(
        loc, "tosa.reshape Attempting to collapse into an incompatible shape");
    return {};
  }

  SmallVector<int64_t> intermediateShape;
  if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(),
                             intermediateShape, isDynamic)) {
    (void)rewriter.notifyMatchFailure(
        loc, "tosa.reshape Cannot collapse into given shape");
    return {};
  }
  return rewriter.create<tensor::CollapseShapeOp>(loc, resultTy, operand,
                                                  reassociationMap);
}

Value createExpand(ConversionPatternRewriter &rewriter, Location loc,
                   ShapedType resultTy, Value operand) {
  ShapedType operandTy = cast<ShapedType>(operand.getType());
  if (resultTy == operandTy)
    return operand;

  bool isDynamic = !operandTy.hasStaticShape();

  if (isDynamic && operandTy.getRank() != 1) {
    (void)rewriter.notifyMatchFailure(
        loc, "Cannot expand dynamic dims from more than one dimension");
    return {};
  }

  SmallVector<ReassociationExprs, 4> reassociationMap;
  if (!createReassociationMapsForCollapse(rewriter, resultTy.getShape(),
                                          operandTy.getShape(),
                                          reassociationMap, isDynamic)) {
    (void)rewriter.notifyMatchFailure(
        loc, "tosa.reshape Attempting to expand into an incompatible shape");
    return {};
  }

  SmallVector<int64_t> intermediateShape;
  if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(),
                             intermediateShape, isDynamic) ||
      intermediateShape != operandTy.getShape()) {
    (void)rewriter.notifyMatchFailure(
        loc, "tosa.reshape Cannot expand into given shape");
    return {};
  }
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultTy, operand,
                                                reassociationMap);
}

class ReshapeConverterCollapseExpand
    : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ShapedType operandTy = cast<ShapedType>(adaptor.getInput1().getType());
    ShapedType resultTy = cast<ShapedType>(reshape.getType());
    bool isDynamic = !operandTy.hasStaticShape();

    SmallVector<int64_t> intermediateShape;
    if (!findIntermediateShape(resultTy.getShape(), operandTy.getShape(),
                               intermediateShape, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape, "tosa.reshape Cannot identify an intermediate shape between "
                   "the given two shapes");
    }
    auto intermediateTy = RankedTensorType::get(
        intermediateShape, reshape.getType().getElementType());

    Value collapse = createCollapse(rewriter, reshape.getLoc(), intermediateTy,
                                    adaptor.getInput1());
    if (!collapse)
      return failure();

    Value expand = createExpand(rewriter, reshape.getLoc(), resultTy, collapse);
    if (!expand)
      return failure();

    rewriter.replaceOp(reshape, expand);
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
      Value inputIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, i);
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
    Value axisValue = rewriter.createOrFold<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(axis));
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
