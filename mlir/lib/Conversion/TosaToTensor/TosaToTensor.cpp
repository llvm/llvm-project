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

namespace {

// Infer the type to which the input of a 'tosa.reshape' op must be cast when
// lowered.
TensorType inferReshapeInputType(TypedValue<TensorType> input,
                                 ArrayRef<int64_t> newShape) {
  // No need to cast input for non-empty target shape
  if (!newShape.empty())
    return input.getType();

  // The input type must be cast into a tensor with the same rank and all static
  // dimensions set to 1. This prevents the generation of a tensor.collapse_shape
  // op that converts a dynamically shaped tensor into a 0D tensor. While such
  // construct is not incorrect on its own, bufferization cannot properly handle
  // it at the moment, so we avoid it.
  SmallVector<int64_t> shape(input.getType().getRank(), 1);
  return input.getType().clone(shape);
}

// Infer the result type of 'tensor.expand_shape' in the collapse-expand
// pair emitted for a 'tosa.reshape' op.
TensorType inferReshapeExpandedType(TensorType inputType,
                                    ArrayRef<int64_t> newShape) {
  // Special case for 0D output tensor. Note: Watch out when using Type::clone()
  // with just '{}', as it will invoke the incorrect overload.
  if (newShape.empty())
    return inputType.clone(ArrayRef<int64_t>{});

  // Check if the input is static, and if so, get its total size
  bool inputIsStatic = inputType.hasStaticShape();
  int64_t totalSize = inputIsStatic ? inputType.getNumElements() : -1;

  // Compute result shape
  auto resultShape = llvm::map_to_vector(newShape, [&](int64_t size) -> int64_t {
    // If this is not a placeholder, do not change it
    if (size >= 0)
      return size;

    // If we do not know the total size of the tensor, keep this dimension
    // dynamic in the result shape.
    if (!inputIsStatic)
      return ShapedType::kDynamic;

    // Calculate the product of all elements in 'newShape' except for the -1
    // placeholder, which we discard by negating the result.
    int64_t totalSizeNoPlaceholder = -std::accumulate(
        newShape.begin(), newShape.end(), 1, std::multiplies<int64_t>());

    // If there is a 0 component in 'newShape', resolve the placeholder as 0.
    if (totalSizeNoPlaceholder == 0)
      return 0;

    // Resolve the placeholder as the quotient between the total tensor size and
    // the product of all other sizes.
    return totalSize / totalSizeNoPlaceholder;
  });

  bool resultIsStatic = !ShapedType::isDynamicShape(resultShape);

  // A syntactic restriction in 'tensor.expand_shape' forbids a dynamically
  // shaped input from being reshaped into a statically shaped result. We may
  // simply turn the first result dimension dynamic to address this.
  if (!inputIsStatic && resultIsStatic)
    resultShape[0] = ShapedType::kDynamic;

  // The 'tensor.expand_shape' op also forbids a statically shaped input from
  // being reshaped into a dynamically shaped result, but the placeholder
  // inference algorithm above guarantees that this will never be the case.
  assert(!inputIsStatic || resultIsStatic);

  // Create result type
  return inputType.clone(resultShape);
}

// Infer the result type of 'tensor.collapse_shape' in the collapse-expand
// pair emitted for a 'tosa.reshape' op.
TensorType inferReshapeCollapsedType(TensorType lhsType, TensorType rhsType) {
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();

  if (lhsShape.empty() || rhsShape.empty())
    return lhsType.clone(ArrayRef<int64_t>{});

  if (ShapedType::isDynamicShape(lhsShape) || ShapedType::isDynamicShape(rhsShape))
    return lhsType.clone({ShapedType::kDynamic});

  SmallVector<int64_t> intermediateShape;
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

  // Static shapes are guaranteed to be compatible by the op verifier, so all
  // leftover dimensions should be 1.
  for (; currLhsDim < lhsShape.size(); currLhsDim++) {
    assert(lhsShape[currLhsDim] == 1);
  }
  for (; currRhsDim < rhsShape.size(); currRhsDim++) {
    assert(rhsShape[currRhsDim] == 1);
  }

  return lhsType.clone(intermediateShape);
}

SmallVector<ReassociationExprs>
createReassociationMapForCollapse(OpBuilder &builder, Type srcType, Type dstType) {
  auto srcShape = cast<TensorType>(srcType).getShape();
  auto dstShape = cast<TensorType>(dstType).getShape();

  if (srcShape.empty() || dstShape.empty())
    return {};

  if (ShapedType::isDynamicShape(srcShape) || ShapedType::isDynamicShape(dstShape)) {
    assert(dstShape.size() == 1);
    SmallVector<AffineExpr, 2> exprs;
    for (auto i : llvm::seq<int64_t>(srcShape.size()))
      exprs.push_back(builder.getAffineDimExpr(i));
    return {exprs};
  }

  SmallVector<ReassociationExprs> reassociationMap(dstShape.size());
  unsigned currSrcDim = 0, currDstDim = 0;
  while (currSrcDim < srcShape.size() && currDstDim < dstShape.size()) {
    int64_t dstSize = dstShape[currDstDim];
    int64_t srcSize = srcShape[currSrcDim];
    while (srcSize < dstSize && currSrcDim < srcShape.size()) {
      reassociationMap[currDstDim].push_back(
          builder.getAffineDimExpr(currSrcDim++));
      srcSize *= srcShape[currSrcDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currDstDim].push_back(
          builder.getAffineDimExpr(currSrcDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currDstDim == dstShape.size() - 1 || dstShape[currDstDim + 1] != 1) {
        while (currSrcDim < srcShape.size() && srcShape[currSrcDim] == 1) {
          reassociationMap[currDstDim].push_back(
              builder.getAffineDimExpr(currSrcDim++));
        }
      }
    }
    currDstDim++;
  }

  // If the source and target shapes are compatible, both iterators must have
  // reached the end. This condition is guaranteed by the op verifier for
  // static shapes.
  assert(currSrcDim == srcShape.size() && currDstDim == dstShape.size());
  return reassociationMap;
}

// Create a tensor.collapse_shape op that reshapes the input into the given
// result type.
Value createCollapse(OpBuilder &builder, Location loc, TensorType resultType,
                     Value input) {
  auto reassociationMap =
      createReassociationMapForCollapse(builder, input.getType(), resultType);
  return builder.createOrFold<tensor::CollapseShapeOp>(loc, resultType, input,
                                                       reassociationMap);
}

// Create a tensor.expand_shape op that reshapes the input into the given result
// type.
Value createExpand(OpBuilder &builder, Location loc, TensorType resultType,
                   Value input) {
  auto reassociationMap =
      createReassociationMapForCollapse(builder, resultType, input.getType());
  return builder.createOrFold<tensor::ExpandShapeOp>(loc, resultType, input,
                                                     reassociationMap);
}

class ReshapeConverter : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = reshape.getLoc();
    auto resultType = cast_if_present<ShapedType>(
        getTypeConverter()->convertType(reshape.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(reshape.getLoc(),
                                         "could not convert result type");
    }
    auto input = dyn_cast<TypedValue<TensorType>>(adaptor.getInput1());
    if (!input) {
      return rewriter.notifyMatchFailure(reshape.getLoc(),
                                         "expected input type to be tensor");
    }
    auto newShape = reshape.getNewShape();

    // Infer all intermediate types
    auto inputType = inferReshapeInputType(input, newShape);
    auto expandedType = inferReshapeExpandedType(inputType, newShape);
    auto collapsedType = inferReshapeCollapsedType(inputType, expandedType);

    // Cast input if needed
    auto castInput = rewriter.createOrFold<tensor::CastOp>(loc, inputType, input);

    // Emit collaspe-expand pair
    auto collapsed = createCollapse(rewriter, loc, collapsedType, castInput);
    auto expanded = createExpand(rewriter, loc, expandedType, collapsed);

    // Cast to final result type if needed
    auto result = rewriter.createOrFold<tensor::CastOp>(loc, resultType, expanded);
    rewriter.replaceOp(reshape, result);
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
    Value input = adaptor.getInput1();
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

class PadConverter : public OpConversionPattern<tosa::PadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::PadOp padOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
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
    TypeConverter &converter, RewritePatternSet *patterns) {
  patterns
      ->add<ConcatConverter, PadConverter, ReshapeConverter, SliceConverter>(
          converter, patterns->getContext());
}
