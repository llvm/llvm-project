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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

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
  // dimensions set to 1. This prevents the generation of a
  // tensor.collapse_shape op that converts a dynamically shaped tensor into a
  // 0D tensor. While such construct is not incorrect on its own, bufferization
  // cannot properly handle it at the moment, so we avoid it.
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
  auto resultShape =
      llvm::map_to_vector(newShape, [&](int64_t size) -> int64_t {
        // If this is not a placeholder, do not change it.
        if (size >= 0)
          return size;

        // If we do not know the total size of the tensor, keep this dimension
        // dynamic in the result shape.
        if (!inputIsStatic)
          return ShapedType::kDynamic;

        // Calculate the product of all elements in 'newShape' except for the -1
        // placeholder, which we discard by negating the result.
        int64_t totalSizeNoPlaceholder = -llvm::product_of(newShape);

        // If there is a 0 component in 'newShape', resolve the placeholder as
        // 0.
        if (totalSizeNoPlaceholder == 0)
          return 0;

        // Resolve the placeholder as the quotient between the total tensor size
        // and the product of all other sizes.
        return totalSize / totalSizeNoPlaceholder;
      });

  bool resultIsStatic = ShapedType::isStaticShape(resultShape);

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

  if (ShapedType::isDynamicShape(lhsShape) ||
      ShapedType::isDynamicShape(rhsShape))
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
createReassociationMapForCollapse(OpBuilder &builder, Type srcType,
                                  Type dstType) {
  auto srcShape = cast<TensorType>(srcType).getShape();
  auto dstShape = cast<TensorType>(dstType).getShape();

  if (srcShape.empty() || dstShape.empty())
    return {};

  if (ShapedType::isDynamicShape(srcShape) ||
      ShapedType::isDynamicShape(dstShape)) {
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

// Returns true if 'reifyShape' can materialize the given TOSA shape value.
// This only inspects the IR, so that a pattern can bail out before it has
// created any operation.
bool canReifyShape(Value shape) {
  Operation *def = shape.getDefiningOp();
  // A shape that is a block argument has no computation to rebuild.
  if (!def)
    return false;

  if (isa<tosa::ConstShapeOp, tosa::DimOp>(def))
    return true;

  if (isa<tosa::ConcatShapeOp, tosa::AddShapeOp, tosa::SubShapeOp,
          tosa::MulShapeOp>(def))
    return llvm::all_of(def->getOperands(), canReifyShape);

  return false;
}

// Materializes the dimensions of a TOSA shape value as index values.
//
// A value of type '!tosa.shape' has no counterpart in the tensor dialect, so
// shape operations are not lowered on their own. Instead, an operation that
// consumes a shape rebuilds the shape computation with 'arith' and 'tensor'
// operations. The shape operations are left without uses and can be removed
// as dead code.
//
// Callers must check 'canReifyShape' first.
SmallVector<Value> reifyShape(OpBuilder &builder, Location loc, Value shape) {
  Operation *def = shape.getDefiningOp();
  assert(def && "expected shape to be defined by an operation");
  SmallVector<Value> dims;

  if (auto constShape = dyn_cast<tosa::ConstShapeOp>(def)) {
    for (const APInt &dim : constShape.getValues().getValues<APInt>())
      dims.push_back(
          arith::ConstantIndexOp::create(builder, loc, dim.getSExtValue()));
    return dims;
  }

  if (auto dimOp = dyn_cast<tosa::DimOp>(def)) {
    Value input = dimOp.getInput1();
    int64_t axis = dimOp.getAxis();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (inputType && !inputType.isDynamicDim(axis))
      dims.push_back(arith::ConstantIndexOp::create(
          builder, loc, inputType.getDimSize(axis)));
    else
      dims.push_back(tensor::DimOp::create(builder, loc, input, axis));
    return dims;
  }

  if (isa<tosa::ConcatShapeOp>(def)) {
    for (Value operand : def->getOperands())
      llvm::append_range(dims, reifyShape(builder, loc, operand));
    return dims;
  }

  // Element-wise binary operations.
  SmallVector<Value> lhs = reifyShape(builder, loc, def->getOperand(0));
  SmallVector<Value> rhs = reifyShape(builder, loc, def->getOperand(1));
  for (auto [l, r] : llvm::zip_equal(lhs, rhs)) {
    if (isa<tosa::AddShapeOp>(def))
      dims.push_back(builder.createOrFold<arith::AddIOp>(loc, l, r));
    else if (isa<tosa::SubShapeOp>(def))
      dims.push_back(builder.createOrFold<arith::SubIOp>(loc, l, r));
    else if (isa<tosa::MulShapeOp>(def))
      dims.push_back(builder.createOrFold<arith::MulIOp>(loc, l, r));
    else
      llvm_unreachable("unexpected shape operation");
  }
  return dims;
}

class ReshapeConverter : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = reshape.getLoc();
    auto resultType =
        getTypeConverter()->convertType<ShapedType>(reshape.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(reshape.getLoc(),
                                         "could not convert result type");
    }
    auto input = dyn_cast<TypedValue<TensorType>>(adaptor.getInput1());
    if (!input) {
      return rewriter.notifyMatchFailure(reshape.getLoc(),
                                         "expected input type to be tensor");
    }

    llvm::SmallVector<int64_t> newShape;
    if (!tosa::getConstShapeValues(reshape.getShape().getDefiningOp(),
                                   newShape)) {
      // The new shape is computed at runtime. Rebuild its dimensions as index
      // values and reshape with 'tensor.reshape', which takes the new shape as
      // an operand. The inferable dimension size (-1) is only supported for a
      // constant shape.
      Value shape = reshape.getShape();
      if (!canReifyShape(shape))
        return rewriter.notifyMatchFailure(
            reshape, "unsupported computation of the shape operand");

      int64_t shapeRank = cast<tosa::shapeType>(shape.getType()).getRank();
      if (resultType.hasRank() && resultType.getRank() != shapeRank)
        return rewriter.notifyMatchFailure(
            reshape, "rank of the shape operand differs from the result rank");

      SmallVector<Value> dims = reifyShape(rewriter, loc, shape);
      auto shapeTensorType =
          RankedTensorType::get({shapeRank}, rewriter.getIndexType());
      Value shapeTensor =
          tensor::FromElementsOp::create(rewriter, loc, shapeTensorType, dims);
      rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(reshape, resultType, input,
                                                     shapeTensor);
      return success();
    }

    // Infer all intermediate types
    auto inputType = inferReshapeInputType(input, newShape);
    auto expandedType = inferReshapeExpandedType(inputType, newShape);
    auto collapsedType = inferReshapeCollapsedType(inputType, expandedType);

    // Cast input if needed
    auto castInput =
        rewriter.createOrFold<tensor::CastOp>(loc, inputType, input);

    // Emit collaspe-expand pair
    auto collapsed = createCollapse(rewriter, loc, collapsedType, castInput);
    auto expanded = createExpand(rewriter, loc, expandedType, collapsed);

    // Cast to final result type if needed
    auto result =
        rewriter.createOrFold<tensor::CastOp>(loc, resultType, expanded);
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

    ElementsAttr startElems;
    ElementsAttr sizeElems;

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

    SmallVector<int64_t> strides, sizes;
    strides.resize(cast<ShapedType>(sliceOp.getType()).getRank(), 1);

    SmallVector<Value> dynSizes;
    for (const auto &i : llvm::enumerate(sliceSizes)) {
      int64_t size = i.value();
      size_t index = i.index();
      sizes.push_back(size == -1 ? ShapedType::kDynamic : size);
      if (ShapedType::isStatic(sizes.back()))
        continue;

      auto dim = tensor::DimOp::create(rewriter, loc, input, index);
      auto offset = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(sliceStarts[index]));
      dynSizes.push_back(arith::SubIOp::create(rewriter, loc, dim, offset));
    }

    auto newSliceOp = tensor::ExtractSliceOp::create(
        rewriter, sliceOp.getLoc(), sliceOp.getType(), input, ValueRange({}),
        dynSizes, ValueRange({}), rewriter.getDenseI64ArrayAttr(sliceStarts),
        rewriter.getDenseI64ArrayAttr(sizes),
        rewriter.getDenseI64ArrayAttr(strides));

    // Remove const_shape ops when it no longer has use point.
    Operation *startConstShape = sliceOp.getStart().getDefiningOp();
    if (startConstShape->getResult(0).hasOneUse())
      rewriter.eraseOp(startConstShape);

    Operation *sizeConstShape = sliceOp.getSize().getDefiningOp();
    if (sizeConstShape->getResult(0).hasOneUse())
      rewriter.eraseOp(sizeConstShape);

    rewriter.replaceOp(sliceOp, newSliceOp.getResult());
    return success();
  }
};

class PadConverter : public OpConversionPattern<tosa::PadOp> {
public:
  using Base::Base;

  LogicalResult
  matchAndRewrite(tosa::PadOp padOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = padOp.getLoc();
    auto input = padOp.getInput1();

    ElementsAttr paddingElems;
    if (!matchPattern(padOp.getPadding(), m_Constant(&paddingElems))) {
      return rewriter.notifyMatchFailure(
          padOp, "padding must be a static shape value");
    }
    llvm::SmallVector<int64_t> paddingVals;
    for (auto idx : paddingElems.getValues<IntegerAttr>()) {
      paddingVals.push_back(static_cast<int64_t>(idx.getInt()));
    }

    // The TOSA specification requires a negative padding to be diagnosed.
    // We emit a runtime assertion, as this could be dead or unreachable code.
    if (llvm::any_of(paddingVals, [](int64_t val) { return val < 0; })) {
      Value fail =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
      cf::AssertOp::create(
          rewriter, loc, fail,
          rewriter.getStringAttr("tosa.pad: padding must be non-negative"));
    }

    ShapedType inputTy = cast<ShapedType>(input.getType());
    int64_t rank = inputTy.getRank();

    // Setup the default constantAttr.

    Value padConstant = rewriter.createOrFold<tensor::ExtractOp>(
        loc, padOp.getPadConst(),
        ValueRange({arith::ConstantIndexOp::create(rewriter, loc, 0)}));

    if (!padConstant) {
      return rewriter.notifyMatchFailure(
          padOp, "tosa.pad was unable to determine the pad constant value.");
    }

    SmallVector<OpFoldResult, 3> lowValues;
    SmallVector<OpFoldResult, 3> highValues;

    lowValues.reserve(rank);
    highValues.reserve(rank);

    for (int i = 0; i < rank; i++) {
      Value lowVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(paddingVals[2 * i]));
      Value highVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(paddingVals[2 * i + 1]));
      lowValues.push_back(lowVal);
      highValues.push_back(highVal);
    }

    auto newPadOp = tensor::PadOp::create(rewriter, loc, padOp.getType(), input,
                                          lowValues, highValues, padConstant);

    rewriter.replaceOp(padOp, newPadOp.getResult());
    return success();
  }
};

struct ConcatConverter : public OpConversionPattern<tosa::ConcatOp> {
  using OpConversionPattern<tosa::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(
        op, resultType, rewriter.getI64IntegerAttr(op.getAxis()),
        adaptor.getInput1());
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToTensorConversionPatterns(
    const TypeConverter &converter, RewritePatternSet *patterns) {
  patterns
      ->add<ConcatConverter, PadConverter, ReshapeConverter, SliceConverter>(
          converter, patterns->getContext());
}
