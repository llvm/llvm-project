//===- DecomposeTensorReshapeOp.cpp - Decompose tensor reshape op
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decompose tensor reshape op into tensor collapse-expand pair.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::tensor;

namespace {

// Infer the type to which the input of a 'tensor.reshape' op must be cast when
// lowered.
TensorType inferReshapeInputType(TypedValue<TensorType> input,
                                 SmallVector<OpFoldResult> newShape) {
  // No need to cast input for non-empty target shape
  if (!newShape.empty())
    return input.getType();

  // The input type must be cast into a tensor with the same rank and all static
  // dimensions set to 1. This prevents the generation of a
  // tensor.collapse_shape op that converts a dynamically shaped tensor into a
  // 0D tensor.
  SmallVector<int64_t> shape(input.getType().getRank(), 1);
  return input.getType().clone(shape);
}

// Infer the result type of 'tensor.expand_shape' in the collapse-expand
// pair emitted for a 'tensor.reshape' op.
TensorType inferReshapeExpandedType(TensorType inputType,
                                    SmallVector<int64_t> newShape) {
  // Special case for 0D output tensor. Note: Watch out when using Type::clone()
  // with just '{}', as it will invoke the incorrect overload.
  if (newShape.empty())
    return inputType.clone(ArrayRef<int64_t>{});

  // Check if the input is static, and if so, get its total size
  bool inputIsStatic = inputType.hasStaticShape();
  int64_t totalSize = inputIsStatic ? inputType.getNumElements() : -1;

  // Compute result shape
  bool resultIsStatic = true;
  auto resultShape =
      llvm::map_to_vector(newShape, [&](int64_t size) -> int64_t {
        // If this is not a placeholder, do not change it
        if (size >= 0)
          return size;

        // If we do not know the total size of the tensor, keep this dimension
        // dynamic in the result shape.
        if (!inputIsStatic) {
          resultIsStatic = false;
          return ShapedType::kDynamic;
        }

        // Calculate the product of all elements in 'newShape' except for the -1
        // placeholder, which we discard by negating the result.
        int64_t totalSizeNoPlaceholder = -std::accumulate(
            newShape.begin(), newShape.end(), 1, std::multiplies<int64_t>());

        // If there is a 0 component in 'newShape', resolve the placeholder as
        // 0.
        if (totalSizeNoPlaceholder == 0)
          return 0;

        // Resolve the placeholder as the quotient between the total tensor size
        // and the product of all other sizes.
        return totalSize / totalSizeNoPlaceholder;
      });

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
// pair emitted for a 'tensor.reshape' op.
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
                   Value input, SmallVector<OpFoldResult> outputShape) {
  auto reassociationMap =
      createReassociationMapForCollapse(builder, resultType, input.getType());
  return builder.createOrFold<tensor::ExpandShapeOp>(
      loc, resultType, input, reassociationMap, outputShape);
}

struct DecomposeTensorReshapeOp : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = reshapeOp.getLoc();
    auto resultType = reshapeOp.getResult().getType();
    Value input = reshapeOp.getOperand();
    Value newShape = reshapeOp.getShape();
    auto fromElementsOp = newShape.getDefiningOp<FromElementsOp>();
    if (!fromElementsOp)
      return failure();
    SmallVector<OpFoldResult> newShapeList(fromElementsOp.getElements());

    // Infer all intermediate types
    auto inputType = inferReshapeInputType(input, newShapeList);
    auto expandedType =
        inferReshapeExpandedType(inputType, resultType.getShape());
    auto collapsedType = inferReshapeCollapsedType(inputType, expandedType);

    // Cast input if needed
    auto castInput =
        rewriter.createOrFold<tensor::CastOp>(loc, inputType, input);

    // Emit collaspe-expand pair
    auto collapsed = createCollapse(rewriter, loc, collapsedType, castInput);
    auto expanded =
        createExpand(rewriter, loc, expandedType, collapsed, newShapeList);

    // Cast to final result type if needed
    auto result =
        rewriter.createOrFold<tensor::CastOp>(loc, resultType, expanded);
    rewriter.replaceOp(reshapeOp, result);
    return success();
  }
};

} // namespace

void tensor::populateDecomposeTensorReshapePatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeTensorReshapeOp>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct DecomposeTensorReshapeOpPass final
    : public tensor::impl::DecomposeTensorReshapeOpBase<
          DecomposeTensorReshapeOpPass> {
  void runOnOperation() override;
};

} // namespace

void DecomposeTensorReshapeOpPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tensor::populateDecomposeTensorReshapePatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> tensor::createDecomposeTensorReshapeOpPass() {
  return std::make_unique<DecomposeTensorReshapeOpPass>();
}
