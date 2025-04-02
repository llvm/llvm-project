//===- InferTypeOpImpl.cpp - InferType Interface external models *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;
using namespace mlir::tensor;

/// For reshape op compute the shape at dimension `dimIndex` of the output in
/// terms of shape of the `src`, when the reshape op is a collapsing
/// operation. It is the product of the shape of the collapsed dimensions of the
/// `src`.
static OpFoldResult getCollapsedOutputDimFromInputShape(
    OpBuilder &builder, Location loc, int64_t dimIndex, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociationMap) {
  if (!ShapedType::isDynamic(dstStaticShape[dimIndex])) {
    // Static dimension: return Attribute.
    return builder.getIndexAttr(dstStaticShape[dimIndex]);
  }
  AffineMap map = reassociationMap[dimIndex];
  unsigned startPos =
      cast<AffineDimExpr>(map.getResults().front()).getPosition();
  unsigned endPos = cast<AffineDimExpr>(map.getResults().back()).getPosition();
  AffineExpr expr;
  SmallVector<OpFoldResult> dynamicDims;
  for (auto dim : llvm::seq_inclusive(startPos, endPos)) {
    dynamicDims.push_back(builder.createOrFold<tensor::DimOp>(loc, src, dim));
    AffineExpr currExpr = builder.getAffineSymbolExpr(dim - startPos);
    expr = (expr ? expr * currExpr : currExpr);
  }

  // Dynamic dimension: return Value.
  return affine::makeComposedAffineApply(
             builder, loc, AffineMap::get(0, endPos - startPos + 1, expr),
             dynamicDims)
      ->getResult(0);
}

/// Given the `src` of a collapsing reshape op and its reassociation maps,
/// compute the shape of the result of the reshape.
static SmallVector<OpFoldResult, 4> getCollapsedOutputShapeFromInputShape(
    OpBuilder &builder, Location loc, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation) {
  return llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, dstStaticShape.size()), [&](int64_t dim) {
        return getCollapsedOutputDimFromInputShape(
            builder, loc, dim, src, dstStaticShape, reassociation);
      }));
}

struct ReifyCollapseShapeOp
    : public ReifyRankedShapedTypeOpInterface::ExternalModel<
          ReifyCollapseShapeOp, CollapseShapeOp> {
  LogicalResult
  reifyResultShapes(Operation *op, OpBuilder &b,
                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) const {
    auto loc = op->getLoc();
    auto reshapeOp = cast<tensor::CollapseShapeOp>(op);
    reifiedReturnShapes.push_back(getCollapsedOutputShapeFromInputShape(
        b, loc, reshapeOp.getSrc(), reshapeOp.getResultType().getShape(),
        reshapeOp.getReassociationMaps()));
    return success();
  }
};

namespace {

struct ReifyExpandShapeOp
    : public ReifyRankedShapedTypeOpInterface::ExternalModel<ReifyExpandShapeOp,
                                                             ExpandShapeOp> {
  LogicalResult
  reifyResultShapes(Operation *op, OpBuilder &b,
                    ReifiedRankedShapedTypeDims &reifyResultShapes) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    SmallVector<OpFoldResult> resultShapes =
        expandShapeOp.getMixedOutputShape();
    reifyResultShapes.emplace_back(std::move(resultShapes));
    return success();
  }
};

struct ReifyPadOp
    : public ReifyRankedShapedTypeOpInterface::ExternalModel<ReifyPadOp,
                                                             PadOp> {
  LogicalResult
  reifyResultShapes(Operation *op, OpBuilder &b,
                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) const {
    auto padOp = cast<PadOp>(op);
    Location loc = padOp.getLoc();
    auto lowPad = padOp.getMixedLowPad();
    auto highPad = padOp.getMixedHighPad();
    SmallVector<OpFoldResult> shapes;
    for (auto dim : llvm::seq<int64_t>(0, padOp.getSourceType().getRank())) {
      if (!padOp.getResultType().isDynamicDim(dim)) {
        shapes.push_back(b.getIndexAttr(padOp.getResultType().getDimSize(dim)));
        continue;
      }

      // Shape along each dimension is source dim + low pad + high pad.
      SmallVector<OpFoldResult> mapOperands;
      mapOperands.push_back(
          b.createOrFold<tensor::DimOp>(loc, padOp.getSource(), dim));
      mapOperands.push_back(lowPad[dim]);
      mapOperands.push_back(highPad[dim]);
      AffineExpr expr = b.getAffineDimExpr(0) + b.getAffineSymbolExpr(0) +
                        b.getAffineSymbolExpr(1);
      shapes.push_back(getValueOrCreateConstantIndexOp(
          b, loc,
          affine::makeComposedFoldedAffineApply(
              b, loc, AffineMap::get(1, 2, expr), mapOperands)));
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
    return success();
  }
};

} // namespace

void mlir::tensor::registerInferTypeOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    ExpandShapeOp::attachInterface<ReifyExpandShapeOp>(*ctx);
    CollapseShapeOp::attachInterface<ReifyCollapseShapeOp>(*ctx);
    PadOp::attachInterface<ReifyPadOp>(*ctx);
  });
}
