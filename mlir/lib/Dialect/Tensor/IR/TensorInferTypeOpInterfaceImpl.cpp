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

/// Compute a map that for a given dimension of the expanded type gives the
/// dimension in the collapsed type it maps to. Essentially its the inverse of
/// the `reassocation` maps.
static llvm::DenseMap<int64_t, int64_t>
getExpandedDimToCollapsedDimMap(ArrayRef<AffineMap> reassociation) {
  llvm::DenseMap<int64_t, int64_t> expandedDimToCollapsedDim;
  for (const auto &map : enumerate(reassociation)) {
    unsigned startPos =
        cast<AffineDimExpr>(map.value().getResults().front()).getPosition();
    unsigned endPos =
        cast<AffineDimExpr>(map.value().getResults().back()).getPosition();
    for (auto dim : llvm::seq_inclusive(startPos, endPos)) {
      expandedDimToCollapsedDim[dim] = map.index();
    }
  }
  return expandedDimToCollapsedDim;
}

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

/// For an expanding reshape op, compute the value for a dimension of the output
/// from the shape of the input.
static OpFoldResult getExpandedOutputDimFromInputShape(
    OpBuilder &builder, Location loc, int64_t dimIndex, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation,
    llvm::DenseMap<int64_t, int64_t> &expandedDimToCollapsedDim) {
  if (!ShapedType::isDynamic(dstStaticShape[dimIndex])) {
    // Static dimension: return Attribute.
    return builder.getIndexAttr(dstStaticShape[dimIndex]);
  }
  unsigned sourceDimPos = expandedDimToCollapsedDim[dimIndex];
  unsigned startPos =
      cast<AffineDimExpr>(reassociation[sourceDimPos].getResults().front())
          .getPosition();
  unsigned endPos =
      cast<AffineDimExpr>(reassociation[sourceDimPos].getResults().back())
          .getPosition();
  int64_t linearizedStaticDim = 1;
  for (auto d :
       llvm::enumerate(dstStaticShape.slice(startPos, endPos - startPos + 1))) {
    if (d.index() + startPos == static_cast<unsigned>(dimIndex))
      continue;
    assert(!ShapedType::isDynamic(d.value()) &&
           "single dimension cannot be expanded into multiple dynamic "
           "dimensions");
    linearizedStaticDim *= d.value();
  }
  OpFoldResult sourceDim =
      builder.create<tensor::DimOp>(loc, src, sourceDimPos).getResult();

  // Dynamic dimension: return Value.
  return affine::makeComposedAffineApply(
             builder, loc,
             AffineMap::get(
                 0, 1,
                 builder.getAffineSymbolExpr(0).floorDiv(linearizedStaticDim)),
             sourceDim)
      ->getResult(0);
}

/// Given the `src` of an expanding reshape op, the reassociation maps and the
/// result type, compute the shape of the result of the reshape.
static SmallVector<OpFoldResult, 4> getExpandedOutputShapeFromInputShape(
    OpBuilder &builder, Location loc, Value src,
    ArrayRef<int64_t> dstStaticShape, ArrayRef<AffineMap> reassociation) {
  llvm::DenseMap<int64_t, int64_t> expandedDimToCollapsedDim =
      getExpandedDimToCollapsedDimMap(reassociation);
  return llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, dstStaticShape.size()), [&](int64_t dim) {
        return getExpandedOutputDimFromInputShape(builder, loc, dim, src,
                                                  dstStaticShape, reassociation,
                                                  expandedDimToCollapsedDim);
      }));
}

static SmallVector<OpFoldResult, 4>
getReshapeOutputShapeFromInputShape(OpBuilder &builder, Location loc, Value src,
                                    ArrayRef<int64_t> dstStaticShape,
                                    ArrayRef<AffineMap> reassocation) {
  return dstStaticShape.size() >
                 static_cast<size_t>(
                     llvm::cast<ShapedType>(src.getType()).getRank())
             ? getExpandedOutputShapeFromInputShape(
                   builder, loc, src, dstStaticShape, reassocation)
             : getCollapsedOutputShapeFromInputShape(
                   builder, loc, src, dstStaticShape, reassocation);
}

template <typename OpTy>
struct ReifyExpandOrCollapseShapeOp
    : public ReifyRankedShapedTypeOpInterface::ExternalModel<
          ReifyExpandOrCollapseShapeOp<OpTy>, OpTy> {
  LogicalResult
  reifyResultShapes(Operation *op, OpBuilder &b,
                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) const {
    auto loc = op->getLoc();
    auto reshapeOp = cast<OpTy>(op);
    reifiedReturnShapes.push_back(getReshapeOutputShapeFromInputShape(
        b, loc, reshapeOp.getSrc(), reshapeOp.getResultType().getShape(),
        reshapeOp.getReassociationMaps()));
    return success();
  }
};

namespace {

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
    ExpandShapeOp::attachInterface<
        ReifyExpandOrCollapseShapeOp<tensor::ExpandShapeOp>>(*ctx);
    CollapseShapeOp::attachInterface<
        ReifyExpandOrCollapseShapeOp<tensor::CollapseShapeOp>>(*ctx);
    PadOp::attachInterface<ReifyPadOp>(*ctx);
  });
}
