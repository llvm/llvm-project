//===- ReshapeOpsUtils.cpp - Utilities used by structured ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

#include <numeric>

using namespace mlir;

Optional<SmallVector<ReassociationIndices>>
mlir::getReassociationIndicesForReshape(ShapedType sourceType,
                                        ShapedType targetType) {
  // Make the sourceType greater rank than the targetType. If they are same
  // rank, then its an unsupported reshape op.
  if (sourceType.getRank() == targetType.getRank())
    return llvm::None;
  if (sourceType.getRank() < targetType.getRank())
    std::swap(sourceType, targetType);

  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  ArrayRef<int64_t> targetShape = targetType.getShape();
  unsigned sourceDim = 0;
  SmallVector<ReassociationIndices> reassociationMap;
  reassociationMap.reserve(targetType.getRank());

  ReassociationIndices currIndices;
  int64_t prodOfCollapsedDims = 1;
  while (sourceDim < sourceShape.size()) {
    unsigned targetDim = reassociationMap.size();

    // If all the dimensions of the targetShape are exhausted, then the
    // remaining dims in the source shape must be all 1s. So for such cases, set
    // 1 as the target shape. The actual reassociation indices will be handled
    // later.
    int64_t currTargetShape =
        (targetDim < targetType.getRank() ? targetShape[targetDim] : 1);
    while (sourceShape[sourceDim] != ShapedType::kDynamicSize &&
           prodOfCollapsedDims * sourceShape[sourceDim] < currTargetShape &&
           sourceDim < sourceShape.size()) {
      prodOfCollapsedDims *= sourceShape[sourceDim];
      currIndices.push_back(sourceDim++);
    }

    // If the current expanded dimension is dynamic, then the collapsed
    // dimensions should also be dynamic and product of all previous unprocessed
    // dimensions of the expanded shape should be 1.
    if (sourceShape[sourceDim] == ShapedType::kDynamicSize &&
        (currTargetShape != ShapedType::kDynamicSize ||
         prodOfCollapsedDims != 1))
      return llvm::None;

    // If the collapsed dim is dynamic, the current expanded dim should also
    // be dynamic.
    if (currTargetShape == ShapedType::kDynamicSize &&
        sourceShape[sourceDim] != ShapedType::kDynamicSize)
      return llvm::None;

    // For static shapes, if the product of dimensions of the expanded shape
    // should match the collapsed dimension shape.
    if (prodOfCollapsedDims * sourceShape[sourceDim] != currTargetShape)
      return llvm::None;

    currIndices.push_back(sourceDim++);
    // If the reassociation is empty but the currIndices is not, this by
    // definition is folding unit-dimensions with the result being scalar type.
    // So only append the `currIndices` if reassociation map is not empty.
    if (targetDim == targetShape.size()) {
      while (sourceDim < sourceShape.size())
        currIndices.push_back(sourceDim++);
      if (!reassociationMap.empty() && !currIndices.empty())
        reassociationMap.back().append(currIndices.begin(), currIndices.end());
      // Break out of the loops. We should be done here.
      break;
    }
    reassociationMap.emplace_back(ReassociationIndices{});
    std::swap(reassociationMap.back(), currIndices);
    prodOfCollapsedDims = 1;
  }
  // All the dimensions in the two shapes must have been processed.
  if (reassociationMap.size() != targetShape.size() ||
      sourceDim != sourceShape.size())
    return llvm::None;
  return reassociationMap;
}

Optional<SmallVector<ReassociationIndices>> mlir::composeReassociationIndices(
    ArrayRef<ReassociationIndices> producerReassociations,
    ArrayRef<ReassociationIndices> consumerReassociations,
    MLIRContext *context) {
  SmallVector<ReassociationIndices> composedIndices;
  // Make the producer the larger sized vector. If they are of same size, the
  // resulting reshape is not a supported reshape op.
  if (producerReassociations.size() == consumerReassociations.size())
    return llvm::None;
  if (producerReassociations.size() < consumerReassociations.size())
    std::swap(producerReassociations, consumerReassociations);

  // Handle the corner case of the result being a rank 0 shaped type. Return an
  // empty reassociation.
  if (consumerReassociations.empty())
    return composedIndices;

  size_t consumerDims = std::accumulate(
      consumerReassociations.begin(), consumerReassociations.end(), 0,
      [](size_t all, ReassociationIndicesRef indices) {
        return all + indices.size();
      });
  if (producerReassociations.size() != consumerDims)
    return llvm::None;

  for (ReassociationIndicesRef consumerIndices : consumerReassociations) {
    ReassociationIndices reassociations;
    for (int64_t consumerIndex : consumerIndices) {
      for (int64_t producerIndex : producerReassociations[consumerIndex])
        reassociations.push_back(producerIndex);
    }
    composedIndices.push_back(std::move(reassociations));
  }
  return composedIndices;
}

SmallVector<SmallVector<AffineExpr, 2>, 2>
mlir::convertReassociationIndicesToExprs(
    MLIRContext *context, ArrayRef<ReassociationIndices> reassociationIndices) {
  SmallVector<SmallVector<AffineExpr, 2>, 2> reassociationMaps;
  for (const auto &indices : reassociationIndices) {
    SmallVector<AffineExpr, 2> reassociationMap;
    reassociationMap.reserve(indices.size());
    for (int64_t index : indices)
      reassociationMap.push_back(mlir::getAffineDimExpr(index, context));
    reassociationMaps.push_back(std::move(reassociationMap));
  }
  return reassociationMaps;
}

template <typename AffineExprTy>
unsigned getMaxPosOfType(ArrayRef<ReassociationExprs> exprArrays) {
  unsigned pos = 0;
  for (const auto &exprs : exprArrays) {
    for (auto expr : exprs) {
      expr.walk([&pos](AffineExpr e) {
        if (auto d = e.dyn_cast<AffineExprTy>())
          pos = std::max(pos, d.getPosition());
      });
    }
  }
  return pos;
}

ArrayAttr mlir::getReassociationIndicesAttribute(
    OpBuilder &b, ArrayRef<ReassociationIndices> reassociation) {
  SmallVector<Attribute, 4> reassociationAttr =
      llvm::to_vector<4>(llvm::map_range(
          reassociation, [&](const ReassociationIndices &indices) -> Attribute {
            return b.getI64ArrayAttr(indices).cast<Attribute>();
          }));
  return b.getArrayAttr(reassociationAttr);
}

SmallVector<ReassociationIndices, 2> mlir::convertReassociationMapsToIndices(
    OpBuilder &b, ArrayRef<ReassociationExprs> reassociationExprs) {
  SmallVector<ReassociationIndices, 2> reassociationIndices;
  for (const auto &exprs : reassociationExprs) {
    ReassociationIndices indices;
    indices.reserve(exprs.size());
    for (const auto &expr : exprs)
      indices.push_back(expr.cast<AffineDimExpr>().getPosition());
    reassociationIndices.push_back(indices);
  }
  return reassociationIndices;
}

SmallVector<AffineMap, 4>
mlir::getSymbolLessAffineMaps(ArrayRef<ReassociationExprs> reassociation) {
  unsigned maxDim = getMaxPosOfType<AffineDimExpr>(reassociation);
  assert(getMaxPosOfType<AffineSymbolExpr>(reassociation) == 0 &&
         "Expected symbol-less expressions");
  SmallVector<AffineMap, 4> maps;
  maps.reserve(reassociation.size());
  for (const auto &exprs : reassociation) {
    assert(!exprs.empty());
    maps.push_back(AffineMap::get(maxDim + 1, 0, exprs, exprs[0].getContext()));
  }
  return maps;
}
bool mlir::isReassociationValid(ArrayRef<AffineMap> reassociation,
                                int *invalidIndex) {
  if (reassociation.empty())
    return true;
  unsigned nDims = reassociation[0].getNumDims();
  unsigned nextExpectedDim = 0;
  for (const auto &it : llvm::enumerate(reassociation)) {
    auto m = it.value();
    if (m.getNumDims() != nDims || m.getNumSymbols() != 0) {
      if (invalidIndex)
        *invalidIndex = it.index();
      return false;
    }
    for (auto e : m.getResults()) {
      auto d = e.dyn_cast<AffineDimExpr>();
      if (!d || d.getPosition() != nextExpectedDim++) {
        if (invalidIndex)
          *invalidIndex = it.index();
        return false;
      }
    }
  }
  if (nextExpectedDim != nDims) {
    if (invalidIndex)
      *invalidIndex = reassociation.size() - 1;
    return false;
  }
  return true;
}

LogicalResult mlir::reshapeLikeShapesAreCompatible(
    function_ref<LogicalResult(const Twine &)> emitError,
    ArrayRef<int64_t> collapsedShape, ArrayRef<int64_t> expandedShape,
    ArrayRef<ReassociationIndices> reassociationMaps, bool isExpandingReshape) {
  unsigned expandedDimStart = 0;
  for (const auto &map : llvm::enumerate(reassociationMaps)) {
    Optional<int64_t> dynamicShape;
    int64_t linearizedStaticShape = 1;
    for (const auto &dim : llvm::enumerate(
             expandedShape.slice(expandedDimStart, map.value().size()))) {
      if (ShapedType::isDynamic(dim.value())) {
        if (isExpandingReshape && dynamicShape) {
          return emitError("invalid to have a single dimension (" +
                           Twine(map.index()) +
                           ") expanded into multiple dynamic dims (" +
                           Twine(expandedDimStart + dynamicShape.getValue()) +
                           "," + Twine(expandedDimStart + dim.index()) + ")");
        }
        dynamicShape = dim.index();
      } else {
        linearizedStaticShape *= dim.value();
      }
    }
    if (dynamicShape) {
      if (!ShapedType::isDynamic(collapsedShape[map.index()])) {
        return emitError(
            "expected dimension " + Twine(map.index()) +
            " of collapsed type to be dynamic since one or more of the "
            "corresponding dimensions in the expanded type is dynamic");
      }
    } else {
      if (collapsedShape[map.index()] != linearizedStaticShape) {
        return emitError("expected dimension " + Twine(map.index()) +
                         " of collapsed type to be static value of " +
                         Twine(linearizedStaticShape));
      }
    }
    expandedDimStart += map.value().size();
  }
  return success();
}
