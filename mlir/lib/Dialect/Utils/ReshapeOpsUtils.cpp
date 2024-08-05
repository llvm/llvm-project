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
#include <optional>

using namespace mlir;

std::optional<SmallVector<ReassociationIndices>>
mlir::getReassociationIndicesForReshape(ShapedType sourceType,
                                        ShapedType targetType) {
  if (sourceType.getRank() > targetType.getRank())
    return getReassociationIndicesForCollapse(sourceType.getShape(),
                                              targetType.getShape());
  if (sourceType.getRank() < targetType.getRank())
    return getReassociationIndicesForCollapse(targetType.getShape(),
                                              sourceType.getShape());
  return std::nullopt;
}

std::optional<SmallVector<ReassociationIndices>>
mlir::getReassociationIndicesForCollapse(ArrayRef<int64_t> sourceShape,
                                         ArrayRef<int64_t> targetShape) {
  if (sourceShape.size() <= targetShape.size())
    return std::nullopt;
  unsigned sourceDim = 0;
  SmallVector<ReassociationIndices> reassociationMap;
  reassociationMap.reserve(targetShape.size());

  ReassociationIndices currIndices;
  int64_t prodOfCollapsedDims = 1;
  while (sourceDim < sourceShape.size()) {
    unsigned targetDim = reassociationMap.size();
    // If we have mapped all the target dimensions stop and handle the remaining
    // tail of size-1 dimensions explicitly.
    if (targetDim == targetShape.size())
      break;

    int64_t currTargetShape = targetShape[targetDim];
    while (sourceDim < sourceShape.size() &&
           sourceShape[sourceDim] != ShapedType::kDynamic &&
           prodOfCollapsedDims * sourceShape[sourceDim] < currTargetShape) {
      prodOfCollapsedDims *= sourceShape[sourceDim];
      currIndices.push_back(sourceDim++);
    }

    // If the current expanded dimension is dynamic, then the collapsed
    // dimensions should also be dynamic and product of all previous unprocessed
    // dimensions of the expanded shape should be 1.
    if (sourceShape[sourceDim] == ShapedType::kDynamic &&
        (currTargetShape != ShapedType::kDynamic || prodOfCollapsedDims != 1))
      return std::nullopt;

    // If the collapsed dim is dynamic, the current expanded dim should also
    // be dynamic.
    if (currTargetShape == ShapedType::kDynamic &&
        sourceShape[sourceDim] != ShapedType::kDynamic)
      return std::nullopt;

    // For static shapes, if the product of dimensions of the expanded shape
    // should match the collapsed dimension shape.
    if (prodOfCollapsedDims * sourceShape[sourceDim] != currTargetShape)
      return std::nullopt;

    currIndices.push_back(sourceDim++);
    reassociationMap.emplace_back(ReassociationIndices{});
    std::swap(reassociationMap.back(), currIndices);
    prodOfCollapsedDims = 1;
  }
  // All the dimensions in the target must have been processed.
  if (reassociationMap.size() != targetShape.size())
    return std::nullopt;
  // Process any remaining entries in the source shape. They all need to be
  // 1 or dynamic.
  for (; sourceDim < sourceShape.size(); sourceDim++) {
    if (sourceShape[sourceDim] != ShapedType::kDynamic &&
        sourceShape[sourceDim] != 1)
      return std::nullopt;
    // The map is empty when the target type is a scalar.
    if (!reassociationMap.empty())
      reassociationMap.back().push_back(sourceDim);
  }
  return reassociationMap;
}

std::optional<SmallVector<ReassociationIndices>>
mlir::composeReassociationIndices(
    ArrayRef<ReassociationIndices> producerReassociations,
    ArrayRef<ReassociationIndices> consumerReassociations,
    MLIRContext *context) {
  SmallVector<ReassociationIndices> composedIndices;
  // Make the producer the larger sized vector. If they are of same size, the
  // resulting reshape is not a supported reshape op.
  if (producerReassociations.size() == consumerReassociations.size())
    return std::nullopt;
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
    return std::nullopt;

  for (ReassociationIndicesRef consumerIndices : consumerReassociations) {
    ReassociationIndices reassociations;
    for (int64_t consumerIndex : consumerIndices) {
      llvm::append_range(reassociations, producerReassociations[consumerIndex]);
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
        if (auto d = dyn_cast<AffineExprTy>(e))
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
            return cast<Attribute>(b.getI64ArrayAttr(indices));
          }));
  return b.getArrayAttr(reassociationAttr);
}

SmallVector<ReassociationIndices, 2> mlir::convertReassociationMapsToIndices(
    ArrayRef<ReassociationExprs> reassociationExprs) {
  SmallVector<ReassociationIndices, 2> reassociationIndices;
  for (const auto &exprs : reassociationExprs) {
    ReassociationIndices indices;
    indices.reserve(exprs.size());
    for (const auto &expr : exprs)
      indices.push_back(cast<AffineDimExpr>(expr).getPosition());
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
      auto d = dyn_cast<AffineDimExpr>(e);
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
    bool foundDynamicShape = false;
    int64_t linearizedStaticShape = 1;

    for (const auto &dim : llvm::enumerate(
             expandedShape.slice(expandedDimStart, map.value().size()))) {
      if (ShapedType::isDynamic(dim.value()))
        foundDynamicShape = true;
      else
        linearizedStaticShape *= dim.value();
    }
    if (foundDynamicShape) {
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

bool mlir::hasNonIdentityLayout(Type type) {
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return !memrefType.getLayout().isIdentity();
  return false;
}

llvm::SmallBitVector
mlir::getSlicedDimensions(ArrayRef<OpFoldResult> sliceInputShape,
                          ArrayRef<Range> sliceParams) {
  assert(sliceParams.size() == sliceInputShape.size() &&
         "only supports non rank-reducing case");
  llvm::SmallBitVector mask(sliceInputShape.size());
  unsigned idx = 0;
  for (const auto &[offset, size, stride] : sliceParams) {
    std::optional<int64_t> offsetConst = getConstantIntValue(offset);
    std::optional<int64_t> strideConst = getConstantIntValue(stride);
    mask[idx] = !isEqualConstantIntOrValue(size, sliceInputShape[idx]) ||
                (!strideConst || *strideConst != 1) ||
                (!offsetConst || *offsetConst != 0);
    idx++;
  }
  return mask;
}

llvm::SmallBitVector mlir::getLinearizedDimensions(
    ArrayRef<ReassociationIndices> reassociationIndices) {
  llvm::SmallBitVector result(reassociationIndices.size());
  for (const auto &it : llvm::enumerate(reassociationIndices))
    result[it.index()] = it.value().size() > 1;
  return result;
}

SmallVector<Range> SliceFromCollapseHelper::getExtractSliceParams(
    MLIRContext *ctx, ArrayRef<ValueRange> multiIndices) {
  unsigned loopIdx = 0;
  auto oneAttr = IntegerAttr::get(IndexType::get(ctx), 1);
  auto zeroAttr = IntegerAttr::get(IndexType::get(ctx), 0);
  SmallVector<Range> offsetsSizesAndStrides;
  offsetsSizesAndStrides.reserve(collapseShapeInputShape.size());
  for (const auto &it : llvm::enumerate(reassociationIndices)) {
    // Case 1: Linearized dimensions that have also been sliced. These
    // are size of 1 because we are iterating over these dimensions. The
    // offsets are exactly the de-linearized multi-indices.
    if (slicedDimensions[it.index()] && linearizedDimensions[it.index()]) {
      llvm::append_range(
          offsetsSizesAndStrides,
          llvm::map_range(multiIndices[loopIdx++], [&](Value v) -> Range {
            return Range{getAsOpFoldResult(v), oneAttr, oneAttr};
          }));
      continue;
    }

    // Case 2: One or possibly multiple combined input dimensions, but we
    // have proven that these are not sliced. In this case we just take
    // the full extent of each dimension in the reassociation list.
    if (linearizedDimensions[it.index()]) {
      llvm::append_range(
          offsetsSizesAndStrides,
          llvm::map_range(it.value(), [&](int64_t idx) -> Range {
            return {zeroAttr, collapseShapeInputShape[idx], oneAttr};
          }));
      continue;
    }

    // Case 3: A single index, but it may be sliced.
    offsetsSizesAndStrides.push_back(sliceParams[it.index()]);
  }
  return offsetsSizesAndStrides;
}

SmallVector<Range>
SliceFromCollapseHelper::getInsertSliceParams(MLIRContext *ctx,
                                              ValueRange tileIndices) {
  auto one = IntegerAttr::get(IndexType::get(ctx), 1);
  auto zero = IntegerAttr::get(IndexType::get(ctx), 0);
  SmallVector<Range> insertParams;
  insertParams.reserve(linearizedDimensions.size());
  unsigned loopIdx = 0;
  for (unsigned i = 0; i < linearizedDimensions.size(); i++) {
    if (linearizedDimensions[i] && slicedDimensions[i]) {
      insertParams.push_back(Range{tileIndices[loopIdx++], one, one});
      continue;
    }
    insertParams.push_back(Range{zero, sliceParams[i].size, one});
  }
  return insertParams;
}

/// Returns the index of the only non-unit dimension among `indices` of `shape`,
/// if such a dimension exists and `indices` has more than one element.
/// Otherwise, return std::nullopt.
static std::optional<int64_t> getUniqueNonUnitDim(ArrayRef<int64_t> indices,
                                                  ArrayRef<int64_t> shape) {
  // Return false if more than one of the dimensions in this group are not 1.
  std::optional<int64_t> dimIndex;
  if (indices.size() < 2)
    return std::nullopt;
  for (int64_t idx : indices) {
    if (shape[idx] != 1) {
      if (dimIndex != std::nullopt)
        return std::nullopt;
      dimIndex = idx;
    }
  }
  return dimIndex;
}

// For each segment in the reassociation indices, check whether we can
// simplify that segment with a rank-reducing extract slice. We can do this if
// all but (exactly) one of the corresponding source dims is 1.
static SmallVector<std::optional<int64_t>> getCollapseShapeTrivialSegments(
    RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  SmallVector<std::optional<int64_t>> trivialSegments;
  for (const auto &indices : reassociationIndices)
    trivialSegments.push_back(
        getUniqueNonUnitDim(indices, sourceType.getShape()));
  return trivialSegments;
}

/// Returns true if any of the segments of the reassociation indices for a
/// collapsing reshape can be simplified using a rank-reducing slice.
static FailureOr<SmallVector<std::optional<int64_t>>>
canCollapseShapeBeSimplifiedByRankReducingSlice(
    RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  SmallVector<std::optional<int64_t>> trivialSegments =
      getCollapseShapeTrivialSegments(sourceType, reassociationIndices);
  if (!llvm::any_of(trivialSegments, [](const std::optional<int64_t> &idx) {
        return idx.has_value();
      }))
    return failure();
  return trivialSegments;
}

FailureOr<CollapseShapeRankReducingSliceSimplificationInfo>
mlir::getSimplifyCollapseShapeWithRankReducingSliceInfo(
    RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociationIndices) {
  FailureOr<SmallVector<std::optional<int64_t>>> trivialSegments =
      canCollapseShapeBeSimplifiedByRankReducingSlice(sourceType,
                                                      reassociationIndices);
  if (failed(trivialSegments))
    return failure();

  // Create the expected result shape of the rank-reducing slice.
  SmallVector<int64_t> sliceShape;
  for (const auto &[nonUnitDim, indices] :
       llvm::zip(*trivialSegments, reassociationIndices)) {
    if (nonUnitDim) {
      sliceShape.push_back(sourceType.getDimSize(*nonUnitDim));
      continue;
    }
    llvm::append_range(sliceShape, llvm::map_range(indices, [&](int64_t idx) {
                         return sourceType.getDimSize(idx);
                       }));
  }
  auto sliceType =
      RankedTensorType::get(sliceShape, sourceType.getElementType());

  // If the rank-reducing slice simplified every segment, then we are done.
  if (sliceShape.size() == reassociationIndices.size())
    return CollapseShapeRankReducingSliceSimplificationInfo{sliceType,
                                                            std::nullopt};

  // Otherwise, we need to create a new collapse_shape op for the segments that
  // weren't covered by the slice. By design, the new reassociation indices has
  // the same number of groups as the old reassociation indices.
  SmallVector<ReassociationIndices> newReassociationIndices;
  SmallVector<int64_t, 2> reassociation;
  int64_t groupIdx = 0;
  for (int64_t dimIdx = 0; dimIdx < sliceType.getRank(); dimIdx++) {
    reassociation.push_back(dimIdx);
    if ((*trivialSegments)[groupIdx] ||
        reassociation.size() == reassociationIndices[groupIdx].size()) {
      newReassociationIndices.push_back(reassociation);
      reassociation.clear();
      groupIdx++;
    }
  }

  return CollapseShapeRankReducingSliceSimplificationInfo{
      sliceType, newReassociationIndices};
}

PackingMetadata mlir::computePackingMetadata(int64_t packedRank,
                                             ArrayRef<int64_t> innerDimPos) {
  PackingMetadata res;
  res.insertPositions.reserve(innerDimPos.size());
  // The pack insert position is the position + the number of previously
  // inserted positions + offset.
  // The offset controls whether the packing dimension is the first or last.
  //
  // Example
  // =======
  // Consider packing from a hypothetical ABCD layout to ABCDba whose
  // pack.inner_dims is [1, 0]. The first step consists in undoing the
  // permutation and producing AaBbCD. This is achieved purely by computing the
  // insert positions of `b` and `a` into `ABCD`, starting from [1, 0]. One
  // possibility, is to produce insert positions [2, 0], this would result in an
  // aAbBCD layout (i.e. offset 0). The other possibility, is to produce insert
  // positions [3, 1], this would result in an AaBbCD layout (i.e. offset 1).
  // The latter is what we expect from packing.
  int64_t offset = 1;
  for (int64_t pos : innerDimPos) {
    int64_t numInsertedBefore = llvm::count_if(
        innerDimPos, [&pos](int64_t pos2) { return pos > pos2; });
    res.insertPositions.push_back(pos + numInsertedBefore + offset);
  }

  DenseSet<int64_t> posSet(res.insertPositions.begin(),
                           res.insertPositions.end());
  res.reassociations.reserve(packedRank);
  for (int64_t i = 1; i <= packedRank; ++i) {
    res.outerPositions.push_back(i - 1);
    if (!posSet.contains(i)) {
      res.reassociations.push_back(ReassociationIndices{i - 1});
      continue;
    }
    res.reassociations.push_back(ReassociationIndices{i - 1, i});
    ++i;
  }
  return res;
}
