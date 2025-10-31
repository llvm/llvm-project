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
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

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

namespace {
/// A simple struct to represent ReassociationIndices as an inclusive interval.
/// It's designed to be feasibly minimal, so the call sites should manage the
/// validity of the range manually.
struct ReassociationIndexRange {
  /// FIXME: Signed type is used for consistency with ReassociationIndices.
  /// We should consider refactoring all reassociation utilities to use unsigned
  /// types.
  int64_t leftIdx = 0, rightIdx = 0;

  /// Util for manual checks of the range's validity
  LogicalResult verify() const {
    return leftIdx >= 0 && (leftIdx <= rightIdx) ? success() : failure();
  }

  /// Checks range's containment within another range. Treats the edges
  /// non-exclusively.
  bool isInRange(const ReassociationIndexRange &outerRange) const {
    return leftIdx >= outerRange.leftIdx && rightIdx <= outerRange.rightIdx;
  }

  unsigned size() const {
    assert(succeeded(verify()));
    return rightIdx - leftIdx + 1;
  }
  bool containsSingleIndex() const { return size() == 1; }

  /// Collects indices that do not overlap between this and another range.
  ReassociationIndices
  getNonOverlappingIndicesWith(ReassociationIndexRange &rhs) const {
    if (rightIdx < rhs.leftIdx) {
      // The intervals do not overlap - concatenate the indices from both.
      auto jointFullIndices = getFullIndices();
      jointFullIndices.append(rhs.getFullIndices());
      return jointFullIndices;
    }
    ReassociationIndices result;
    // Handle the chunk left of the overlapping range.
    int64_t leftStart = std::min(leftIdx, rhs.leftIdx);
    int64_t leftEnd = std::max(leftIdx, rhs.leftIdx);
    llvm::append_range(result, llvm::seq(leftStart, leftEnd));
    // Handle the chunk right of the overlapping range. Symmetrically, we should
    // skip the edge of the overlap AND include the rightmost index.
    int64_t rightStart = std::min(rightIdx, rhs.rightIdx) + 1;
    int64_t rightEnd = std::max(rightIdx, rhs.rightIdx);
    if (rightStart < rightEnd)
      llvm::append_range(result, llvm::seq_inclusive(rightStart, rightEnd));
    return result;
  }

  /// Converts the range into ReassociationIndices.
  ReassociationIndices getFullIndices() const {
    ReassociationIndices result;
    for (int64_t idx = leftIdx; idx <= rightIdx; ++idx) {
      result.push_back(idx);
    }
    return result;
  }
};
} // namespace

/// Starting from `sourceStartIdx`, searches `sourceShape` for the first
/// sequence that can be collapsed into a dynamic dimension (at least one must
/// be present in the source).
/// By default, lazily returns once the first dynamic dimension has been found.
/// Setting `matchGreedily` as `true` will also mark all subsequent
/// source dimensions for collapsing into the target.
static FailureOr<ReassociationIndexRange>
findReassociationRangeForDynamicDim(ArrayRef<int64_t> sourceShape,
                                    int64_t sourceStartIdx,
                                    bool matchGreedily = false) {
  const unsigned numSourceDims = sourceShape.size();
  ReassociationIndexRange sourceShapeAsRange{0, numSourceDims - 1};
  std::optional<ReassociationIndexRange> resultRange = std::nullopt;

  ReassociationIndexRange iterationRange{sourceStartIdx, sourceStartIdx};
  for (; iterationRange.isInRange(sourceShapeAsRange);
       iterationRange.rightIdx++) {
    int64_t sourceSize = sourceShape[iterationRange.rightIdx];
    if (sourceSize == ShapedType::kDynamic) {
      resultRange = iterationRange;
      break;
    }
  }
  if (!resultRange)
    return failure();
  if (matchGreedily)
    resultRange->rightIdx = sourceShapeAsRange.rightIdx;
  return *resultRange;
}

/// Starting from `sourceStartIdx`, searches `sourceShape` for the first
/// sequence of static dimensions such that their product matches `targetSize`.
/// By default, lazily returns once the product matches the target size. Setting
/// `matchGreedily` as `true` will append all neighboring unit dimensions
/// (dimensions of 1) to the match.
static FailureOr<ReassociationIndexRange>
findReassociationRangeForSize(ArrayRef<int64_t> sourceShape,
                              int64_t sourceStartIdx, int64_t targetSize,
                              bool matchGreedily = false) {
  const unsigned numSourceDims = sourceShape.size();
  ReassociationIndexRange sourceShapeAsRange{0, numSourceDims - 1};
  std::optional<ReassociationIndexRange> resultRange = std::nullopt;

  ReassociationIndexRange iterationRange{sourceStartIdx, sourceStartIdx};
  int64_t prodOfCollapsedDims = 1;
  while (iterationRange.isInRange(sourceShapeAsRange)) {
    int64_t sourceSize = sourceShape[iterationRange.rightIdx];
    if (sourceSize == ShapedType::kDynamic) {
      // Reassociation for a static dim cannot include a dynamic dim. Reset
      // induction variables to essentially restart the loop from the next
      // source dimension.
      prodOfCollapsedDims = 1;
      iterationRange = {iterationRange.rightIdx + 1,
                        iterationRange.rightIdx + 1};
      continue;
    }
    prodOfCollapsedDims *= sourceSize;
    // If the target size has been exceeded without matching, we need to shift
    // the range start right. From the start of the range, roll back the
    // multiplication until the target size exceeds the product again.
    while (prodOfCollapsedDims > targetSize &&
           !iterationRange.containsSingleIndex()) {
      int64_t frontSourceSize = sourceShape[iterationRange.leftIdx];
      prodOfCollapsedDims /= frontSourceSize;
      // Shrink the range rightwards
      iterationRange.leftIdx++;
    }
    // We could've reached the target size with the current dimension,
    // also as a result of the above shift to right.
    if (prodOfCollapsedDims == targetSize) {
      resultRange = iterationRange;
      break;
    }
    // Increment the iteration range
    iterationRange.rightIdx++;
  }
  if (!resultRange)
    return failure();
  if (matchGreedily) {
    // We now want to collect all unit dimensions directly after the target
    // product match. Advance the iterator to avoid OOB when the product match
    // happens at the last element.
    iterationRange.rightIdx++;
    while (iterationRange.isInRange(sourceShapeAsRange) &&
           sourceShape[iterationRange.rightIdx] == 1) {
      resultRange = iterationRange;
      iterationRange.rightIdx++;
    }
  }
  return *resultRange;
}

/// Attempts to find a valid collapsing reassociation of `sourceShape` into
/// `targetShape` through a simple traversal. If successful, an array of source
/// index ranges is returned, correspondingly to each dimension in the target
/// shape. The resulting indices shall fully cover the `sourceShape` without
/// overlaps.
///
/// The algorithm is essentially a lazy one, searching for non-greedy matches -
/// it will only yield a greedy match for the last target dimension.
/// FIXME: The algorithm can only backtrack when it needs to append an offset
/// for a static target dimension to the preceding dynamic one (this retains the
/// linear complexity). As feasible, consider adding further backtracking
/// routines to enable more reassociations, e.g.:
/// - ?x2x?x2 into ?x2
static FailureOr<SmallVector<ReassociationIndexRange>>
findReassociationRangesForCollapse(ArrayRef<int64_t> sourceShape,
                                   ArrayRef<int64_t> targetShape) {
  unsigned numSourceDims = sourceShape.size(),
           numTargetDims = targetShape.size();
  assert(numSourceDims > numTargetDims);
  ReassociationIndexRange sourceShapeAsRange{0, numSourceDims - 1};

  SmallVector<ReassociationIndexRange> reassocRanges;
  reassocRanges.reserve(numTargetDims);
  // We'll iterate in strides of 2 to enable pseudo-backtracking for simple
  // cases, e.g.:
  // - ?x2x3x5 into ?x15
  std::optional<int64_t> prevTargetSize = std::nullopt;
  for (unsigned targetDimIdx = 0, sourceDimIdx = 0;
       targetDimIdx < numTargetDims; ++targetDimIdx) {
    int64_t targetSize = targetShape[targetDimIdx];
    // Simply check if there are any subsequent target dimensions left - if not,
    // the match must be made greedily.
    bool shouldMatchGreedily = targetDimIdx == numTargetDims - 1;
    FailureOr<ReassociationIndexRange> sourceRange;
    if (targetSize == ShapedType::kDynamic) {
      sourceRange = findReassociationRangeForDynamicDim(
          sourceShape, sourceDimIdx, shouldMatchGreedily);
    } else {
      sourceRange = findReassociationRangeForSize(
          sourceShape, sourceDimIdx, targetSize, shouldMatchGreedily);
    }

    // Run sanity checks on the returned index range.
    if (failed(sourceRange) || failed(sourceRange->verify()) ||
        !sourceRange->isInRange(sourceShapeAsRange))
      return failure();
    if (sourceRange->leftIdx > sourceDimIdx) {
      // If some source dimensions had to be skipped in order to find a match,
      // they must be collapsed into the directly preceding dynamic dimension.
      if (!prevTargetSize || prevTargetSize != ShapedType::kDynamic)
        return failure();
      reassocRanges.back().rightIdx = sourceRange->leftIdx - 1;
    }

    // Store the gathered information as required for the next iteration.
    prevTargetSize = targetSize;
    sourceDimIdx = sourceRange->rightIdx + 1;
    reassocRanges.push_back(*sourceRange);
  }
  // Fail if the source shape wasn't a full match for the target shape. We only
  // need to check the last recorded index - any other gaps should have been
  // mended by the main loop.
  if (reassocRanges.back().rightIdx < sourceShapeAsRange.rightIdx)
    return failure();
  return reassocRanges;
}

/// A variant of `findReassociationRangesForCollapse(...)` that can also scan
/// the shapes right-to-left.
static FailureOr<SmallVector<ReassociationIndexRange>>
findReassociationRangesForCollapse(ArrayRef<int64_t> sourceShape,
                                   ArrayRef<int64_t> targetShape,
                                   bool iterateRightToLeft) {
  if (!iterateRightToLeft)
    return findReassociationRangesForCollapse(sourceShape, targetShape);
  // NB: To iterate right-to-left, we currently reverse the shapes and then
  // reverse the result back. The reversed shapes must not be temporary, as
  // we're passing through an ArrayRef.
  // FIXME: It would be preferable to avoid the expensive copies. At the moment,
  // this approach is chosen for readability of the main implementation.
  std::vector<int64_t> sourceToReverse = sourceShape.vec(),
                       targetToReverse = targetShape.vec();
  std::reverse(sourceToReverse.begin(), sourceToReverse.end());
  std::reverse(targetToReverse.begin(), targetToReverse.end());
  auto invertedRanges =
      findReassociationRangesForCollapse(sourceToReverse, targetToReverse);
  if (failed(invertedRanges))
    return failure();
  SmallVector<ReassociationIndexRange> &rangesToInvert = *invertedRanges;
  unsigned numSourceDims = sourceShape.size();
  // We have received the ranges for inverted shapes. Now we have to invert
  // the ranges back to correspond with the original source shape.
  for (auto &range : rangesToInvert) {
    int64_t invLeftIdx = range.leftIdx, invRightIdx = range.rightIdx;
    range.leftIdx = numSourceDims - 1 - invRightIdx;
    range.rightIdx = numSourceDims - 1 - invLeftIdx;
  }
  // Also invert the ordering of the ranges to correspond with the original
  // target shape.
  std::reverse(rangesToInvert.begin(), rangesToInvert.end());
  return rangesToInvert;
}

std::optional<SmallVector<ReassociationIndices>>
mlir::getReassociationIndicesForCollapse(ArrayRef<int64_t> sourceShape,
                                         ArrayRef<int64_t> targetShape) {
  unsigned numSourceDims = sourceShape.size(),
           numTargetDims = targetShape.size();
  // We're supposed to search for a collapsing reassociation. If the sizes
  // match, there's no actual collapsing taking place - it's either a no-op or a
  // `tensor.reshape`-style reassociation (that would be beyond the scope of
  // this utility).
  if (numSourceDims <= numTargetDims)
    return std::nullopt;
  // Early handling for scalar target types. We should report an invalid
  // reassociation for non-unit static dimensions - no chance to collapse these
  // into a scalar.
  if (numTargetDims == 0) {
    for (unsigned sourceDimIdx = 0; sourceDimIdx < numSourceDims;
         ++sourceDimIdx) {
      int64_t sourceSize = sourceShape[sourceDimIdx];
      if (sourceSize != 1 && sourceSize != ShapedType::kDynamic)
        return std::nullopt;
    }
    return SmallVector<ReassociationIndices>{};
  }

  // Collect source ranges by iterating over the target shape left-to-right.
  FailureOr<SmallVector<ReassociationIndexRange>> maybeForwardRanges =
      findReassociationRangesForCollapse(sourceShape, targetShape);
  if (failed(maybeForwardRanges))
    return std::nullopt;
  auto &ranges = *maybeForwardRanges;
  // Now do the same in reverse. We need to get another valid reassociation
  // through some other strategy, and then compare the results in order to
  // disambiguate mixed subshapes, such as:
  // ?x?x? into ?x?, ?x2x? into ?x?, ?x2x3x6x? into ?x6x?
  // This leads us to lose some of the reassociation opportunities that can only
  // be found by iterating in a certain direction, e.g. 2x2x? into 2x? - without
  // backtracking, the algorithm will fail right-to-left. However, this is the
  // best way to preserve correctness.
  FailureOr<SmallVector<ReassociationIndexRange>> maybeReverseRanges =
      findReassociationRangesForCollapse(sourceShape, targetShape,
                                         /*iterateRightToLeft=*/true);
  if (failed(maybeReverseRanges))
    return std::nullopt;
  auto &reverseRanges = *maybeReverseRanges;

  if (ranges.size() != numTargetDims || reverseRanges.size() != numTargetDims)
    return std::nullopt;
  // Now we can check for ambiguity of each target dimension's reassociation. If
  // successful, we put the full indices into our result map for the target
  // shape.
  SmallVector<ReassociationIndices> reassociationMap(numTargetDims);
  for (unsigned targetDimIdx = 0; targetDimIdx < numTargetDims;
       ++targetDimIdx) {
    ReassociationIndexRange &range = ranges[targetDimIdx];
    ReassociationIndexRange &reverseRange = reverseRanges[targetDimIdx];
    // Get non-overlapping indices between the ranges
    ReassociationIndices nonMatchingIndices =
        range.getNonOverlappingIndicesWith(reverseRange);
    // Unit dimensions can be collapsed wherever - this is the only ambiguity
    // that we allow.
    for (int64_t sourceDimIdx : nonMatchingIndices) {
      if (sourceShape[sourceDimIdx] != 1)
        return std::nullopt;
    }
    reassociationMap[targetDimIdx] = range.getFullIndices();
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

  size_t consumerDims =
      llvm::accumulate(consumerReassociations, size_t(0),
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
static unsigned getMaxPosOfType(ArrayRef<ReassociationExprs> exprArrays) {
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
    Builder &b, ArrayRef<ReassociationIndices> reassociation) {
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
      if (ShapedType::isStatic(collapsedShape[map.index()])) {
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
      llvm::append_range(offsetsSizesAndStrides,
                         llvm::map_range(it.value(), [&](int64_t idx) -> Range {
                           return {zeroAttr, collapseShapeInputShape[idx],
                                   oneAttr};
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

OpFoldResult mlir::reshapeConstantSource(DenseElementsAttr source,
                                         TensorType result,
                                         std::optional<Attribute> cst) {
  if (source && source.isSplat() && result.hasStaticShape() &&
      (!cst.has_value() || source.getSplatValue<Attribute>() == cst.value()))
    return source.resizeSplat(result);

  return {};
}
