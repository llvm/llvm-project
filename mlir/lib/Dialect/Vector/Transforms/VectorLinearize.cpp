//===- VectorLinearize.cpp - vector linearization transforms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns and pass for linearizing ND vectors into 1D.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

namespace {

/// Transform `values` to have 1 fewer element. Do this by combining the element
/// at index `index` with the preceding element. Combine these 2 consecutive
/// elements using the combining function `f`.
template <typename Container, typename Combiner>
static void collapseWithPrevious(Container &values, unsigned index,
                                 const Combiner &combiningFunction) {

  assert(values.size() > 1 && "values has fewer than 2 elements");
  assert(index > 0 && index < values.size() &&
         "index not in range [1, rank(values))");

  auto combined = combiningFunction(values[index - 1], values[index]);
  values[index - 1] = std::move(combined);
  std::copy(values.begin() + index + 1, values.end(), values.begin() + index);
  values.pop_back();
}

/// Examples:
/// values = (2, 3, 4) index = 0 ===> assertion failure
/// values = (2, 3, 4) index = 1 ===> (6, 4)
/// values = (2, 3, 4) index = 2 ===> (2, 12)
static void collapseMul(SmallVector<int64_t> &values, unsigned index) {

  auto combiner = [](int64_t a, int64_t b) { return a * b; };
  return collapseWithPrevious(values, index, combiner);
}

/// Examples:
/// values = (true, false, false) index = 0 ===> assertion failure
/// values = (true, false, false) index = 1 ===> (true, false)
/// values = (true, false, false) index = 2 ===> (true, false)
static void collapseOr(SmallVector<bool> &values, unsigned index) {

  auto combiner = [](bool a, bool b) { return a || b; };
  return collapseWithPrevious(values, index, combiner);
}

/// Collapse dimension `dim` and the preceding dimension into a single
/// dimension, if possible. If not possible, return `vectorType`.
static VectorType getReducedType(VectorType vectorType, unsigned dim) {

  if (!vectorType || vectorType.getRank() <= 1)
    return vectorType;

  ArrayRef<bool> scalableDims = vectorType.getScalableDims();
  assert(scalableDims.size() > 1 && "rank and mask size not same size");

  // 2 scalable dimensions cannot be collapsed together.
  if (scalableDims[dim - 1] && scalableDims[dim])
    return vectorType;

  SmallVector<bool> newMask(vectorType.getScalableDims());
  collapseOr(newMask, dim);

  SmallVector<int64_t> newShape(vectorType.getShape());
  collapseMul(newShape, dim);

  return VectorType::get(newShape, vectorType.getElementType(), newMask);
}

/// Collapse the final 2 dimensions of `vectorType`, if possible.
/// If not possible, return `vectorType`.
static VectorType getReducedType(VectorType vectorType) {

  if (!vectorType || vectorType.getRank() < 2)
    return vectorType;

  return getReducedType(vectorType, vectorType.getRank() - 1);
}

/// Collapse all the dimensions of `vectorType` into a single dimension, if
/// possible.
static FailureOr<Type> getRankOneType(VectorType vectorType) {

  // Multiple scalable dimensions cannot be collapsed together.
  if (!vectorType || vectorType.getNumScalableDims() > 1)
    return failure();

  VectorType rankOneType =
      VectorType::get({vectorType.getNumElements()},
                      vectorType.getElementType(), vectorType.isScalable());

  return rankOneType;
}

/// If `value` is a vector type of a rank other than 1, use a shape_cast to
/// get a vector of rank 1, if possible.
static FailureOr<Value> getCollapsedToRankOne(Value value,
                                              PatternRewriter &rewriter) {

  auto vectorType = dyn_cast<VectorType>(value.getType());
  if (!vectorType)
    return failure();

  FailureOr<Type> rankOneType = getRankOneType(vectorType);
  if (failed(rankOneType))
    return failure();

  return rewriter.createOrFold<vector::ShapeCastOp>(value.getLoc(),
                                                    rankOneType.value(), value);
}

/// Consider inserting a vector of shape `small` into a vector of shape `large`,
/// at position `offsets`: this function enumeratates all the indices in `large`
/// that are written to. The enumeration is with row-major ordering.
///
/// Example: insert a 1x2 vector into a 4x5 vector at position (1,3). The 2
/// positions written to are (1,3) and (1,4), which have linearized indices 8
/// and 9. So [8,9] is returned.
///
/// The length of the returned vector is equal to the number of elements in
/// the shape `small` (i.e. the product of dimensions of `small`).
///
/// Possible input for `large`, `small` and `offsets`:
///    large  =  4, 5, 6, 7, 8
///    small  =     1, 6, 7, 8
///  offsets  =  2, 3, 0
///
/// `small` and `large` must not have more elements than `large`. If `offsets`
/// has fewer elements than `large`, it has implicit trailing 0s. If `small` has
/// fewer elements than `large`, it has implicit leading 1s. So the example
/// above is equivalent to
///
///    large  =  4, 5, 6, 7, 8
///    small  =  1, 1, 6, 7, 8
///  offsets  =  2, 3, 0, 0, 0
static SmallVector<int64_t>
getStridedSliceInsertionIndices(ArrayRef<int64_t> small,
                                ArrayRef<int64_t> large,
                                ArrayRef<int64_t> offsets) {

  assert((large.size() >= small.size()) &&
         "rank of 'large' cannot be lower than rank of 'small'");
  assert((large.size() >= offsets.size()) &&
         "rank of 'large' cannot be lower than the number of offsets");
  const unsigned delta = large.size() - small.size();
  const unsigned nOffsets = offsets.size();
  auto getSmall = [&](int64_t i) -> int64_t {
    return i >= delta ? small[i - delta] : 1;
  };
  auto getOffset = [&](int64_t i) -> int64_t {
    return i < nOffsets ? offsets[i] : 0;
  };

  // Using 2 vectors of indices, at each iteration populate the updated set of
  // indices based on the old set of indices, and the size of the small
  // vector in the current iteration.
  SmallVector<int64_t> indices{0};
  const int largeRank = large.size();
  int64_t stride = 1;
  for (int i = largeRank - 1; i >= 0; --i) {
    const int64_t currentSize = indices.size();
    const int64_t smallSize = getSmall(i);
    const int64_t nextSize = currentSize * smallSize;
    SmallVector<int64_t> nextIndices(nextSize);
    int64_t *base = nextIndices.begin();
    int64_t offset = getOffset(i) * stride;
    for (int j = 0; j < smallSize; ++j) {
      for (int k = 0; k < currentSize; ++k) {
        base[k] = indices[k] + offset;
      }
      offset += stride;
      base += currentSize;
    }
    stride *= large[i];
    indices = std::move(nextIndices);
  }
  return indices;
}

/// Combine the first 2 elements of `position` into a single element, if
/// possible. The positions are merged based on the shape of `vectorType`.
/// The returned value specifies if `position` changes.
static bool collapseFront(SmallVector<OpFoldResult> &position,
                          VectorType vectorType, PatternRewriter &rewriter) {

  if (position.size() <= 1)
    return false;

  assert(vectorType && "expected a vector type");
  assert(vectorType.getRank() > 1 &&
         "vectorType must have rank no less than size of 'position'");

  Attribute attributeDimZero = dyn_cast<Attribute>(position[0]);
  Attribute attributeDimOne = dyn_cast<Attribute>(position[1]);

  // We don't currently support combining dynamic positions:
  if (!attributeDimZero || !attributeDimOne)
    return false;

  int64_t intDimZero = cast<IntegerAttr>(attributeDimZero).getInt();
  int64_t intDimOne = cast<IntegerAttr>(attributeDimOne).getInt();

  int64_t newLeadingPos = intDimZero * vectorType.getDimSize(1) + intDimOne;
  IntegerAttr leadingPos = rewriter.getI64IntegerAttr(newLeadingPos);
  position[1] = leadingPos;
  position.erase(position.begin());
  return true;
}

/// Return true if the mask operation has 0 or 1 non-unit dimensions.
static bool
isCreateMaskWithAtMostOneNonUnit(vector::CreateMaskOp createMaskOp) {
  ArrayRef<int64_t> shape = createMaskOp.getType().getShape();
  bool multipleNonUnitDim =
      llvm::count_if(shape, [](int64_t dim) { return dim > 1; }) > 1;
  return !multipleNonUnitDim;
}

/// Find the inner most dimension `dim` such that an insert_strided_slice or
/// extract_strided_slice slice can be rewritten by collapsing dimensions `dim`
/// and `dim` - 1. If such a dimension is found, update `largeType`, `small`,
/// and `offsets` in place, and return `true`. If no such dimension is found,
/// return `false`.
///
/// The assumptions on the sizes of `small`, `largeType`, and `offsets` are the
/// same as the function `getStridedSliceInsertionIndices`, please see the
/// example there.

/// The return type encapsulating the types of the 'collapsed' operation.
struct StridedSliceTriple {
  VectorType small;
  VectorType large;
  SmallVector<int64_t> offsets;
};

static FailureOr<StridedSliceTriple>
collapseInnerMostPossible(VectorType smallType, const VectorType largeType,
                          const ArrayRef<int64_t> offsets) {

  assert(largeType.getRank() >= smallType.getRank() &&
         "rank of 'small' is greater than rank of 'large'");

  // Rank-1 cannot be reduced to rank-0.
  if (largeType.getRank() <= 1)
    return failure();

  // Prepend to the small type so that it has the same rank as the large type.
  // Doing this upfront requires data copies before confirming that we won't
  // return failure, but simplifies the logic significantly and so is deemed
  // worth it.
  smallType = [&]() {
    const int64_t dr = largeType.getRank() - smallType.getRank();
    SmallVector<int64_t> shape(smallType.getShape());
    SmallVector<bool> scale(smallType.getScalableDims());
    shape.insert(shape.begin(), dr, 1);
    scale.insert(scale.begin(), dr, false);
    return VectorType::get(shape, smallType.getElementType(), scale);
  }();

  const ArrayRef<int64_t> smallShape = smallType.getShape();
  const ArrayRef<int64_t> largeShape = largeType.getShape();
  const ArrayRef<bool> scalableDims = largeType.getScalableDims();

  // The algorithm iterates through the dimensions of the small type, from
  // the back (inner-most dimension) to the front. When the remaining prefix is
  // all 1's, the condition of collapsibility is more relaxed. Specifically,
  // when the prefix is not all 1's, then the corresponding sizes the large and
  // small types must match. To detect for all 1's, we keep track of the product
  // of dimensions visited and compare it the total number of elements in the
  // small type:
  const int64_t totalElementsInSmall = smallType.getNumElements();

  int64_t suffixElementsInSmall = 1;
  for (int si = smallType.getRank() - 1; si > 0; --si) {

    suffixElementsInSmall *= smallShape[si];
    if ((suffixElementsInSmall != totalElementsInSmall) &&
        (smallShape[si] != largeShape[si]))
      continue;

    // Can only collapse scalable dims if the resulting collapsed dimension is
    // the same size in the 2 vectors.
    if (scalableDims[si] || scalableDims[si - 1]) {
      if (smallShape[si] != largeShape[si] ||
          smallShape[si - 1] != largeShape[si - 1])
        continue;
    }

    const VectorType flatLarge = getReducedType(largeType, si);
    if (flatLarge == largeType)
      continue;

    VectorType flatSmall = getReducedType(smallType, si);
    SmallVector<int64_t> flatOffsets(offsets);
    flatOffsets.resize(largeType.getRank(), 0);
    flatOffsets[si - 1] *= largeShape[si];
    flatOffsets[si - 1] += flatOffsets[si];
    flatOffsets.erase(flatOffsets.begin() + si);
    return StridedSliceTriple{flatSmall, flatLarge, flatOffsets};
  }
  return failure();
}

/// Convert an array of attributes into a vector of integers.
static FailureOr<SmallVector<int64_t>> intsFromArrayAttr(ArrayAttr attributes) {

  if (!attributes || llvm::any_of(attributes, [](Attribute a) {
        return !isa<IntegerAttr>(a);
      }))
    return failure();

  SmallVector<int64_t> asIntegers;
  asIntegers.reserve(attributes.size());
  for (auto attr : attributes)
    asIntegers.push_back(cast<IntegerAttr>(attr).getInt());

  return asIntegers;
}

/// Return `value` with dimensions `dim` and its preceding dimension combined,
/// if possible. Otherwise return `value`.
static Value getReducedValue(PatternRewriter &rewriter, Value value,
                             unsigned dim) {

  VectorType vectorType = dyn_cast<VectorType>(value.getType());
  if (!vectorType)
    return value;

  VectorType reducedType = getReducedType(vectorType, dim);
  return rewriter.createOrFold<vector::ShapeCastOp>(value.getLoc(), reducedType,
                                                    value);
}

/// Reduce the inner two dimensions of `value` using a shape_cast, if possible.
static Value getReducedValue(PatternRewriter &rewriter, Value value) {

  VectorType vectorType = dyn_cast<VectorType>(value.getType());
  if (!vectorType || vectorType.getRank() <= 1)
    return value;

  return getReducedValue(rewriter, value, vectorType.getRank() - 1);
}

/// Reduce the innermost 2 dimensions of values in `values` using a shape_cast,
/// otherwise retain the original value.
static SmallVector<Value> getReducedValues(ValueRange values,
                                           PatternRewriter &rewriter) {

  SmallVector<Value> replacements;
  replacements.reserve(values.size());
  for (auto val : values)
    replacements.push_back(getReducedValue(rewriter, val));

  return replacements;
}

using PreCondition = std::function<LogicalResult(Operation *)>;

/// This class automates the running of a user provided matcher at the start of
/// `matchAndRewrite`. Classes that inherit from it must implement
/// `postConditionMatchAndRewrite` instead of `matchAndRewrite`.
template <class TOp>
struct OpRewritePatternWithPreCondition : OpRewritePattern<TOp> {
  OpRewritePatternWithPreCondition(MLIRContext *context, const PreCondition &p,
                                   PatternBenefit benefit = 1)
      : OpRewritePattern<TOp>(context, benefit), preCondition(p) {}

private:
  LogicalResult matchAndRewrite(TOp op, PatternRewriter &rewriter) const final {
    if (failed(preCondition(op)))
      return rewriter.notifyMatchFailure(op, "the precondition failed");
    return postConditionMatchAndRewrite(op, rewriter);
  }

  virtual LogicalResult
  postConditionMatchAndRewrite(TOp op, PatternRewriter &rewriter) const = 0;

  PreCondition preCondition;
};

/// Linearize the innermost 2 dimensions of a vector.bitcast
///
/// BEFORE:
/// %b = vector.bitcast %arg0 :  vector<1x3x[8]xi8> to vector<1x3x[2]xi32>
///
/// AFTER:
/// %0 = vector.shape_cast %arg0 : vector<1x3x[8]xi8> to vector<[24]xi8>
/// %1 = vector.bitcast %0 : vector<[24]xi8> to vector<[6]xi32>
/// %b = vector.shape_cast %1 : vector<[6]xi32> to vector<1x3x[2]xi32>
struct CollapseInnerVectorBitCast final
    : OpRewritePatternWithPreCondition<vector::BitCastOp> {

  CollapseInnerVectorBitCast(MLIRContext *context, const PreCondition &p,
                             PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::BitCastOp>(context, p,
                                                            benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::BitCastOp bitCast,
                               PatternRewriter &rewriter) const final {

    VectorType preType = bitCast.getResultVectorType();
    VectorType postType = getReducedType(preType);
    if (postType == preType)
      return rewriter.notifyMatchFailure(bitCast, "result type is irreducible");
    Value source = getReducedValue(rewriter, bitCast.getSource());
    Value newBitCast =
        rewriter.create<vector::BitCastOp>(bitCast.getLoc(), postType, source);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(bitCast, preType,
                                                     newBitCast);
    return success();
  }
};

/// Linearize the innermost 2 dimensions of a vectorizable operation.
///
/// BEFORE:
/// %s = math.sin %0 : vector<4x3x2xi8>
///
/// AFTER:
/// %1 = vector.shape_cast %0 : vector<4x3x2xi8> to vector<4x6xi8>
/// %2 = math.sin %1 : vector<4x6xi8>
/// %s = vector.shape_cast %2 : vector<4x6xi8> to vector<4x3x2xi8>
struct CollapseInnerVectorizable final
    : OpTraitRewritePattern<OpTrait::Vectorizable> {
  using OpTraitRewritePattern::OpTraitRewritePattern;

public:
  CollapseInnerVectorizable(MLIRContext *context, const PreCondition &p,
                            PatternBenefit benefit)
      : OpTraitRewritePattern<OpTrait::Vectorizable>(context, benefit),
        preCondition(p) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (failed(preCondition(op)))
      return rewriter.notifyMatchFailure(op, "precondition failed");

    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "does not have 1 result");

    auto preType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!preType)
      return rewriter.notifyMatchFailure(op, "unique result is not a vector");

    VectorType postType = getReducedType(preType);
    if (postType == preType)
      return rewriter.notifyMatchFailure(op, "result has an irreducible type");

    OperationState newOpState(op->getLoc(), op->getName());
    newOpState.addOperands(getReducedValues(op->getOperands(), rewriter));
    newOpState.addTypes(postType);
    newOpState.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(newOpState);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, preType,
                                                     newOp->getResult(0));

    return success();
  }

private:
  PreCondition preCondition;
};

/// Linearize the innermost 2 dimensions of a vector.shuffle operation.
///
/// BEFORE:
/// %shuffle_2d = vector.shuffle %v1_2d, %v2_2d [ shuffle_indices ]
///
/// AFTER:
/// %v1_1d = vector.shape_cast %v1_2d : [...]
/// %v2_1d = vector.shape_cast %v2_2d : [...]
/// %shuffle_1d = vector.shuffle %v1_1d, %v2_1d [ shuffle_indices_1d ]
/// %shuffle_2d = vector.shape_cast %shuffle_1d :  [...]
///
/// Where `shuffle_indices_1d` are computed by expanding `shuffle_indices`.
struct CollapseInnerVectorShuffle final
    : OpRewritePatternWithPreCondition<vector::ShuffleOp> {

  CollapseInnerVectorShuffle(MLIRContext *context, const PreCondition &p,
                             PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::ShuffleOp>(context, p,
                                                            benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::ShuffleOp shuffleOp,
                               PatternRewriter &rewriter) const final {
    VectorType preType = shuffleOp.getResultVectorType();
    VectorType postType = getReducedType(preType);
    if (postType == preType)
      return rewriter.notifyMatchFailure(shuffleOp, "irreducible type");
    SmallVector<Value> newOperands =
        getReducedValues(shuffleOp.getOperands(), rewriter);
    const ArrayRef<int64_t> oldMask = shuffleOp.getMask();
    const ArrayRef<int64_t> v1Shape = shuffleOp.getV1VectorType().getShape();

    // Only if the outermost dimension is being collapsed does the mask get
    // modified:
    auto factor = v1Shape.size() > 2 ? 1 : v1Shape.back();
    SmallVector<int64_t> indices(oldMask.size() * factor);
    for (auto [i, value] : llvm::enumerate(oldMask)) {
      auto *iter = indices.begin() + factor * i;
      std::iota(iter, iter + factor, factor * value);
    }
    auto newShuffle = rewriter.create<vector::ShuffleOp>(
        shuffleOp.getLoc(), postType, newOperands[0], newOperands[1], indices);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(shuffleOp, preType,
                                                     newShuffle.getResult());
    return success();
  }
};

/// Collapse the 2 innermost dimensions of a vector.extract_strided_slice that
/// can be collapsed.
///
/// BEFORE:
/// %o vector.extract_strided_slice %arg0 [...]
///                               vector<4x8xi8> to vector<2x8xi8>
///
/// AFTER:
/// %0 = vector.shape_cast %arg0 : vector<4x8xi8> to vector<32xi8>
/// %1 = vector.extract_strided_slice %0 [...] vector<32xi8> to vector<16xi8>
/// %o = vector.shape_cast %1 : vector<16xi8> to vector<2x8xi8>
///
/// Note that this pattern will collapse the first pair of successive dimensions
/// that it can, starting from the 2 innermost dimensions and working to the
/// outermost 2 dimensions. If no such pair of dimensions is found, the pattern
/// fails to match
struct CollapseInnerExtractStrided final
    : public OpRewritePatternWithPreCondition<vector::ExtractStridedSliceOp> {
  CollapseInnerExtractStrided(MLIRContext *context, const PreCondition &p,
                              PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::ExtractStridedSliceOp>(
            context, p, benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                               PatternRewriter &rewriter) const final {

    FailureOr<SmallVector<int64_t>> maybeIntOffsets =
        intsFromArrayAttr(extractOp.getOffsets());
    if (failed(maybeIntOffsets))
      return rewriter.notifyMatchFailure(extractOp,
                                         "failed to obtain integer offsets");

    FailureOr<StridedSliceTriple> updated = collapseInnerMostPossible(
        extractOp.getType(), extractOp.getSourceVectorType(),
        maybeIntOffsets.value());
    if (failed(updated))
      return rewriter.notifyMatchFailure(extractOp,
                                         "failed to collapse any dimensions");

    auto flatIn = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), updated->large, extractOp.getVector());

    auto replacement = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), flatIn, updated->offsets, updated->small.getShape(),
        SmallVector<int64_t>(updated->offsets.size(), 1));

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        extractOp, extractOp.getType(), replacement);

    return success();
  }
};

/// Collapse the 2 innermost dimensions of a vector.insert_strided_slice that
/// can be collapsed.
///
/// BEFORE:
/// %o = vector.insert_strided_slice %arg0, %arg1 [...] vector
///                                            <2x2xi8> into vector<3x2xi8>
///
/// AFTER:
/// %0 = vector.shape_cast %arg0 : vector<2x2xi8> to vector<4xi8>
/// %1 = vector.shape_cast %arg1 : vector<3x2xi8> to vector<6xi8>
/// %2 = vector.insert_strided_slice %0, %1 [...] vector<4xi8> into vector<6xi8>
/// %o = vector.shape_cast %2 : vector<6xi8> to vector<3x2xi8>
struct CollapseInnerInsertStrided final
    : public OpRewritePatternWithPreCondition<vector::InsertStridedSliceOp> {
  CollapseInnerInsertStrided(MLIRContext *context, const PreCondition &p,
                             PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::InsertStridedSliceOp>(
            context, p, benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::InsertStridedSliceOp insertOp,
                               PatternRewriter &rewriter) const final {

    FailureOr<SmallVector<int64_t>> maybeIntOffsets =
        intsFromArrayAttr(insertOp.getOffsets());
    if (failed(maybeIntOffsets))
      return rewriter.notifyMatchFailure(insertOp,
                                         "failed to obtain integer offsets");

    FailureOr<StridedSliceTriple> updated =
        collapseInnerMostPossible(insertOp.getSourceVectorType(),
                                  insertOp.getType(), maybeIntOffsets.value());

    if (failed(updated))
      return rewriter.notifyMatchFailure(insertOp,
                                         "failed to collapse any dimensions");

    Value shapeCast = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), updated->small, insertOp.getValueToStore());

    Value collapsedDst = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), updated->large, insertOp.getDest());

    auto replacement = rewriter.create<vector::InsertStridedSliceOp>(
        insertOp.getLoc(), shapeCast, collapsedDst, updated->offsets,
        SmallVector<int64_t>(updated->offsets.size(), 1));

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        insertOp, insertOp.getType(), replacement);

    return success();
  }
};

/// Collapse the 2 innermost dimensions of a vector.extract
///
/// BEFORE:
/// %o = vector.extract %arg0[1, 2] : vector<5x7xi8> from vector<2x3x5x7xi8>
///
/// AFTER:
/// %0 = vector.shape_cast %arg0 : vector<2x3x5x7xi8> to vector<2x3x35xi8>
/// %1 = vector.extract %0[1, 2] : vector<35xi8> from vector<2x3x35xi8>
/// %o = vector.shape_cast %1 : vector<35xi8> to vector<5x7xi8>
struct CollapseInnerExtract final
    : public OpRewritePatternWithPreCondition<vector::ExtractOp> {

  CollapseInnerExtract(MLIRContext *context, const PreCondition &p,
                       PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::ExtractOp>(context, p,
                                                            benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::ExtractOp extractOp,
                               PatternRewriter &rewriter) const final {

    auto vectorType = dyn_cast<VectorType>(extractOp.getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          extractOp, "result type is scalar, cannot collapse inner dimensions");

    VectorType reducedType = getReducedType(vectorType);
    if (reducedType == vectorType)
      return rewriter.notifyMatchFailure(extractOp,
                                         "result type is irreducible");

    Value reducedIn = getReducedValue(rewriter, extractOp.getVector());

    Value reducedExtract = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), reducedIn, extractOp.getMixedPosition());

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(extractOp, vectorType,
                                                     reducedExtract);

    return success();
  }
};

/// Collapse the 2 innermost dimensions of a vector.insert
///
/// BEFORE:
/// %o = vector.insert %arg0, %arg1[1, 2] : vector<5x7xi8> into
/// vector<2x3x5x7xi8>
///
/// AFTER:
/// %0 = vector.shape_cast %arg0 : vector<5x7xi8> to vector<35xi8>
/// %1 = vector.shape_cast %arg1 : vector<2x3x5x7xi8> to vector<2x3x35xi8>
/// %2 = vector.insert %0, %1[1, 2] : vector<35xi8> into vector<2x3x35xi8>
/// %o = vector.shape_cast %2 : vector<2x3x35xi8> to vector<2x3x5x7xi8>
struct CollapseInnerInsert final
    : public OpRewritePatternWithPreCondition<vector::InsertOp> {

  CollapseInnerInsert(MLIRContext *context, const PreCondition &p,
                      PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::InsertOp>(context, p,
                                                           benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::InsertOp insertOp,
                               PatternRewriter &rewriter) const final {

    auto toInsertType = dyn_cast<VectorType>(insertOp.getValueToStoreType());
    if (!toInsertType)
      return rewriter.notifyMatchFailure(
          insertOp,
          "value to insert is scalar, canot collapse inner dimensions");

    VectorType reducedType = getReducedType(toInsertType);
    if (reducedType == toInsertType)
      return rewriter.notifyMatchFailure(
          insertOp, "value to insert has an irreducible type");

    Value reducedToStore =
        getReducedValue(rewriter, insertOp.getValueToStore());
    Value reducedDst = getReducedValue(rewriter, insertOp.getDest());

    auto reducedInsert = rewriter.create<vector::InsertOp>(
        insertOp.getLoc(), reducedToStore, reducedDst,
        insertOp.getMixedPosition());

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        insertOp, insertOp.getType(), reducedInsert);

    return success();
  }
};

/// Collapse the outermost 2 dimensions of a vector.extract
///
/// BEFORE:
/// %o = vector.extract %arg0[1, 2] : vector<5x7xi8> from vector<2x3x5x7xi8>
///
/// AFTER:
/// %0 = vector.shape_cast %arg0 : vector<2x3x5x7xi8> to vector<6x5x7xi8>
/// %o = vector.extract %0[5] : vector<5x7xi8> from vector<6x5x7xi8>
struct CollapseOuterExtract final
    : public OpRewritePatternWithPreCondition<vector::ExtractOp> {

  CollapseOuterExtract(MLIRContext *context, const PreCondition &p,
                       PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::ExtractOp>(context, p,
                                                            benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::ExtractOp extractOp,
                               PatternRewriter &rewriter) const final {

    VectorType srcType = extractOp.getVector().getType();

    SmallVector<OpFoldResult> position = extractOp.getMixedPosition();
    if (!collapseFront(position, srcType, rewriter))
      return rewriter.notifyMatchFailure(
          extractOp, "failed to collapse the outermost 2 dimensions");

    Value reducedIn = getReducedValue(rewriter, extractOp.getVector(), 1);
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(extractOp, reducedIn,
                                                   position);

    return success();
  }
};

/// Collapse the outermost 2 dimensions of a vector.insert
///
/// BEFORE:
/// %o = vector.insert %arg0, %arg1[1, 2] : vector<5x7xi8> into
/// vector<2x3x5x7xi8>
///
/// AFTER:
/// %0 = vector.shape_cast %arg1 : vector<5x7xi8> to vector<6x5x7xi8>
/// %1 = vector.insert %arg0, %0[5] : vector<5x7xi8> into vector<6x5x7xi8>
/// %o = vector.shape_cast %1 : vector<6x5x7xi8> to vector<2x3x5x7xi8>
struct CollapseOuterInsert final
    : public OpRewritePatternWithPreCondition<vector::InsertOp> {

  CollapseOuterInsert(MLIRContext *context, const PreCondition &p,
                      PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::InsertOp>(context, p,
                                                           benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::InsertOp insertOp,
                               PatternRewriter &rewriter) const final {

    VectorType dstType = insertOp.getDestVectorType();

    SmallVector<OpFoldResult> position = insertOp.getMixedPosition();
    if (!collapseFront(position, dstType, rewriter))
      return rewriter.notifyMatchFailure(
          insertOp, "failed to collapse the outermost 2 dimensions");

    Value reducedIn = getReducedValue(rewriter, insertOp.getDest(), 1);

    Value newInsert = rewriter.create<vector::InsertOp>(
        insertOp.getLoc(), insertOp.getValueToStore(), reducedIn, position);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        insertOp, insertOp.getType(), newInsert);

    return success();
  }
};

/// Collapse the outermost 2 dimensions of a vector.splat
///
/// BEFORE:
/// %o = vector.splat %arg0 : vector<2x3x5xi8>
///
/// AFTER:
/// %0 = vector.splat %arg0 : vector<2x15xi8>
/// %o = vector.shape_cast %0 : vector<2x15xi8> to vector<2x3x5xi8>
struct CollapseInnerSplat final
    : public OpRewritePatternWithPreCondition<vector::SplatOp> {

  CollapseInnerSplat(MLIRContext *context, const PreCondition &p,
                     PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::SplatOp>(context, p, benefit) {
  }

  LogicalResult
  postConditionMatchAndRewrite(vector::SplatOp splatOp,
                               PatternRewriter &rewriter) const final {

    auto splatType = splatOp.getType();
    auto reducedType = getReducedType(splatType);
    if (reducedType == splatType)
      return rewriter.notifyMatchFailure(splatOp, "splat type is irreducible");

    rewriter.replaceOpWithNewOp<vector::SplatOp>(splatOp, reducedType,
                                                 splatOp.getOperand());

    return success();
  }
};

/// Convert an vector.insert (rank-2 to rank-1) to a vector.shuffle
///
/// Conversion of higher rank vector.insert operations to vector.shuffle
/// require rank-reducing patterns to be applied first.
///
/// BEFORE
/// %insert_2d = vector.insert %src %dst [ position ]
///
/// AFTER
/// %src_1d = vector.shape_cast %src : [...]
/// %dst_1d = vector.shape_cast %dst : [...]
/// %out_1d = vector.shuffle %dst_1d, %src_1d [ shuffle_indices ]
/// %out_2d = vector.shape_cast %out_1d : [...]
///
/// `shuffle_indices` is computed from `position`.
struct ConvertInsertToShuffle final
    : public OpRewritePatternWithPreCondition<vector::InsertOp> {

  ConvertInsertToShuffle(MLIRContext *context, const PreCondition &p,
                         PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::InsertOp>(context, p,
                                                           benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::InsertOp insertOp,
                               PatternRewriter &rewriter) const final {

    if (insertOp.getDestVectorType().isScalable())
      return rewriter.notifyMatchFailure(
          insertOp, "conversion to shuffle not possible with scalable vectors");

    const Value toStore = insertOp.getValueToStore();
    auto toInsertType = dyn_cast<VectorType>(toStore.getType());
    if (!toInsertType || toInsertType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          insertOp,
          "this pattern only handles the case where rank-1 vectors are stored");

    VectorType dstType = insertOp.getType();
    if (dstType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          insertOp, "this pattern only handles the case where rank-2 vectors "
                    "are inserted into");

    int64_t offset = insertOp.getStaticPosition()[0];
    if (offset == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          insertOp, "conversion to shuffle requires all static positions");

    int64_t nSmall = toInsertType.getNumElements();
    int64_t nLarge = dstType.getNumElements();

    SmallVector<int64_t> mask(nLarge);
    auto *iter = mask.begin() + offset * nSmall;
    std::iota(mask.begin(), mask.end(), 0);
    std::iota(iter, iter + nSmall, nLarge);

    VectorType collapsedType = getReducedType(dstType, 1);
    assert(collapsedType != dstType && "rank-2 to rank-1 failed");

    Value collapsedDst = getReducedValue(rewriter, insertOp.getDest(), 1);
    Value shuffled = rewriter.create<vector::ShuffleOp>(
        insertOp.getLoc(), collapsedType, collapsedDst, toStore, mask);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(insertOp, dstType,
                                                     shuffled);

    return success();
  }
};

/// Convert a vector.extract (rank-2 to rank-1) to a vector.shuffle
///
/// Conversion of higher rank vector.extract operations to vector.shuffle
/// require rank-reducing patterns to be applied first.
///
/// BEFORE:
/// %extract_1d = vector.extract %src_2d [ position ]
///
/// AFTER:
/// %src_1d = vector.shape_cast %src_2d : [...]
/// %out_1d = vector.shuffle %src_1d, %src_1d [ shuffle_indices ] [...]
struct ConvertExtractToShuffle final
    : public OpRewritePatternWithPreCondition<vector::ExtractOp> {

  ConvertExtractToShuffle(MLIRContext *context, const PreCondition &p,
                          PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::ExtractOp>(context, p,
                                                            benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::ExtractOp extractOp,
                               PatternRewriter &rewriter) const final {

    if (extractOp.getSourceVectorType().isScalable())
      return rewriter.notifyMatchFailure(
          extractOp,
          "conversion to shuffle not possible with scalable vectors");

    VectorType srcType = extractOp.getSourceVectorType();
    if (srcType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          extractOp, "this pattern only handles the case where rank-2 vectors "
                     "are extracted from");

    auto dstType = dyn_cast<VectorType>(extractOp.getType());
    if (!dstType || dstType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          extractOp, "this pattern only handles the case where rank-1 vectors "
                     "are extracted");

    int64_t offset = extractOp.getStaticPosition()[0];
    if (offset == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          extractOp, "conversion to shuffle requires all static positions");

    Value collapsedIn = getReducedValue(rewriter, extractOp.getVector(), 1);
    int64_t nSmall = dstType.getNumElements();

    SmallVector<int64_t> mask(nSmall);
    std::iota(mask.begin(), mask.end(), offset * nSmall);

    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        extractOp, dstType, collapsedIn, collapsedIn, mask);

    return success();
  }
};

/// BEFORE
/// %out_nd = vector.extract_strided_slice %source_nd
///         { offsets = [..], strides = [..], sizes = [..] }
///
/// AFTER
/// %source_1d = vector.shape_cast %source_nd [...]
/// %out_1d    = vector.shuffle %source_1d, %source_1d [ shuffle_indices_1d ]
/// %out_nd    = vector.shape_cast %out_1d [...]
///
/// `shuffle_indices_1d` is computed using the offsets and sizes of the
/// original vector.extract_strided_slice operation.
struct ConvertExtractStridedToShuffle final
    : public OpRewritePatternWithPreCondition<vector::ExtractStridedSliceOp> {

  ConvertExtractStridedToShuffle(MLIRContext *context, const PreCondition &p,
                                 PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::ExtractStridedSliceOp>(
            context, p, benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                               PatternRewriter &rewriter) const final {

    if (extractOp.hasNonUnitStrides())
      return rewriter.notifyMatchFailure(
          extractOp, "conversion to shuffle requires unit strides");

    if (extractOp.getSourceVectorType().isScalable())
      return rewriter.notifyMatchFailure(
          extractOp,
          "conversion to shuffle not possible with scalable vectors");

    VectorType extractType = extractOp.getType();

    FailureOr<SmallVector<int64_t>> offsets =
        intsFromArrayAttr(extractOp.getOffsets());
    if (failed(offsets))
      return rewriter.notifyMatchFailure(extractOp,
                                         "failed to get integer offsets");

    SmallVector<int64_t> indices = getStridedSliceInsertionIndices(
        extractType.getShape(), extractOp.getSourceVectorType().getShape(),
        offsets.value());

    FailureOr<Value> flatIn =
        getCollapsedToRankOne(extractOp.getVector(), rewriter);
    FailureOr<Type> flatOutType = getRankOneType(extractType);

    assert(succeeded(flatIn) && succeeded(flatOutType) &&
           "failed to linearize input or type");

    Value shuffled = rewriter.create<vector::ShuffleOp>(
        extractOp.getLoc(), flatOutType.value(), flatIn.value(), flatIn.value(),
        indices);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(extractOp, extractType,
                                                     shuffled);

    return success();
  }
};

/// This pattern converts a vector.insert_strided_slice operation into a
/// vector.shuffle operation that has rank-1 (linearized) operands and result.
///
/// BEFORE
/// %0 = vector.insert_strided_slice %to_store, %into
///             {offsets = [1, 0, 0, 0], strides = [1, 1]}
///                  : vector<2x2xi8> into vector<2x1x3x2xi8>
/// AFTER
/// %to_store_1d
///          = vector.shape_cast %to_store : vector<2x2xi8> to vector<4xi8>
/// %into_1d = vector.shape_cast %into : vector<2x1x3x2xi8> to vector<12xi8>
/// %out_1d  = vector.shuffle %into_1d, %to_store_1d [ shuffle_indices_1d ]
/// %out_nd  = vector.shape_cast %out_1d : vector<12xi8> to vector<2x1x3x2xi8>
///
/// where shuffle_indices_1d in this case is
///     [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 10, 11].
///                        ^^^^^^^^^^^^^^
///                          to_store_1d
struct ConvertInsertStridedToShuffle final
    : public OpRewritePatternWithPreCondition<vector::InsertStridedSliceOp> {

  ConvertInsertStridedToShuffle(MLIRContext *context, const PreCondition &p,
                                PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::InsertStridedSliceOp>(
            context, p, benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::InsertStridedSliceOp insertOp,
                               PatternRewriter &rewriter) const final {

    if (insertOp.hasNonUnitStrides())
      return rewriter.notifyMatchFailure(
          insertOp, "conversion to shuffle requires unit strides");

    if (insertOp.getSourceVectorType().isScalable())
      return rewriter.notifyMatchFailure(
          insertOp, "conversion to shuffle not possible with scalable vectors");

    TypedValue<VectorType> toStore = insertOp.getValueToStore();
    VectorType inputType = toStore.getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();

    VectorType outputType = insertOp.getType();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t nOutputElements = outputType.getNumElements();

    FailureOr<SmallVector<int64_t>> offsets =
        intsFromArrayAttr(insertOp.getOffsets());
    if (failed(offsets))
      return rewriter.notifyMatchFailure(insertOp,
                                         "failed to get integer offsets");

    SmallVector<int64_t> sliceIndices = getStridedSliceInsertionIndices(
        inputShape, outputShape, offsets.value());

    SmallVector<int64_t> indices(nOutputElements);
    std::iota(indices.begin(), indices.end(), 0);
    for (auto [index, sliceIndex] : llvm::enumerate(sliceIndices)) {
      indices[sliceIndex] = index + nOutputElements;
    }

    FailureOr<Value> flatToStore = getCollapsedToRankOne(toStore, rewriter);
    assert(succeeded(flatToStore) && "failed to linearize value to store");

    FailureOr<Value> flatDest =
        getCollapsedToRankOne(insertOp.getDest(), rewriter);
    assert(succeeded(flatDest) &&
           "failed to linearize destination of insert strided slice");

    FailureOr<Type> flatDestType = getRankOneType(outputType);
    assert(succeeded(flatDestType) &&
           "failed to get rank-1 type for destination of insert strided slice");

    Value shuffled = rewriter.create<vector::ShuffleOp>(
        insertOp.getLoc(), flatDestType.value(), flatDest.value(),
        flatToStore.value(), indices);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(insertOp, outputType,
                                                     shuffled);

    return success();
  }
};

/// This pattern converts the CreateMaskOp to work on a linearized vector.
///
/// BEFORE:
///   vector.create_mask %arg0, %arg1 : vector<1x4xi1>
///
/// AFTER:
///   %zero = arith.constant 0 : index
///   %cmpi = arith.cmpi sgt, %arg0, %zero : index
///   %index = arith.index_cast %cmpi : i1 to index
///   %mul = arith.andi %index, %arg1 : index
///   %mask = vector.create_mask %mul : vector<4xi1>
///   %shape_cast = vector.shape_cast %mask : vector<4xi1> to vector<1x4xi1>
///
/// There can be at most one non-unit dimension in the mask type.
struct SqueezeCreateMaskUnitDims final
    : OpRewritePatternWithPreCondition<vector::CreateMaskOp> {

  SqueezeCreateMaskUnitDims(MLIRContext *context, const PreCondition &p,
                            PatternBenefit benefit = 1)
      : OpRewritePatternWithPreCondition<vector::CreateMaskOp>(context, p,
                                                               benefit) {}

  LogicalResult
  postConditionMatchAndRewrite(vector::CreateMaskOp maskOp,
                               PatternRewriter &rewriter) const final {

    VectorType maskType = maskOp.getType();

    if (!isCreateMaskWithAtMostOneNonUnit(maskOp))
      return rewriter.notifyMatchFailure(
          maskOp, "mask type must have at most one non-unit dimension");

    Location loc = maskOp.getLoc();

    FailureOr<Type> flatType = getRankOneType(maskType);
    if (failed(flatType))
      return rewriter.notifyMatchFailure(maskOp,
                                         "failed to convert to rank-1 type");

    if (flatType.value() == maskType)
      return rewriter.notifyMatchFailure(
          maskOp, "mask type is already rank linearized");

    // First, get the product of (clamped) mask sizes in the unit-dimensions.
    Value prod = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    int nonUnitDim = -1;
    for (unsigned i = 0; i < maskType.getRank(); ++i) {

      Value dimRange = maskOp.getOperands()[i];
      int64_t dimSize = maskType.getDimSize(i);
      if (dimSize <= 1) {
        Value nxt = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sgt, dimRange, zero);
        prod = rewriter.create<arith::MulIOp>(loc, prod, nxt);

      } else {
        assert(nonUnitDim == -1 && "at most 1 non-unit expected");
        nonUnitDim = i;
      }
    }

    prod =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), prod);

    // Finally, multiply by the size in the dimension that is not unit.
    if (nonUnitDim != -1) {
      Value v = maskOp.getOperands()[nonUnitDim];
      prod = rewriter.create<arith::MulIOp>(loc, prod, v);
    }

    Value newMask =
        rewriter.create<vector::CreateMaskOp>(loc, flatType.value(), prod);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(maskOp, maskType, newMask);

    return success();
  }
};

enum class LinearizePattern {
  // Patterns to collapse the 2 innermost dimensions:
  CollapseInnerVectorizable,
  CollapseInnerVectorBitCast,
  CollapseInnerVectorShuffle,
  CollapseInnerExtractStrided,
  CollapseInnerInsertStrided,
  CollapseInnerExtract,
  CollapseInnerInsert,
  CollapseInnerSplat,

  // Patterns to collapse the 2 outermost dimensions:
  CollapseOuterExtract,
  CollapseOuterInsert,

  // Patterns to convert ops to shuffle:
  ConvertInsertToShuffle,
  ConvertExtractToShuffle,
  ConvertInsertStridedToShuffle,
  ConvertExtractStridedToShuffle,

  // Patterns to remove unit dimensions:
  SqueezeCreateMaskUnitDims,

  // The number of patterns in this enum:
  N
};

/// This class contains functions to control the set of linearization patterns
/// to include for the conversion, and their priority.
struct VectorLinearizePatterns {

public:
  /// By default all patterns are enabled and have benefit 1.
  VectorLinearizePatterns() {
    enabled.fill(true);
    benefits.fill(PatternBenefit(1));
  }

  /// Add the patterns enabled for the conversion to `patterns`.
  void addToPatternSet(RewritePatternSet &patterns,
                       const PreCondition &pc) const;

  VectorLinearizePatterns &enable(LinearizePattern id, bool e = true) {
    enabled[static_cast<unsigned>(id)] = e;
    return *this;
  }

  VectorLinearizePatterns &enableAll(bool e = true) {
    enabled.fill(e);
    return *this;
  }

  bool isEnabled(LinearizePattern id) const {
    return enabled[static_cast<unsigned>(id)];
  }

  PatternBenefit getBenefit(LinearizePattern id) const {
    return benefits[static_cast<unsigned>(id)];
  }

  VectorLinearizePatterns &setBenefits(PatternBenefit benefit) {
    benefits.fill(benefit);
    return *this;
  }

  VectorLinearizePatterns &setBenefit(LinearizePattern id,
                                      PatternBenefit benefit) {
    getBenefitRef(id) = benefit;
    return *this;
  }

  VectorLinearizePatterns &incrementBenefit(LinearizePattern id,
                                            unsigned inc = 1) {
    getBenefitRef(id) = getBenefit(id).getBenefit() + 1;
    return *this;
  }

private:
  std::array<bool, static_cast<unsigned>(LinearizePattern::N)> enabled;
  std::array<PatternBenefit, static_cast<unsigned>(LinearizePattern::N)>
      benefits;

  PatternBenefit &getBenefitRef(LinearizePattern id) {
    unsigned idInt = static_cast<unsigned>(id);
    assert(idInt < static_cast<unsigned>(LinearizePattern::N) &&
           "invalid linearization pattern id");
    return benefits[idInt];
  }

  template <typename T>
  void addIfEnabled(RewritePatternSet &patterns,
                    std::function<LogicalResult(Operation *)> preCond,
                    LinearizePattern id) const {
    if (isEnabled(id)) {
      patterns.add<T>(patterns.getContext(), preCond, getBenefit(id));
    }
  }
};

void VectorLinearizePatterns::addToPatternSet(RewritePatternSet &patterns,
                                              const PreCondition &pc) const {

  using LP = LinearizePattern;

  addIfEnabled<CollapseInnerVectorizable>(patterns, pc,
                                          LP::CollapseInnerVectorizable);

  addIfEnabled<CollapseInnerVectorBitCast>(patterns, pc,
                                           LP::CollapseInnerVectorBitCast);

  addIfEnabled<CollapseInnerVectorShuffle>(patterns, pc,
                                           LP::CollapseInnerVectorShuffle);

  addIfEnabled<CollapseInnerExtractStrided>(patterns, pc,
                                            LP::CollapseInnerExtractStrided);

  addIfEnabled<CollapseInnerInsertStrided>(patterns, pc,
                                           LP::CollapseInnerInsertStrided);

  addIfEnabled<CollapseInnerExtract>(patterns, pc, LP::CollapseInnerExtract);

  addIfEnabled<CollapseInnerInsert>(patterns, pc, LP::CollapseInnerExtract);

  addIfEnabled<CollapseOuterExtract>(patterns, pc, LP::CollapseOuterExtract);

  addIfEnabled<CollapseOuterInsert>(patterns, pc, LP::CollapseOuterInsert);

  addIfEnabled<CollapseInnerSplat>(patterns, pc, LP::CollapseInnerSplat);

  addIfEnabled<ConvertInsertToShuffle>(patterns, pc,
                                       LP::ConvertInsertToShuffle);

  addIfEnabled<ConvertExtractToShuffle>(patterns, pc,
                                        LP::ConvertExtractToShuffle);

  addIfEnabled<ConvertInsertStridedToShuffle>(
      patterns, pc, LP::ConvertInsertStridedToShuffle);

  addIfEnabled<ConvertExtractStridedToShuffle>(
      patterns, pc, LP::ConvertExtractStridedToShuffle);

  addIfEnabled<SqueezeCreateMaskUnitDims>(patterns, pc,
                                          LP::SqueezeCreateMaskUnitDims);
}

} // namespace

void vector::populateForVectorLinearize(RewritePatternSet &patterns,
                                        const PreCondition &preCondition,
                                        PatternBenefit benefit) {
  VectorLinearizePatterns vlp;

  // We want to perform rank reduction as much as possible before converting to
  // shuffle. We do this by setting the benefit of all the patterns that do not
  // convert to shuffle to be 1 higher.
  vlp.enableAll(true)
      .setBenefits(benefit.getBenefit() + 1)
      .setBenefit(LinearizePattern::ConvertExtractToShuffle, benefit)
      .setBenefit(LinearizePattern::ConvertInsertToShuffle, benefit)
      .setBenefit(LinearizePattern::ConvertExtractStridedToShuffle, benefit)
      .setBenefit(LinearizePattern::ConvertInsertStridedToShuffle, benefit);

  vlp.addToPatternSet(patterns, preCondition);
}

void vector::populateForStridedRankReduction(RewritePatternSet &patterns,
                                             PatternBenefit benefit) {
  VectorLinearizePatterns()
      .enableAll(false)
      .setBenefits(benefit)
      .enable(LinearizePattern::CollapseInnerExtractStrided)
      .enable(LinearizePattern::CollapseInnerInsertStrided)
      .addToPatternSet(patterns, [](auto x) { return success(); });
}
