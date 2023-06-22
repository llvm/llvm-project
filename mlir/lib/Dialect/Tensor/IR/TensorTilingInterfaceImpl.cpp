//===- TensorTilingInterface.cpp - Tiling Interface  models *- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

struct PadOpTiling : public TilingInterface::ExternalModel<PadOpTiling, PadOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto padOp = cast<PadOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        padOp.getResultType().getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    ReifiedRankedShapedTypeDims reifiedShapes;
    (void)reifyResultShapes(b, op, reifiedShapes);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
    for (const auto &ub : enumerate(reifiedShapes[0]))
      loopRanges[ub.index()].size = ub.value();
    return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    FailureOr<TilingResult> result =
        tensor::bubbleUpPadSlice(b, cast<PadOp>(op), offsets, sizes);
    if (failed(result))
      return failure();
    return result.value();
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
};

template <typename OpTy>
static SmallVector<Range> getPackUnPackIterationDomain(OpTy op,
                                                       OpBuilder &builder) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  OpBuilder::InsertionGuard g(builder);
  int64_t rank = (std::is_same<OpTy, PackOp>::value) ? op.getSourceRank()
                                                     : op.getDestRank();
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  ReifiedRankedShapedTypeDims resultShape;
  (void)reifyResultShapes(builder, op, resultShape);
  SmallVector<Range> loopBounds(rank);
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].stride = one;
    loopBounds[dim].size = resultShape[0][dim];
  }
  return loopBounds;
}

static void applyPermToRange(SmallVector<OpFoldResult> &offsets,
                             SmallVector<OpFoldResult> &sizes,
                             ArrayRef<int64_t> permutation) {
  if (permutation.empty())
    return;
  applyPermutationToVector<OpFoldResult>(offsets, permutation);
  applyPermutationToVector<OpFoldResult>(sizes, permutation);
}

struct PackOpTiling
    : public TilingInterface::ExternalModel<PackOpTiling, PackOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    // Note that here we only consider untiled dimensions and outer tiled data
    // dimensions, the inner tiled data dimensions are materialized when
    // building the body of the operation.
    auto packOp = cast<PackOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        packOp.getSourceRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    return getPackUnPackIterationDomain<PackOp>(cast<PackOp>(op), b);
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto packOp = cast<PackOp>(op);
    Location loc = packOp.getLoc();

    // The tiling is applied on interchanged dimensions. We have to undo the
    // interchange to map sizes and offsets to the original input.
    int64_t inputRank = packOp.getSourceRank();
    SmallVector<OpFoldResult> origOffsets(offsets.begin(), offsets.end());
    SmallVector<OpFoldResult> origSizes(sizes.begin(), sizes.end());
    applyPermToRange(origOffsets, origSizes,
                     invertPermutationVector(packOp.getOuterDimsPerm()));

    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        packOp.getDimAndTileMapping();
    SmallVector<OpFoldResult> srcDimValues =
        tensor::getMixedSizes(b, loc, packOp.getSource());
    SmallVector<OpFoldResult> inputIndices, inputSizes;
    for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
      using AV = affine::AffineValueExpr;
      affine::AffineBuilder ab(b, loc);
      AffineExpr dim0, dim1, sym;
      bindDims(b.getContext(), dim0, dim1);
      bindSymbols(b.getContext(), sym);
      if (dimAndTileMapping.count(dim)) {
        // If the data dimension is tiled, the i-th index is the product of
        // offset_i and tile_i, and the i-th size is the product of sizes_i and
        // tile_i.
        auto avOffset = AV(dim0).bind(origOffsets[dim]);
        auto avSize = AV(dim0).bind(origSizes[dim]);
        auto avTileSize = AV(sym).bind(dimAndTileMapping[dim]);
        inputIndices.push_back(ab.mul(avOffset, avTileSize));
        inputSizes.push_back(ab.mul(avSize, avTileSize));
      } else {
        inputIndices.push_back(origOffsets[dim]);
        inputSizes.push_back(origSizes[dim]);
      }

      // Limit the size of the input operand for incomplete tiles.
      if (packOp.getPaddingValue()) {
        OpFoldResult dimSize = srcDimValues[dim];
        auto avDimSize = AV(dim0).bind(dimSize);
        auto avInputIdx = AV(dim1).bind(inputIndices.back());
        inputSizes.back() =
            ab.min({inputSizes.back(), ab.sub(avDimSize, avInputIdx)});
      }
    }

    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(inputRank, oneAttr);

    SmallVector<Value> tiledOperands;
    tiledOperands.push_back(b.create<ExtractSliceOp>(
        loc, packOp.getSource(), inputIndices, inputSizes, strides));

    SmallVector<OpFoldResult> outputOffsets, outputSizes;
    if (failed(getResultTilePosition(op, b, 0, offsets, sizes, outputOffsets,
                                     outputSizes)))
      return {};

    strides.append(packOp.getDestRank() - inputRank, oneAttr);
    auto extractSlice = b.create<ExtractSliceOp>(
        loc, packOp.getDest(), outputOffsets, outputSizes, strides);
    tiledOperands.push_back(extractSlice);

    if (auto val = packOp.getPaddingValue())
      tiledOperands.push_back(val);
    for (auto tile : packOp.getInnerTiles())
      tiledOperands.push_back(tile);

    Operation *tiledPackOp = b.create<PackOp>(
        loc, TypeRange{extractSlice.getType()}, tiledOperands, op->getAttrs());

    return TilingResult{{tiledPackOp},
                        SmallVector<Value>(tiledPackOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    // The iteration domain is over outer dimensions of packed layout. In this
    // context, the outer dimensions of `resultOffsets` are `offsets`. The
    // inner dimensions of `resultOffsets` are zeros because tiling is not
    // applied to them.
    auto packOp = cast<PackOp>(op);
    int64_t inputRank = packOp.getSourceRank();
    int64_t outputRank = packOp.getDestRank();
    auto zeroAttr = b.getI64IntegerAttr(0);
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultOffsets.append(outputRank - inputRank, zeroAttr);

    ReifiedRankedShapedTypeDims outputShape;
    (void)reifyResultShapes(b, packOp, outputShape);
    resultSizes.assign(sizes.begin(), sizes.end());
    for (auto dataTileDim : llvm::seq<unsigned>(inputRank, outputRank))
      resultSizes.push_back(outputShape[0][dataTileDim]);

    return success();
  }
};

struct UnpackTileDimInfo {
  bool isAlignedToInnerTileSize;
  OpFoldResult sourceOffset;
  OpFoldResult sourceSize;
  OpFoldResult resultOffset;
  OpFoldResult destExpandedSize;
};

/// Returns the needed information for tiling unpack op on `tileDim` with given
/// `tileOffset` and `tileSize`. For more details, see the comment of the
/// `getTiledImplementation`.
static UnpackTileDimInfo getUnpackTileDimInfo(OpBuilder &b, UnPackOp unpackOp,
                                              int64_t tileDim,
                                              OpFoldResult tileOffset,
                                              OpFoldResult tileSize) {
  UnpackTileDimInfo info;
  Attribute zeroAttr = b.getIndexAttr(0);
  Attribute oneAttr = b.getIndexAttr(1);
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      unpackOp.getDimAndTileMapping();
  // The dimension is not one of packed data dimension.
  if (!dimAndTileMapping.count(tileDim)) {
    info.isAlignedToInnerTileSize = true;
    info.sourceOffset = tileOffset;
    info.sourceSize = tileSize;
    info.resultOffset = zeroAttr;
    info.destExpandedSize = tileSize;
    return info;
  }

  Location loc = unpackOp.getLoc();
  using AV = affine::AffineValueExpr;
  affine::AffineBuilder ab(b, loc);
  AffineExpr dim0, dim1, sym0;
  bindDims(b.getContext(), dim0, dim1);
  bindSymbols(b.getContext(), sym0);

  OpFoldResult innerTileSize = dimAndTileMapping[tileDim];

  info.isAlignedToInnerTileSize = false;
  FailureOr<int64_t> cstSize = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB,
      getValueOrCreateConstantIndexOp(b, loc, tileSize), /*dim=*/std::nullopt,
      /*stopCondition=*/nullptr, /*closedUB=*/true);
  std::optional<int64_t> cstInnerSize = getConstantIntValue(innerTileSize);
  if (!failed(cstSize) && cstInnerSize) {
    if (*cstSize % *cstInnerSize == 0)
      info.isAlignedToInnerTileSize = true;

    // If the tiling size equals to the inner tiling size, the outer dims are
    // always 1.
    if (*cstInnerSize == *cstSize) {
      auto lhs = AV(dim0).bind(tileOffset);
      auto rhs = AV(dim1).bind(innerTileSize);
      info.sourceOffset = ab.floor(lhs, rhs);
      info.sourceSize = oneAttr;
      info.resultOffset = zeroAttr;
      info.destExpandedSize = tileSize;
      return info;
    }
  }

  if (info.isAlignedToInnerTileSize) {
    info.sourceOffset =
        ab.floor(AV(dim0).bind(tileOffset), AV(dim1).bind(innerTileSize));
    info.resultOffset = zeroAttr;
    info.destExpandedSize = tileSize;

    // The ceilDiv is needed here because there could be incomplete tile even
    // it is perfect tiling cases. E.g.,
    //   %0 = unpack tensor<33x2xf32> into tensor<64xf32>
    // If the tiling size is 32, there will be 3 tiles. Two of them have
    // size=32; one of them have size=2. The size is represented using
    // affine_min op; we need ceilDiv.
    info.sourceSize =
        ab.ceil(AV(dim0).bind(tileSize), AV(dim1).bind(innerTileSize));
    return info;
  }

  affine::DivModValue firstCoord = affine::getDivMod(
      b, loc, getValueOrCreateConstantIndexOp(b, loc, tileOffset),
      getValueOrCreateConstantIndexOp(b, loc, innerTileSize));
  OpFoldResult tileExclusiveBound =
      ab.add(AV(dim0).bind(tileOffset), AV(dim1).bind(tileSize));
  affine::DivModValue lastCoord = affine::getDivMod(
      b, loc,
      getValueOrCreateConstantIndexOp(
          b, loc,
          ab.sub(AV(dim0).bind(tileExclusiveBound), AV(dim1).bind(oneAttr))),
      getValueOrCreateConstantIndexOp(b, loc, innerTileSize));

  OpFoldResult lengthMinusOne = ab.sub(AV(dim0).bind(lastCoord.quotient),
                                       AV(dim1).bind(firstCoord.quotient));
  info.sourceSize =
      ab.add(AV(dim0).bind(lengthMinusOne), AV(dim1).bind(oneAttr));
  info.sourceOffset = firstCoord.quotient;
  info.resultOffset = firstCoord.remainder;
  // Do not create an Affine ops for expanded size because the affine op is too
  // complicated which would trigger an issue in affine ops simplification.
  info.destExpandedSize = b.createOrFold<arith::MulIOp>(
      loc, getValueOrCreateConstantIndexOp(b, loc, info.sourceSize),
      getValueOrCreateConstantIndexOp(b, loc, innerTileSize));
  return info;
}

struct UnPackOpTiling
    : public TilingInterface::ExternalModel<UnPackOpTiling, UnPackOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto unpackOp = cast<UnPackOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        unpackOp.getDestRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    return getPackUnPackIterationDomain<UnPackOp>(cast<UnPackOp>(op), b);
  }

  /// There are two cases in tiling unpack ops. If the tiling size is aligned to
  /// the inner tile size, the corresponding tiles of source are all complete.
  /// Otherwise, there are in-complete tiles. We will need to expand the slice
  /// of source for getting complete tiles. The tiled unpack op unpacks more
  /// data from source, so We'll need an extract_slice op to shift and truncate
  /// the output.
  /// Take Nn_to_N as an example. Say that N=32, n=8, and tiling_size=15. The
  /// coordinates of second tile (i.e., result[15..31]) are
  /// [(1, 7), (2, 0,), (2, 1) ... (3, 6), (3, 7)]. The first row and the last
  /// row are incomplete tiles. To represent the unpack op, we have to complete
  /// the rows. I.e., the input coordinates would start with (1, 0); end with
  /// (3, 7). In this context, the tiled unpack produces a (3 * n) elements
  /// because there are 3 rows in total. Follow by a tensor.extract_slice op, we
  /// can get the actual result.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto unpackOp = cast<UnPackOp>(op);
    int64_t srcRank = unpackOp.getSourceRank();
    int64_t destRank = unpackOp.getDestRank();
    int64_t numInnerTiles = srcRank - destRank;
    Location loc = unpackOp.getLoc();

    // The perfect tiling case indicates that the tiling sizes are multiple of
    // inner_tile_size. In this context, no extra data is needed when
    // representing the tiled unpack op.
    bool isPerfectTilingCase = true;
    Attribute oneAttr = b.getIndexAttr(1);
    SmallVector<OpFoldResult> sliceSrcStrides(destRank, oneAttr);
    SmallVector<OpFoldResult> sliceSrcIndices, sliceSrcSizes;
    SmallVector<OpFoldResult> destExpandedSizes, resultOffsetsFromDest;
    for (auto dim : llvm::seq<int64_t>(0, destRank)) {
      UnpackTileDimInfo info =
          getUnpackTileDimInfo(b, unpackOp, dim, offsets[dim], sizes[dim]);
      if (!info.isAlignedToInnerTileSize)
        isPerfectTilingCase = false;
      sliceSrcIndices.push_back(info.sourceOffset);
      sliceSrcSizes.push_back(info.sourceSize);
      destExpandedSizes.push_back(info.destExpandedSize);
      resultOffsetsFromDest.push_back(info.resultOffset);
    }

    // The tiling is applied on destination dimensions. We have to apply the
    // interchange on source dimensions if outer_dims_perm is set.
    applyPermToRange(sliceSrcIndices, sliceSrcSizes,
                     unpackOp.getOuterDimsPerm());
    Attribute zeroAttr = b.getIndexAttr(0);
    sliceSrcIndices.append(numInnerTiles, zeroAttr);
    sliceSrcSizes.append(unpackOp.getMixedTiles());
    sliceSrcStrides.append(numInnerTiles, oneAttr);
    Value sliceSource =
        b.create<ExtractSliceOp>(loc, unpackOp.getSource(), sliceSrcIndices,
                                 sliceSrcSizes, sliceSrcStrides);

    SmallVector<OpFoldResult> destStrides(destRank, oneAttr);
    Value sliceDest;
    if (isPerfectTilingCase) {
      sliceDest = b.create<ExtractSliceOp>(loc, unpackOp.getDest(), offsets,
                                           sizes, destStrides);
    } else {
      sliceDest = b.create<EmptyOp>(loc, destExpandedSizes,
                                    unpackOp.getDestType().getElementType());
    }

    SmallVector<Value> tiledOperands = {sliceSource, sliceDest};
    for (auto tile : unpackOp.getInnerTiles())
      tiledOperands.push_back(tile);

    Operation *tiledUnpackOp = b.create<UnPackOp>(
        loc, TypeRange{sliceDest.getType()}, tiledOperands, op->getAttrs());

    if (isPerfectTilingCase)
      return TilingResult{{tiledUnpackOp},
                          SmallVector<Value>(tiledUnpackOp->getResults())};

    auto extractSlice =
        b.create<ExtractSliceOp>(loc, tiledUnpackOp->getResult(0),
                                 resultOffsetsFromDest, sizes, destStrides);
    return TilingResult{{tiledUnpackOp}, {extractSlice.getResult()}};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets = llvm::to_vector(offsets);
    resultSizes = llvm::to_vector(sizes);
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    FailureOr<TilingResult> tilingResult =
        getTiledImplementation(op, b, offsets, sizes);
    if (failed(tilingResult))
      return failure();
    return tilingResult.value();
  }
};

} // namespace

FailureOr<TilingResult> tensor::bubbleUpPadSlice(OpBuilder &b,
                                                 tensor::PadOp padOp,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes,
                                                 bool generateZeroSliceGuard) {
  // Only constant padding value supported.
  Value padValue = padOp.getConstantPaddingValue();
  if (!padValue)
    return failure();

  // Helper variables and functions for various arithmetic operations. These
  // are used extensively for computing new offset/length and padding values.
  Location loc = padOp->getLoc();
  AffineExpr dim0, dim1;
  bindDims(b.getContext(), dim0, dim1);
  // Add two integers.
  auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
  auto add = [&](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineApply(b, loc, addMap, {v1, v2});
  };
  // Subtract two integers.
  auto subMap = AffineMap::get(2, 0, {dim0 - dim1});
  auto sub = [&](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineApply(b, loc, subMap, {v1, v2});
  };
  // Take the minimum of two integers.
  auto idMap = AffineMap::getMultiDimIdentityMap(2, b.getContext());
  auto min = [&](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineMin(b, loc, idMap, {v1, v2});
  };
  // Take the maximum of two integers.
  auto max = [&](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineMax(b, loc, idMap, {v1, v2});
  };
  // Zero index-typed integer.
  OpFoldResult zero = b.getIndexAttr(0);

  // Compute new offsets, lengths, low padding, high padding.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;
  SmallVector<OpFoldResult> newLows, newHighs;
  // Set to true if the original data source is not read at all.
  bool hasZeroLen = false;
  // Same as hasZeroLen, but for dynamic dimension sizes. This condition
  // is true if the original data source turns out to be unused at runtime.
  Value dynHasZeroLenCond;

  int64_t rank = padOp.getSourceType().getRank();
  for (unsigned dim = 0; dim < rank; ++dim) {
    auto low = padOp.getMixedLowPad()[dim];
    bool hasLowPad = !isConstantIntValue(low, 0);
    auto high = padOp.getMixedHighPad()[dim];
    bool hasHighPad = !isConstantIntValue(high, 0);
    auto offset = offsets[dim];
    auto length = sizes[dim];
    auto srcSize = tensor::getMixedSize(b, loc, padOp.getSource(), dim);

    // The new amount of low padding is `low - offset`. Except for the case
    // where none of the low padding is read. In that case, the new amount of
    // low padding is zero.
    //
    // Optimization: If low = 0, then newLow = 0.
    OpFoldResult newLow = hasLowPad ? max(zero, sub(low, offset)) : zero;
    newLows.push_back(newLow);

    // Start reading the data from position `offset - low`. Since the original
    // read may have started in the low padding zone, this value could be
    // negative. Therefore, start reading from:
    //
    // max(offset - low, 0)
    //
    // The original read could also have started in the high padding zone.
    // In that case, set the offset to the end of source tensor. The new
    // ExtractSliceOp length will be zero in that case. (Effectively reading
    // no data from the source.)
    //
    // Optimization: If low = 0, then the formula can be simplified.
    OpFoldResult newOffset = hasLowPad
                                 ? min(max(sub(offset, low), zero), srcSize)
                                 : min(offset, srcSize);
    newOffsets.push_back(newOffset);

    // The original ExtractSliceOp was reading until position `offset +
    // length`. Therefore, the corresponding position within the source tensor
    // is:
    //
    // offset + length - low
    //
    // In case the original ExtractSliceOp stopped reading within the low
    // padding zone, this value can be negative. In that case, the end
    // position of the read should be zero. (Similar to newOffset.)
    //
    // The original read could also have stopped in the high padding zone.
    // In that case, set the end positition of the read should be the end of
    // the source tensor. (Similar to newOffset.)
    //
    // endLoc = min(max(offset - low + length, 0), srcSize)
    //
    // The new ExtractSliceOp length is `endLoc - newOffset`.
    //
    // Optimization: If low = 0, then the formula can be simplified.
    OpFoldResult endLoc =
        hasLowPad ? min(max(add(sub(offset, low), length), zero), srcSize)
                  : min(add(offset, length), srcSize);
    OpFoldResult newLength = sub(endLoc, newOffset);
    newLengths.push_back(newLength);

    // Check if newLength is zero. In that case, no SubTensorOp should be
    // executed.
    if (isConstantIntValue(newLength, 0)) {
      hasZeroLen = true;
    } else if (!hasZeroLen) {
      Value check = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          getValueOrCreateConstantIndexOp(b, loc, newLength),
          getValueOrCreateConstantIndexOp(b, loc, zero));
      dynHasZeroLenCond =
          dynHasZeroLenCond
              ? b.create<arith::OrIOp>(loc, check, dynHasZeroLenCond)
              : check;
    }

    // The amount of high padding is simply the number of elements remaining,
    // so that the result has the same length as the original ExtractSliceOp.
    // As an optimization, if the original high padding is zero, then the new
    // high padding must also be zero.
    OpFoldResult newHigh =
        hasHighPad ? sub(sub(length, newLength), newLow) : zero;
    newHighs.push_back(newHigh);

    // Only unit stride supported.
    newStrides.push_back(b.getIndexAttr(1));
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynDims;
  SmallVector<int64_t> shape;
  dispatchIndexOpFoldResults(sizes, dynDims, shape);
  RankedTensorType resultType =
      RankedTensorType::get(shape, padOp.getResultType().getElementType());

  // Insert cast to ensure that types match. (May be folded away.)
  auto castResult = [&](Value val) -> Value {
    if (resultType == val.getType())
      return val;
    return b.create<tensor::CastOp>(loc, resultType, val);
  };

  // In cases where the original data source is unused: Emit a GenerateOp and
  // do not generate a SliceOp. (The result shape of the SliceOp would
  // have a dimension of size 0, the semantics of which is unclear.)
  auto createGenerateOp = [&]() {
    // Create GenerateOp.
    auto generateOp = b.create<tensor::GenerateOp>(
        loc, resultType, dynDims,
        [&](OpBuilder &builder, Location gLoc, ValueRange indices) {
          builder.create<tensor::YieldOp>(gLoc, padValue);
        });
    return generateOp;
  };

  // Emit a SliceOp and a PadOp. Should not be used in cases where
  // the result shape of the new SliceOp has a zero dimension.
  auto createPadOfExtractSlice = [&]() {
    // Create pad(extract_slice(x)).
    Value newSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, padOp.getSource(), newOffsets, newLengths, newStrides);
    auto newPadOp = b.create<PadOp>(
        loc, Type(), newSliceOp, newLows, newHighs,
        /*nofold=*/padOp.getNofold(),
        getPrunedAttributeList(padOp, PadOp::getAttributeNames()));

    // Copy region to new PadOp.
    IRMapping bvm;
    padOp.getRegion().cloneInto(&newPadOp.getRegion(), bvm);

    // Cast result and return.
    return newPadOp;
  };

  // Rewrite extract_slice(pad(x)) into a GenerateOp it is statically known that
  // the original data source x is not used.
  if (hasZeroLen) {
    Operation *generateOp = createGenerateOp();
    return TilingResult{{generateOp}, {castResult(generateOp->getResult(0))}};
  }

  // If there are dynamic dimensions: Generate an scf.if check to avoid
  // creating SliceOps with result dimensions of size 0 at runtime.
  if (generateZeroSliceGuard && dynHasZeroLenCond) {
    Operation *thenOp;
    Operation *elseOp;
    auto result = b.create<scf::IfOp>(
        loc, dynHasZeroLenCond,
        /*thenBuilder=*/
        [&](OpBuilder &b, Location loc) {
          thenOp = createGenerateOp();
          b.create<scf::YieldOp>(loc, castResult(thenOp->getResult(0)));
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location loc) {
          elseOp = createPadOfExtractSlice();
          b.create<scf::YieldOp>(loc, castResult(elseOp->getResult(0)));
        });
    return TilingResult{{elseOp}, SmallVector<Value>(result->getResults())};
  }

  Operation *newPadOp = createPadOfExtractSlice();
  return TilingResult{{newPadOp}, {castResult(newPadOp->getResult(0))}};
}

void mlir::tensor::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    tensor::PadOp::attachInterface<PadOpTiling>(*ctx);
    tensor::PackOp::attachInterface<PackOpTiling>(*ctx);
    tensor::UnPackOp::attachInterface<UnPackOpTiling>(*ctx);
  });
}

void mlir::tensor::registerTilingInterfaceExternalModelsForPackUnPackOps(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    tensor::PackOp::attachInterface<PackOpTiling>(*ctx);
    tensor::UnPackOp::attachInterface<UnPackOpTiling>(*ctx);
  });
}
