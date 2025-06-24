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
#include "mlir/Interfaces/InferTypeOpInterface.h"
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

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
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
  SmallVector<OpFoldResult> newOffsets, newLengths;
  SmallVector<OpFoldResult> newLows, newHighs;
  // Set to true if the original data source is not read at all.
  bool hasZeroLen = false;
  // Same as hasZeroLen, but for dynamic dimension sizes. This condition
  // is true if the original data source turns out to be unused at runtime.
  Value dynHasZeroLenCond;

  int64_t rank = padOp.getSourceType().getRank();
  // Only unit stride supported.
  SmallVector<OpFoldResult> newStrides(rank, b.getIndexAttr(1));
  for (unsigned dim = 0; dim < rank; ++dim) {
    auto low = padOp.getMixedLowPad()[dim];
    bool hasLowPad = !isZeroInteger(low);
    auto high = padOp.getMixedHighPad()[dim];
    bool hasHighPad = !isZeroInteger(high);
    auto offset = offsets[dim];
    auto length = sizes[dim];
    // If the dim has no padding, we dont need to calculate new values for that
    // dim as the exisiting ones are correct even after the pattern.
    if (!hasLowPad && !hasHighPad) {
      newOffsets.push_back(offset);
      newLengths.push_back(length);
      newLows.push_back(low);
      newHighs.push_back(high);
      continue;
    }

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
    // srcSize - newOffset represents how much length we have available
    // and length - newLow represents how much length we want at most.
    // Note that there are many ways to order this indexing math to compute
    // newLength, but we want to make sure that the final affine.min ops in the
    // sequence are bounding the index to as small a value as possible. If
    // ValueBoundsOpInterface is used, this calculation will get upper bounds
    // from the affine.min ops, so we want to use the smallest known value to
    // set the bound at the end of the computation sequence. In this case, the
    // index will be upper bounded by length - newLow.
    OpFoldResult newLength = min(sub(srcSize, newOffset), sub(length, newLow));
    // Optimization: If low = 0, then newLow = 0. then newLength >= 0 assuming
    // length >= 0.
    if (hasLowPad)
      newLength = max(newLength, zero);
    newLengths.push_back(newLength);

    // Check if newLength is zero. In that case, no SubTensorOp should be
    // executed.
    if (isZeroInteger(newLength)) {
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
    auto newSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, padOp.getSource(), newOffsets, newLengths, newStrides);
    auto newPadOp = b.create<PadOp>(
        loc, Type(), newSliceOp, newLows, newHighs,
        /*nofold=*/padOp.getNofold(),
        getPrunedAttributeList(padOp, PadOp::getAttributeNames()));

    // Copy region to new PadOp.
    IRMapping bvm;
    padOp.getRegion().cloneInto(&newPadOp.getRegion(), bvm);

    // Cast result and return.
    return std::make_tuple(newPadOp, newSliceOp);
  };

  // Rewrite extract_slice(pad(x)) into a GenerateOp it is statically known that
  // the original data source x is not used.
  if (hasZeroLen) {
    Operation *generateOp = createGenerateOp();
    return TilingResult{{generateOp},
                        {castResult(generateOp->getResult(0))},
                        /*generatedSlices=*/{}};
  }

  // If there are dynamic dimensions: Generate an scf.if check to avoid
  // creating SliceOps with result dimensions of size 0 at runtime.
  if (generateZeroSliceGuard && dynHasZeroLenCond) {
    Operation *thenOp;
    Operation *elseOp;
    Operation *sliceOp;
    auto result = b.create<scf::IfOp>(
        loc, dynHasZeroLenCond,
        /*thenBuilder=*/
        [&](OpBuilder &b, Location loc) {
          thenOp = createGenerateOp();
          b.create<scf::YieldOp>(loc, castResult(thenOp->getResult(0)));
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location loc) {
          std::tie(elseOp, sliceOp) = createPadOfExtractSlice();
          b.create<scf::YieldOp>(loc, castResult(elseOp->getResult(0)));
        });
    return TilingResult{
        {elseOp}, SmallVector<Value>(result->getResults()), {sliceOp}};
  }

  auto [newPadOp, sliceOp] = createPadOfExtractSlice();
  return TilingResult{
      {newPadOp}, {castResult(newPadOp->getResult(0))}, {sliceOp}};
}

void mlir::tensor::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    tensor::PadOp::attachInterface<PadOpTiling>(*ctx);
  });
}
