//===- TensorTilingInterface.cpp - Tiling Interface  models *- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

struct PadOpTiling : public TilingInterface::ExternalModel<PadOpTiling, PadOp> {

  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    ReifiedRankedShapedTypeDims reifiedShapes;
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
    (void)reifyShapedTypeInterface.reifyResultShapes(b, reifiedShapes);

    auto padOp = cast<PadOp>(op);
    SmallVector<OpFoldResult> mixedSizes = getAsOpFoldResult(reifiedShapes[0]);
    Value initTensor = b.create<linalg::InitTensorOp>(
        op->getLoc(), mixedSizes, padOp.getResultType().getElementType());
    return {initTensor};
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    auto padOp = cast<PadOp>(op);
    SmallVector<StringRef> iteratorTypes(padOp.getResultType().getRank(),
                                         getParallelIteratorTypeName());
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    ReifiedRankedShapedTypeDims reifiedShapes;
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
    (void)reifyShapedTypeInterface.reifyResultShapes(b, reifiedShapes);

    Location loc = op->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
    for (const auto &ub : enumerate(reifiedShapes[0]))
      loopRanges[ub.index()].size = ub.value();
    return loopRanges;
  }

  SmallVector<Operation *>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    Operation *result =
        tensor::bubbleUpPadSlice(b, cast<PadOp>(op), offsets, sizes);
    if (!result)
      return {};
    return {result};
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

} // namespace

Operation *tensor::bubbleUpPadSlice(OpBuilder &b, tensor::PadOp padOp,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    bool generateZeroSliceGuard) {
  // Only constant padding value supported.
  Value padValue = padOp.getConstantPaddingValue();
  if (!padValue)
    return nullptr;

  // Helper variables and functions for various arithmetic operations. These
  // are used extensively for computing new offset/length and padding values.
  Location loc = padOp->getLoc();
  AffineExpr dim0, dim1;
  bindDims(b.getContext(), dim0, dim1);
  // Add two integers.
  auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
  auto add = [&](Value v1, Value v2) {
    return b.createOrFold<AffineApplyOp>(loc, addMap, ValueRange{v1, v2});
  };
  // Subtract two integers.
  auto subMap = AffineMap::get(2, 0, {dim0 - dim1});
  auto sub = [&](Value v1, Value v2) {
    return b.createOrFold<AffineApplyOp>(loc, subMap, ValueRange{v1, v2});
  };
  // Take the minimum of two integers.
  auto idMap = AffineMap::getMultiDimIdentityMap(2, b.getContext());
  auto min = [&](Value v1, Value v2) {
    return b.createOrFold<AffineMinOp>(loc, idMap, ValueRange{v1, v2});
  };
  // Take the maximum of two integers.
  auto max = [&](Value v1, Value v2) {
    return b.createOrFold<AffineMaxOp>(loc, idMap, ValueRange{v1, v2});
  };
  // Zero index-typed integer.
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);

  // Helper function for filling static/dynamic low/high padding indices
  // vectors of PadOp.
  auto appendIndex = [&](Value val, SmallVector<Value> &dynIndices,
                         SmallVector<int64_t> &staticIndices) {
    if (auto constInt = getConstantIntValue(val)) {
      staticIndices.push_back(*constInt);
    } else {
      staticIndices.push_back(ShapedType::kDynamicSize);
      dynIndices.push_back(val);
    }
  };

  // Compute new offsets, lengths, low padding, high padding.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;
  SmallVector<Value> newLows, newHighs;
  SmallVector<int64_t> staticNewLows, staticNewHighs;
  // Set to true if the original data source is not read at all.
  bool hasZeroLen = false;
  // Same as hasZeroLen, but for dynamic dimension sizes. This condition
  // is true if the original data source turns out to be unused at runtime.
  Value dynHasZeroLenCond;

  int64_t rank = padOp.getSourceType().getRank();
  for (unsigned dim = 0; dim < rank; ++dim) {
    auto low =
        getValueOrCreateConstantIndexOp(b, loc, padOp.getMixedLowPad()[dim]);
    bool hasLowPad = getConstantIntValue(low) != static_cast<int64_t>(0);
    auto high =
        getValueOrCreateConstantIndexOp(b, loc, padOp.getMixedHighPad()[dim]);
    bool hasHighPad = getConstantIntValue(high) != static_cast<int64_t>(0);
    auto offset = getValueOrCreateConstantIndexOp(b, loc, offsets[dim]);
    auto length = getValueOrCreateConstantIndexOp(b, loc, sizes[dim]);
    auto srcSize = b.createOrFold<tensor::DimOp>(loc, padOp.getSource(), dim);

    // The new amount of low padding is `low - offset`. Except for the case
    // where none of the low padding is read. In that case, the new amount of
    // low padding is zero.
    //
    // Optimization: If low = 0, then newLow = 0.
    Value newLow = hasLowPad ? max(zero, sub(low, offset)) : zero;
    appendIndex(newLow, newLows, staticNewLows);

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
    Value newOffset = hasLowPad ? min(max(sub(offset, low), zero), srcSize)
                                : min(offset, srcSize);
    newOffsets.push_back(getAsOpFoldResult(newOffset));

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
    Value endLoc = hasLowPad
                       ? min(max(add(sub(offset, low), length), zero), srcSize)
                       : min(add(offset, length), srcSize);
    Value newLength = sub(endLoc, newOffset);
    newLengths.push_back(getAsOpFoldResult(newLength));

    // Check if newLength is zero. In that case, no SubTensorOp should be
    // executed.
    if (auto newLengthInt = getConstantIntValue(newLength)) {
      hasZeroLen |= *newLengthInt == 0;
    } else {
      Value check = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            newLength, zero);
      dynHasZeroLenCond =
          dynHasZeroLenCond
              ? b.create<arith::OrIOp>(loc, check, dynHasZeroLenCond)
              : check;
    }

    // The amount of high padding is simply the number of elements remaining,
    // so that the result has the same length as the original ExtractSliceOp.
    // As an optimization, if the original high padding is zero, then the new
    // high padding must also be zero.
    Value newHigh = hasHighPad ? sub(sub(length, newLength), newLow) : zero;
    appendIndex(newHigh, newHighs, staticNewHighs);

    // Only unit stride supported.
    newStrides.push_back(b.getIndexAttr(1));
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynDims;
  SmallVector<int64_t> shape;
  dispatchIndexOpFoldResults(sizes, dynDims, shape, ShapedType::kDynamicSize);
  RankedTensorType resultType =
      RankedTensorType::get(shape, padOp.getResultType().getElementType());

  // Insert cast to ensure that types match. (May be folded away.)
  auto castResult = [&](Value val) -> Operation * {
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
    return castResult(generateOp);
  };

  // Emit a SliceOp and a PadOp. Should not be used in cases where
  // the result shape of the new SliceOp has a zero dimension.
  auto createPadOfExtractSlice = [&]() {
    // Create pad(extract_slice(x)).
    auto newSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, padOp.getSource(), newOffsets, newLengths, newStrides);
    auto newPadOp = b.create<PadOp>(loc, newSliceOp, staticNewLows,
                                    staticNewHighs, newLows, newHighs);

    // Copy region to new PadOp.
    BlockAndValueMapping bvm;
    padOp.getRegion().cloneInto(&newPadOp.getRegion(), bvm);

    // Cast result and return.
    return castResult(newPadOp);
  };

  // Rewrite extract_slice(pad(x)) into a GenerateOp it is statically known that
  // the original data source x is not used.
  if (hasZeroLen)
    return createGenerateOp();

  // If there are dynamic dimensions: Generate an scf.if check to avoid
  // creating SliceOps with result dimensions of size 0 at runtime.
  if (generateZeroSliceGuard && dynHasZeroLenCond) {
    auto result = b.create<scf::IfOp>(
        loc, resultType, dynHasZeroLenCond,
        /*thenBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, createGenerateOp()->getResult(0));
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, createPadOfExtractSlice()->getResult(0));
        });
    return result;
  }
  return createPadOfExtractSlice();
}

void mlir::tensor::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    tensor::PadOp::attachInterface<PadOpTiling>(*ctx);
  });
}
