//===- VectorEmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// The emulation only works on 1D memref types.
/// To make this work on N-D memref, we need to linearize the offset.
///
/// For example, to emulate i4 to i8, the following op:
///
/// %0 = memref.load %arg0[%v0, %v1] :
///                  memref<?x?xi4, strided<[?, ?], offset: ?>>
///
/// can be replaced with
///
/// %b, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %0
///
/// %linearized_offset = %v0 * %stride#0 + %v1 * %stride#1
/// %linearized_size = %size0 * %size1
/// %scaled_linear_offset = %linearized_offset / 8 * 4
/// %scaled_base_offset = %offset / 8 * 4
///
/// %linearized = memref.reinterpret_cast %b, offset = [%scaled_base_offset],
///                      sizes = [%linearized_size], strides = [%stride#1]
///
/// %new_load = vector.load %linearized[%scaled_linear_offset] :
///                         memref<?xi8, strided<[?], offset: ?>>

static Value
linearizeVectorLoad(Location loc, MemRefType sourceType, int srcBits,
                    int dstBits, SmallVector<Value> indices,
                    memref::ExtractStridedMetadataOp stridedMetadata,
                    int numElements, OpBuilder &builder) {
  auto srcElementType = sourceType.getElementType();
  unsigned sourceRank = indices.size();

  Value baseBuffer = stridedMetadata.getBaseBuffer();
  SmallVector<Value> baseSizes = stridedMetadata.getSizes();
  SmallVector<Value> baseStrides = stridedMetadata.getStrides();
  Value baseOffset = stridedMetadata.getOffset();
  assert(indices.size() == baseStrides.size());

  // Create the affine symbols and values for linearization.
  SmallVector<AffineExpr> symbols(2 * sourceRank + 2);
  bindSymbolsList(builder.getContext(), MutableArrayRef{symbols});
  symbols[0] = builder.getAffineSymbolExpr(0);
  AffineExpr addMulMap = symbols.front();
  AffineExpr mulMap = symbols.front();

  SmallVector<OpFoldResult> offsetValues(2 * sourceRank + 2);
  offsetValues[0] = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> sizeValues(sourceRank + 1);
  sizeValues[0] = builder.getIndexAttr(1);

  for (unsigned i = 0; i < sourceRank; ++i) {
    unsigned offsetIdx = 2 * i + 1;
    addMulMap = addMulMap + symbols[offsetIdx] * symbols[offsetIdx + 1];
    offsetValues[offsetIdx] = indices[i];
    offsetValues[offsetIdx + 1] = baseStrides[i];

    unsigned sizeIdx = i + 1;
    mulMap = mulMap * symbols[sizeIdx];
    sizeValues[sizeIdx] = baseSizes[i];
  }

  // Adjust linearizedOffset by the scale factor (dstBits / srcBits).
  OpFoldResult scaler = builder.getIndexAttr(dstBits / srcBits);
  AffineExpr scaledAddMulMap = addMulMap.floorDiv(symbols.back());
  offsetValues.back() = scaler;

  OpFoldResult linearizedOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, scaledAddMulMap, offsetValues);
  OpFoldResult linearizedSize =
      affine::makeComposedFoldedAffineApply(builder, loc, mulMap, sizeValues);

  // Adjust baseOffset by the scale factor (dstBits / srcBits).
  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  OpFoldResult adjustBaseOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, s0.floorDiv(s1), {baseOffset, scaler});

  // Flatten n-D MemRef to 1-D MemRef.
  auto layoutAttr = StridedLayoutAttr::get(sourceType.getContext(),
                                           ShapedType::kDynamic, {1});
  int64_t staticShape = sourceType.hasStaticShape()
                            ? sourceType.getNumElements()
                            : ShapedType::kDynamic;
  auto flattenMemrefType = MemRefType::get(
      staticShape, srcElementType, layoutAttr, sourceType.getMemorySpace());

  auto reinterpret = builder.create<memref::ReinterpretCastOp>(
      loc, flattenMemrefType, baseBuffer,
      getValueOrCreateConstantIndexOp(builder, loc, adjustBaseOffset),
      getValueOrCreateConstantIndexOp(builder, loc, linearizedSize),
      baseStrides.back());

  return builder.create<vector::LoadOp>(
      loc, VectorType::get(numElements, srcElementType),
      reinterpret.getResult(),
      getValueOrCreateConstantIndexOp(builder, loc, linearizedOffset));
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertVectorLoad
//===----------------------------------------------------------------------===//

struct ConvertVectorLoad final : OpConversionPattern<vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto sourceType = cast<MemRefType>(adaptor.getBase().getType());
    Type oldElementType = op.getType().getElementType();
    Type newElementType = sourceType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

    // Adjust the number of elements to load when emulating narrow types,
    // and then cast back to the original type with vector.bitcast op.
    // Here only the 1-D vector load is considered, and the N-D memref types
    // should be linearized.
    // For example, to emulate i4 to i8, the following op:
    //
    // %1 = vector.load %0[%c0, %c0] : memref<3x4xi4>, vector<4xi4>
    //
    // can be replaced with
    //
    // %1 = vector.load %0[%linear_index] : memref<12xi8>, vector<2xi8>
    // %2 = vector.bitcast %1 : vector<2xi8> to vector<4xi4>
    //
    // TODO: Currently, only the even number of elements loading is supported.
    // To deal with the odd number of elements, one has to extract the
    // subvector at the proper offset after bit-casting.

    auto origElements = op.getVectorType().getNumElements();
    if (origElements % scale != 0)
      return failure();

    auto stridedMetadata = rewriter.create<memref::ExtractStridedMetadataOp>(
        loc, adaptor.getBase());

    auto numElements = int(std::ceil(double(origElements) / scale));
    auto newLoad = linearizeVectorLoad(loc, sourceType, srcBits, dstBits,
                                       adaptor.getIndices(), stridedMetadata,
                                       numElements, rewriter);

    numElements *= scale;
    auto castType = VectorType::get(numElements, oldElementType);
    auto bitCast = rewriter.create<vector::BitCastOp>(loc, castType, newLoad);

    rewriter.replaceOp(op, bitCast->getResult(0));
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

void vector::populateVectorNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {

  // Populate `vector.*` conversion patterns.
  patterns.add<ConvertVectorLoad>(typeConverter, patterns.getContext());
}
