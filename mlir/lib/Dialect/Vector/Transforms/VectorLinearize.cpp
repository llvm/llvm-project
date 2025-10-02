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

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <numeric>
#include <optional>

using namespace mlir;

static FailureOr<Attribute>
linearizeConstAttr(Location loc, ConversionPatternRewriter &rewriter,
                   VectorType resType, Attribute value) {

  if (auto dstElementsAttr = dyn_cast<DenseElementsAttr>(value)) {
    if (resType.isScalable() && !isa<SplatElementsAttr>(value))
      return rewriter.notifyMatchFailure(
          loc,
          "Cannot linearize a constant scalable vector that's not a splat");

    return dstElementsAttr.reshape(resType);
  }

  if (auto poisonAttr = dyn_cast<ub::PoisonAttr>(value))
    return poisonAttr;

  return rewriter.notifyMatchFailure(loc, "unsupported attr type");
}

namespace {

struct LinearizeConstantLike final
    : OpTraitConversionPattern<OpTrait::ConstantLike> {
  using OpTraitConversionPattern::OpTraitConversionPattern;

  LinearizeConstantLike(const TypeConverter &typeConverter,
                        MLIRContext *context, PatternBenefit benefit = 1)
      : OpTraitConversionPattern(typeConverter, context, benefit) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(loc, "expected 1 result");

    const TypeConverter &typeConverter = *getTypeConverter();
    auto resType =
        typeConverter.convertType<VectorType>(op->getResult(0).getType());
    assert(resType && "expected 1-D vector type");

    StringAttr attrName = rewriter.getStringAttr("value");
    Attribute value = op->getAttr(attrName);
    if (!value)
      return rewriter.notifyMatchFailure(loc, "no 'value' attr");

    FailureOr<Attribute> newValue =
        linearizeConstAttr(loc, rewriter, resType, value);
    if (failed(newValue))
      return failure();

    FailureOr<Operation *> convertResult =
        convertOpResultTypes(op, /*operands=*/{}, typeConverter, rewriter);
    if (failed(convertResult))
      return failure();

    Operation *newOp = *convertResult;
    newOp->setAttr(attrName, *newValue);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct LinearizeVectorizable final
    : OpTraitConversionPattern<OpTrait::Vectorizable> {
  using OpTraitConversionPattern::OpTraitConversionPattern;

public:
  LinearizeVectorizable(const TypeConverter &typeConverter,
                        MLIRContext *context, PatternBenefit benefit = 1)
      : OpTraitConversionPattern(typeConverter, context, benefit) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Operation *> newOp =
        convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return success();
  }
};

template <typename TOp>
static bool stridesAllOne(TOp op) {
  static_assert(
      std::is_same_v<TOp, vector::ExtractStridedSliceOp> ||
          std::is_same_v<TOp, vector::InsertStridedSliceOp>,
      "expected vector.extract_strided_slice or vector.insert_strided_slice");
  ArrayAttr strides = op.getStrides();
  return llvm::all_of(strides, isOneInteger);
}

/// Convert an array of attributes into a vector of integers, if possible.
static FailureOr<SmallVector<int64_t>> intsFromArrayAttr(ArrayAttr attrs) {
  if (!attrs)
    return failure();
  SmallVector<int64_t> ints;
  ints.reserve(attrs.size());
  for (auto attr : attrs) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      ints.push_back(intAttr.getInt());
    } else {
      return failure();
    }
  }
  return ints;
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
SmallVector<int64_t> static getStridedSliceInsertionIndices(
    ArrayRef<int64_t> small, ArrayRef<int64_t> large,
    ArrayRef<int64_t> offsets) {

  // Example of alignment between, `large`, `small` and `offsets`:
  //    large  =  4, 5, 6, 7, 8
  //    small  =     1, 6, 7, 8
  //  offsets  =  2, 3, 0
  //
  // `offsets` has implicit trailing 0s, `small` has implicit leading 1s.
  assert((large.size() >= small.size()) &&
         "rank of 'large' cannot be lower than rank of 'small'");
  assert((large.size() >= offsets.size()) &&
         "rank of 'large' cannot be lower than the number of offsets");
  unsigned delta = large.size() - small.size();
  unsigned nOffsets = offsets.size();
  auto getSmall = [&](int64_t i) -> int64_t {
    return i >= delta ? small[i - delta] : 1;
  };
  auto getOffset = [&](int64_t i) -> int64_t {
    return i < nOffsets ? offsets[i] : 0;
  };

  // Using 2 vectors of indices, at each iteration populate the updated set of
  // indices based on the old set of indices, and the size of the small vector
  // in the current iteration.
  SmallVector<int64_t> indices{0};
  int64_t stride = 1;
  for (int i = large.size() - 1; i >= 0; --i) {
    int64_t currentSize = indices.size();
    int64_t smallSize = getSmall(i);
    int64_t nextSize = currentSize * smallSize;
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

/// This pattern converts a vector.extract_strided_slice operation into a
/// vector.shuffle operation that has a rank-1 (linearized) operand and result.
///
/// For example, the following:
///
/// ```
///   vector.extract_strided_slice %source
///         { offsets = [..], strides = [..], sizes = [..] }
/// ```
///
/// is converted to :
/// ```
///   %source_1d = vector.shape_cast %source
///   %out_1d    = vector.shuffle %source_1d, %source_1d [ shuffle_indices_1d ]
///   %out_nd    = vector.shape_cast %out_1d
/// ```
///
/// `shuffle_indices_1d` is computed using the offsets and sizes of the original
/// vector.extract_strided_slice operation.
struct LinearizeVectorExtractStridedSlice final
    : public mlir::OpConversionPattern<mlir::vector::ExtractStridedSliceOp> {
  using Base::Base;
  LinearizeVectorExtractStridedSlice(const TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractStridedSliceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    VectorType flatOutputType = getTypeConverter()->convertType<VectorType>(
        extractStridedSliceOp.getType());
    assert(flatOutputType && "vector type expected");

    // Expect a legalization failure if the strides are not all 1 (if ever the
    // verifier for extract_strided_slice allows non-1 strides).
    if (!stridesAllOne(extractStridedSliceOp)) {
      return rewriter.notifyMatchFailure(
          extractStridedSliceOp,
          "extract_strided_slice with strides != 1 not supported");
    }

    FailureOr<SmallVector<int64_t>> offsets =
        intsFromArrayAttr(extractStridedSliceOp.getOffsets());
    if (failed(offsets)) {
      return rewriter.notifyMatchFailure(extractStridedSliceOp,
                                         "failed to get integer offsets");
    }

    ArrayRef<int64_t> inputShape =
        extractStridedSliceOp.getSourceVectorType().getShape();

    ArrayRef<int64_t> outputShape = extractStridedSliceOp.getType().getShape();

    SmallVector<int64_t> indices = getStridedSliceInsertionIndices(
        outputShape, inputShape, offsets.value());

    Value srcVector = adaptor.getSource();
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        extractStridedSliceOp, flatOutputType, srcVector, srcVector, indices);
    return success();
  }
};

/// This pattern converts a vector.insert_strided_slice operation into a
/// vector.shuffle operation that has rank-1 (linearized) operands and result.
///
/// For example, the following:
/// ```
///  %0 = vector.insert_strided_slice %to_store, %into
///             {offsets = [1, 0, 0, 0], strides = [1, 1]}
///                  : vector<2x2xi8> into vector<2x1x3x2xi8>
/// ```
///
/// is converted to
/// ```
///  %to_store_1d
///           = vector.shape_cast %to_store : vector<2x2xi8> to vector<4xi8>
///  %into_1d = vector.shape_cast %into : vector<2x1x3x2xi8> to vector<12xi8>
///  %out_1d  = vector.shuffle %into_1d, %to_store_1d [ shuffle_indices_1d ]
///  %out_nd  = vector.shape_cast %out_1d : vector<12xi8> to vector<2x1x3x2xi8>
/// ```
///
/// where shuffle_indices_1d in this case is
///     [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 10, 11].
///                        ^^^^^^^^^^^^^^
///                          to_store_1d
///
struct LinearizeVectorInsertStridedSlice final
    : public mlir::OpConversionPattern<mlir::vector::InsertStridedSliceOp> {
  using Base::Base;
  LinearizeVectorInsertStridedSlice(const TypeConverter &typeConverter,
                                    MLIRContext *context,
                                    PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::InsertStridedSliceOp insertStridedSliceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Expect a legalization failure if the strides are not all 1 (if ever the
    // verifier for insert_strided_slice allows non-1 strides).
    if (!stridesAllOne(insertStridedSliceOp)) {
      return rewriter.notifyMatchFailure(
          insertStridedSliceOp,
          "insert_strided_slice with strides != 1 not supported");
    }

    VectorType inputType = insertStridedSliceOp.getValueToStore().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();

    VectorType outputType = insertStridedSliceOp.getType();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t nOutputElements = outputType.getNumElements();

    FailureOr<SmallVector<int64_t>> offsets =
        intsFromArrayAttr(insertStridedSliceOp.getOffsets());
    if (failed(offsets)) {
      return rewriter.notifyMatchFailure(insertStridedSliceOp,
                                         "failed to get integer offsets");
    }
    SmallVector<int64_t> sliceIndices = getStridedSliceInsertionIndices(
        inputShape, outputShape, offsets.value());

    SmallVector<int64_t> indices(nOutputElements);
    std::iota(indices.begin(), indices.end(), 0);
    for (auto [index, sliceIndex] : llvm::enumerate(sliceIndices)) {
      indices[sliceIndex] = index + nOutputElements;
    }

    Value flatToStore = adaptor.getValueToStore();
    Value flatDest = adaptor.getDest();
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(insertStridedSliceOp,
                                                   flatDest.getType(), flatDest,
                                                   flatToStore, indices);
    return success();
  }
};

/// This pattern converts the ShuffleOp that works on nD (n > 1)
/// vectors to a ShuffleOp that works on linearized vectors.
/// Following,
///   vector.shuffle %v1, %v2 [ shuffle_indices ]
/// is converted to :
///   %v1_1d = vector.shape_cast %v1
///   %v2_1d = vector.shape_cast %v2
///   %out_1d = vector.shuffle %v1_1d, %v2_1d [ shuffle_indices_1d ]
///   %out_nd = vector.shape_cast %out_1d
// `shuffle_indices_1d` is computed using the sizes and `shuffle_indices`
/// of the original shuffle operation.
struct LinearizeVectorShuffle final
    : public OpConversionPattern<vector::ShuffleOp> {
  using Base::Base;
  LinearizeVectorShuffle(const TypeConverter &typeConverter,
                         MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstType =
        getTypeConverter()->convertType<VectorType>(shuffleOp.getType());
    assert(dstType && "vector type destination expected.");

    Value vec1 = adaptor.getV1();
    Value vec2 = adaptor.getV2();
    int shuffleSliceLen = 1;
    int rank = shuffleOp.getV1().getType().getRank();

    // If rank > 1, we need to do the shuffle in the granularity of slices
    // instead of scalars. Size of the slice is equal to the rank-1 innermost
    // dims. Mask of the shuffle op specifies which slice to take from the
    // outermost dim.
    if (rank > 1) {
      llvm::ArrayRef<int64_t> shape = shuffleOp.getV1().getType().getShape();
      for (unsigned i = 1; i < shape.size(); ++i) {
        shuffleSliceLen *= shape[i];
      }
    }

    // For each value in the mask, we generate the indices of the source vectors
    // that need to be shuffled to the destination vector. If shuffleSliceLen >
    // 1 we need to shuffle the slices (consecutive shuffleSliceLen number of
    // elements) instead of scalars.
    ArrayRef<int64_t> mask = shuffleOp.getMask();
    int64_t totalSizeOfShuffledElmnts = mask.size() * shuffleSliceLen;
    llvm::SmallVector<int64_t, 2> indices(totalSizeOfShuffledElmnts);
    for (auto [i, value] : llvm::enumerate(mask)) {
      std::iota(indices.begin() + shuffleSliceLen * i,
                indices.begin() + shuffleSliceLen * (i + 1),
                shuffleSliceLen * value);
    }

    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(shuffleOp, dstType, vec1,
                                                   vec2, indices);
    return success();
  }
};

/// This pattern linearizes `vector.extract` operations. It generates a 1-D
/// version of the `vector.extract` operation when extracting a scalar from a
/// vector. It generates a 1-D `vector.shuffle` operation when extracting a
/// subvector from a larger vector.
///
/// Example #1:
///
///     %0 = vector.extract %arg0[1]: vector<8x2xf32> from vector<2x8x2xf32>
///
///   is converted to:
///
///     %0 = vector.shape_cast %arg0 : vector<2x8x2xf32> to vector<32xf32>
///     %1 = vector.shuffle %0, %0 [16, 17, 18, 19, 20, 21, 22, 23,
///                                 24, 25, 26, 27, 28, 29, 30, 31] :
///            vector<32xf32>, vector<32xf32>
///     %2 = vector.shape_cast %1 : vector<16xf32> to vector<8x2xf32>
///
/// Example #2:
///
///     %0 = vector.extract %arg0[1, 2] : i32 from vector<2x4xi32>
///
///   is converted to:
///
///     %0 = vector.shape_cast %arg0 : vector<2x4xi32> to vector<8xi32>
///     %1 = vector.extract %0[6] : i32 from vector<8xi32>
///
struct LinearizeVectorExtract final
    : public OpConversionPattern<vector::ExtractOp> {
  using Base::Base;
  LinearizeVectorExtract(const TypeConverter &typeConverter,
                         MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}
  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(extractOp.getType());
    assert(dstTy && "expected 1-D vector type");

    // Dynamic position is not supported.
    if (extractOp.hasDynamicPosition())
      return rewriter.notifyMatchFailure(extractOp,
                                         "dynamic position is not supported.");

    llvm::ArrayRef<int64_t> shape = extractOp.getSource().getType().getShape();
    int64_t size = extractOp.getSource().getType().getNumElements();

    // Compute linearized offset.
    int64_t linearizedOffset = 0;
    llvm::ArrayRef<int64_t> offsets = extractOp.getStaticPosition();
    for (auto [i, off] : llvm::enumerate(offsets)) {
      size /= shape[i];
      linearizedOffset += offsets[i] * size;
    }

    Value srcVector = adaptor.getSource();
    if (!isa<VectorType>(extractOp.getType())) {
      // Scalar case: generate a 1-D extract.
      Value result = rewriter.createOrFold<vector::ExtractOp>(
          extractOp.getLoc(), srcVector, linearizedOffset);
      rewriter.replaceOp(extractOp, result);
      return success();
    }

    // Vector case: generate a shuffle.

    llvm::SmallVector<int64_t, 2> indices(size);
    std::iota(indices.begin(), indices.end(), linearizedOffset);
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(extractOp, dstTy, srcVector,
                                                   srcVector, indices);

    return success();
  }
};

/// This pattern linearizes `vector.insert` operations. It generates a 1-D
/// version of the `vector.insert` operation when inserting a scalar into a
/// vector. It generates a 1-D `vector.shuffle` operation when inserting a
/// vector into another vector.
///
/// Example #1:
///
///     %0 = vector.insert %source, %destination[0] :
///       vector<2x4xf32> into vector<2x2x4xf32>
///
///   is converted to:
///
///     %0 = vector.shape_cast %source : vector<2x4xf32> to vector<8xf32>
///     %1 = vector.shape_cast %destination :
///            vector<2x2x4xf32> to vector<16xf32>
///     %2 = vector.shuffle %1, %0 [16, 17, 18, 19, 20, 21, 22, 23
///                                  8, 9, 10, 11, 12, 13, 14, 15] :
///            vector<16xf32>, vector<8xf32>
///     %3 = vector.shape_cast %2 : vector<16xf32> to vector<2x2x4xf32>
///
/// Example #2:
///
///     %0 = vector.insert %source, %destination[1, 2]: f32 into vector<2x4xf32>
///
///   is converted to:
///
///     %0 = vector.shape_cast %destination : vector<2x4xf32> to vector<8xf32>
///     %1 = vector.insert %source, %0[6]: f32 into vector<8xf32>
///     %2 = vector.shape_cast %1 : vector<8xf32> to vector<2x4xf32>
///
struct LinearizeVectorInsert final
    : public OpConversionPattern<vector::InsertOp> {
  using Base::Base;
  LinearizeVectorInsert(const TypeConverter &typeConverter,
                        MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}
  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstTy = getTypeConverter()->convertType<VectorType>(
        insertOp.getDestVectorType());
    assert(dstTy && "vector type destination expected.");

    // Dynamic position is not supported.
    if (insertOp.hasDynamicPosition())
      return rewriter.notifyMatchFailure(insertOp,
                                         "dynamic position is not supported.");
    auto srcTy = insertOp.getValueToStoreType();
    auto srcAsVec = dyn_cast<VectorType>(srcTy);
    uint64_t srcSize = srcAsVec ? srcAsVec.getNumElements() : 1;

    auto dstShape = insertOp.getDestVectorType().getShape();
    const auto dstSize = insertOp.getDestVectorType().getNumElements();
    auto dstSizeForOffsets = dstSize;

    // Compute linearized offset.
    int64_t linearizedOffset = 0;
    auto offsetsNd = insertOp.getStaticPosition();
    for (auto [dim, offset] : llvm::enumerate(offsetsNd)) {
      dstSizeForOffsets /= dstShape[dim];
      linearizedOffset += offset * dstSizeForOffsets;
    }

    Location loc = insertOp.getLoc();
    Value valueToStore = adaptor.getValueToStore();

    if (!isa<VectorType>(valueToStore.getType())) {
      // Scalar case: generate a 1-D insert.
      Value result = rewriter.createOrFold<vector::InsertOp>(
          loc, valueToStore, adaptor.getDest(), linearizedOffset);
      rewriter.replaceOp(insertOp, result);
      return success();
    }

    // Vector case: generate a shuffle.
    llvm::SmallVector<int64_t, 2> indices(dstSize);
    auto *origValsUntil = indices.begin();
    std::advance(origValsUntil, linearizedOffset);

    // Original values that remain [0, offset).
    std::iota(indices.begin(), origValsUntil, 0);
    auto *newValsUntil = origValsUntil;
    std::advance(newValsUntil, srcSize);
    // New values [offset, offset+srcNumElements).
    std::iota(origValsUntil, newValsUntil, dstSize);
    // The rest of original values [offset+srcNumElements, end);
    std::iota(newValsUntil, indices.end(), linearizedOffset + srcSize);

    Value result = rewriter.createOrFold<vector::ShuffleOp>(
        loc, dstTy, adaptor.getDest(), valueToStore, indices);

    rewriter.replaceOp(insertOp, result);
    return success();
  }
};

/// This pattern converts the BitCastOp that works on nD (n > 1)
/// vectors to a BitCastOp that works on linearized vectors.
/// Following,
///   vector.bitcast %v1: vector<4x2xf32> to vector<4x4xf16>
/// is converted to :
///   %v1_1d = vector.shape_cast %v1: vector<4x2xf32> to vector<8xf32>
///   %out_1d = vector.bitcast %v1_1d: vector<8xf32> to vector<16xf16>
///   %out_nd = vector.shape_cast %out_1d: vector<16xf16> to vector<4x4xf16>
struct LinearizeVectorBitCast final
    : public OpConversionPattern<vector::BitCastOp> {
  using Base::Base;
  LinearizeVectorBitCast(const TypeConverter &typeConverter,
                         MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}
  LogicalResult
  matchAndRewrite(vector::BitCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = getTypeConverter()->convertType(castOp.getType());
    assert(resType && "expected 1-D vector type");
    rewriter.replaceOpWithNewOp<vector::BitCastOp>(castOp, resType,
                                                   adaptor.getSource());
    return mlir::success();
  }
};

/// This pattern converts the SplatOp to work on a linearized vector.
/// Following,
///   vector.splat %value : vector<4x4xf32>
/// is converted to:
///   %out_1d = vector.splat %value : vector<16xf32>
///   %out_nd = vector.shape_cast %out_1d : vector<16xf32> to vector<4x4xf32>
struct LinearizeVectorSplat final
    : public OpConversionPattern<vector::SplatOp> {
  using Base::Base;

  LinearizeVectorSplat(const TypeConverter &typeConverter, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::SplatOp splatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstTy = getTypeConverter()->convertType(splatOp.getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(splatOp, "cannot convert type.");
    rewriter.replaceOpWithNewOp<vector::SplatOp>(splatOp, adaptor.getInput(),
                                                 dstTy);
    return success();
  }
};

/// This pattern converts the CreateMaskOp to work on a linearized vector.
/// It currently supports only 2D masks with a unit outer dimension.
/// Following,
///   vector.create_mask %arg0, %arg1 : vector<1x4xi1>
/// is converted to:
///   %zero = arith.constant 0 : index
///   %cmpi = arith.cmpi sgt, %arg0, %zero : index
///   %index = arith.index_cast %cmpi : i1 to index
///   %mul = arith.andi %index, %arg1 : index
///   %mask = vector.create_mask %mul : vector<4xi1>
///   %shape_cast = vector.shape_cast %mask : vector<4xi1> to vector<1x4xi1>
struct LinearizeVectorCreateMask final
    : OpConversionPattern<vector::CreateMaskOp> {
  using Base::Base;

  LinearizeVectorCreateMask(const TypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::CreateMaskOp createMaskOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = createMaskOp.getLoc();
    VectorType srcTy = createMaskOp.getType();
    auto srcShape = srcTy.getShape();
    if (srcShape.size() != 2)
      return rewriter.notifyMatchFailure(createMaskOp,
                                         "only 2D mask is supported.");

    if (srcShape[0] != 1)
      return rewriter.notifyMatchFailure(
          createMaskOp, "only unit outer dimension is supported.");

    auto dstTy = getTypeConverter()->convertType(srcTy);
    if (!dstTy)
      return rewriter.notifyMatchFailure(createMaskOp, "cannot convert type.");

    // Compare the first operand with 0. If it is greater than 0, the
    // corresponding mask element is set to true, otherwise false.
    // The result of the comparison is then multiplied with
    // the second operand of create_mask to get the 1D mask.
    auto firstOperand = adaptor.getOperands().front();
    auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto isNonZero = rewriter.createOrFold<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, firstOperand, zero);
    auto isNonZeroIndex = rewriter.createOrFold<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), isNonZero);
    auto secondOperand = adaptor.getOperands().back();
    auto maskSize = rewriter.createOrFold<mlir::arith::AndIOp>(
        loc, rewriter.getIndexType(), isNonZeroIndex, secondOperand);

    auto newMask =
        mlir::vector::CreateMaskOp::create(rewriter, loc, dstTy, maskSize);
    rewriter.replaceOp(createMaskOp, newMask);
    return success();
  }
};

/// This pattern linearizes vector.load from vector<1x1x...xN> to vector<N>
/// It currently supports linearization where all but the last dimension are 1
/// The following,
///   vector.load %arg0[%c0, %c0] : memref<1x4xf32>, vector<1x4xf32>
/// is converted to:
///   vector.load %arg0[%c0, %c0] : memref<1x4xf32>, vector<4xf32>
///   vector.shape_cast %load_result : vector<4xf32> to vector<1x4xf32>
/// For generic cases, the vector unroll pass should be used to unroll the load
/// to vector<1x1x...xN> form and then linearized
struct LinearizeVectorLoad final : public OpConversionPattern<vector::LoadOp> {
  using Base::Base;
  LinearizeVectorLoad(const TypeConverter &typeConverter, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType vecTy = loadOp.getType();
    if (!vecTy)
      return rewriter.notifyMatchFailure(loadOp, "expected vector type");

    auto shape = vecTy.getShape();
    auto scalableDims = vecTy.getScalableDims();
    // All but the last dim must be 1, and only the last dim may be scalable (if
    // any).
    if (!llvm::all_of(shape.drop_back(1), [](auto d) { return d == 1; }))
      return rewriter.notifyMatchFailure(loadOp,
                                         "only vector<1x1x...xN> supported");

    if (llvm::any_of(scalableDims.drop_back(1), [](bool s) { return s; }))
      return rewriter.notifyMatchFailure(loadOp,
                                         "only innermost dim may be scalable");

    auto linearTy = typeConverter->convertType<VectorType>(vecTy);

    auto newLoad =
        vector::LoadOp::create(rewriter, loadOp.getLoc(), linearTy,
                               adaptor.getBase(), adaptor.getIndices());
    rewriter.replaceOp(loadOp, newLoad.getResult());
    return success();
  }
};

/// This pattern linearizes vector.store from vector<1x1x...xN> to vector<N>
/// It currently supports linearization where all but the last dimension are 1
/// The following,
///   vector.store %arg0, %arg1[%c0, %c0]s
///     : vector<1x4xf32>, memref<1x4xf32>
/// is converted to:
///   vector.shape_cast %arg0 : vector<1x4xf32> to vector<4xf32>
///   vector.store %arg0, %arg1[%c0, %c0]
///     : vector<4xf32>, memref<1x4xf32>
/// For generic cases, the vector unroll pass should be used to unroll the store
/// to vector<1x1x...xN> form and then linearized
struct LinearizeVectorStore final
    : public OpConversionPattern<vector::StoreOp> {
  using Base::Base;
  LinearizeVectorStore(const TypeConverter &typeConverter, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType vecTy = storeOp.getValueToStore().getType();
    if (!vecTy)
      return rewriter.notifyMatchFailure(storeOp, "expected vector type");

    auto shape = vecTy.getShape();
    auto scalableDims = vecTy.getScalableDims();
    // All but the last dim must be 1, and only the last dim may be scalable (if
    // any).
    if (!llvm::all_of(shape.drop_back(1), [](auto d) { return d == 1; }))
      return rewriter.notifyMatchFailure(storeOp,
                                         "only vector<1x1x...xN> supported");

    if (llvm::any_of(scalableDims.drop_back(1), [](bool s) { return s; }))
      return rewriter.notifyMatchFailure(storeOp,
                                         "only innermost dim may be scalable");

    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        storeOp, adaptor.getValueToStore(), adaptor.getBase(),
        adaptor.getIndices());
    return success();
  }
};

/// This pattern linearizes `vector.from_elements` operations by converting
/// the result type to a 1-D vector while preserving all element values.
/// The transformation creates a linearized `vector.from_elements` followed by
/// a `vector.shape_cast` to restore the original multidimensional shape.
///
/// Example:
///
///     %0 = vector.from_elements %a, %b, %c, %d : vector<2x2xf32>
///
/// is converted to:
///
///     %0 = vector.from_elements %a, %b, %c, %d : vector<4xf32>
///     %1 = vector.shape_cast %0 : vector<4xf32> to vector<2x2xf32>
///
struct LinearizeVectorFromElements final
    : public OpConversionPattern<vector::FromElementsOp> {
  using Base::Base;
  LinearizeVectorFromElements(const TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}
  LogicalResult
  matchAndRewrite(vector::FromElementsOp fromElementsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstTy =
        getTypeConverter()->convertType<VectorType>(fromElementsOp.getType());
    assert(dstTy && "vector type destination expected.");

    OperandRange elements = fromElementsOp.getElements();
    assert(elements.size() == static_cast<size_t>(dstTy.getNumElements()) &&
           "expected same number of elements");
    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(fromElementsOp, dstTy,
                                                        elements);
    return success();
  }
};

/// This pattern linearizes the operand in `vector.to_elements` operations
/// by converting the source type to a 1-D vector while preserving all element
/// values. The transformation creates a linearized `vector.shape_cast`
/// followed by a `vector.to_elements`.
///
/// Example:
///
///     %0:4 = vector.to_elements %v : vector<2x2xf32>
///
/// is converted to:
///
///     %vector_cast = vector.shape_cast %v : vector<2x2xf32> to vector<4xf32>
///     %0:4 = vector.to_elements %vector_cast : vector<4xf32>
///
struct LinearizeVectorToElements final
    : public OpConversionPattern<vector::ToElementsOp> {
  using Base::Base;

  LinearizeVectorToElements(const TypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::ToElementsOp toElementsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    VectorType vecType = toElementsOp.getSource().getType();
    if (vecType.getRank() <= 1)
      return rewriter.notifyMatchFailure(
          toElementsOp, "the rank is already less than or equal to 1");

    assert(vecType.getNumScalableDims() == 0 &&
           "to_elements does not support scalable vectors");
    auto vec1DType =
        VectorType::get({vecType.getNumElements()}, vecType.getElementType());
    Value shapeCast = vector::ShapeCastOp::create(
        rewriter, toElementsOp.getLoc(), vec1DType, toElementsOp.getSource());
    auto newToElementsOp =
        vector::ToElementsOp::create(rewriter, toElementsOp.getLoc(),
                                     toElementsOp.getResultTypes(), shapeCast);
    rewriter.replaceOp(toElementsOp, newToElementsOp);
    return success();
  }
};

} // namespace

/// This method defines the set of operations that are linearizable, and hence
/// that are considered illegal for the conversion target.
static bool isLinearizable(Operation *op) {

  // Only ops that are in the vector dialect, are ConstantLike, or
  // are Vectorizable might be linearized currently.
  StringLiteral vectorDialect = vector::VectorDialect::getDialectNamespace();
  StringRef opDialect = op->getDialect()->getNamespace();
  bool supported = (opDialect == vectorDialect) ||
                   op->hasTrait<OpTrait::ConstantLike>() ||
                   op->hasTrait<OpTrait::Vectorizable>();
  if (!supported)
    return false;

  return TypeSwitch<Operation *, bool>(op)
      // As type legalization is done with vector.shape_cast, shape_cast
      // itself cannot be linearized (will create new shape_casts to linearize
      // ad infinitum).
      .Case<vector::ShapeCastOp>([&](auto) { return false; })
      // The operations
      // - vector.extract_strided_slice
      // - vector.extract
      // - vector.insert_strided_slice
      // - vector.insert
      // are linearized to a rank-1 vector.shuffle by the current patterns.
      // vector.shuffle only supports fixed size vectors, so it is impossible to
      // use this approach to linearize these ops if they operate on scalable
      // vectors.
      .Case<vector::ExtractStridedSliceOp>(
          [&](vector::ExtractStridedSliceOp extractOp) {
            return !extractOp.getType().isScalable();
          })
      .Case<vector::InsertStridedSliceOp>(
          [&](vector::InsertStridedSliceOp insertOp) {
            return !insertOp.getType().isScalable();
          })
      .Case<vector::InsertOp>([&](vector::InsertOp insertOp) {
        return !insertOp.getType().isScalable();
      })
      .Case<vector::ExtractOp>([&](vector::ExtractOp extractOp) {
        return !extractOp.getSourceVectorType().isScalable();
      })
      .Default([&](auto) { return true; });
}

void mlir::vector::populateForVectorLinearize(TypeConverter &typeConverter,
                                              ConversionTarget &target) {

  auto convertType = [](Type type) -> std::optional<Type> {
    VectorType vectorType = dyn_cast<VectorType>(type);
    if (!vectorType || !isLinearizableVector(vectorType))
      return type;

    VectorType linearizedType =
        VectorType::get(vectorType.getNumElements(),
                        vectorType.getElementType(), vectorType.isScalable());
    return linearizedType;
  };
  typeConverter.addConversion(convertType);

  auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                            Location loc) -> Value {
    if (inputs.size() != 1)
      return nullptr;

    Value value = inputs.front();
    if (!isa<VectorType>(type) || !isa<VectorType>(value.getType()))
      return nullptr;

    return vector::ShapeCastOp::create(builder, loc, type, value);
  };
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  target.markUnknownOpDynamicallyLegal(
      [=](Operation *op) -> std::optional<bool> {
        if (!isLinearizable(op))
          return true;
        // This will return true if, for all operand and result types `t`,
        // convertType(t) = t. This is true if there are no rank>=2 vectors.
        return typeConverter.isLegal(op);
      });
}

void mlir::vector::populateVectorLinearizeBasePatterns(
    const TypeConverter &typeConverter, const ConversionTarget &target,
    RewritePatternSet &patterns) {
  patterns
      .add<LinearizeConstantLike, LinearizeVectorizable, LinearizeVectorBitCast,
           LinearizeVectorSplat, LinearizeVectorCreateMask, LinearizeVectorLoad,
           LinearizeVectorStore, LinearizeVectorFromElements,
           LinearizeVectorToElements>(typeConverter, patterns.getContext());
}

void mlir::vector::populateVectorLinearizeShuffleLikeOpsPatterns(
    const TypeConverter &typeConverter, const ConversionTarget &target,
    RewritePatternSet &patterns) {
  patterns.add<LinearizeVectorShuffle, LinearizeVectorExtract,
               LinearizeVectorInsert, LinearizeVectorExtractStridedSlice,
               LinearizeVectorInsertStridedSlice>(typeConverter,
                                                  patterns.getContext());
}
