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

#include "mlir/Dialect/Arith/IR/Arith.h"
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

using namespace mlir;

static bool isLessThanTargetBitWidth(Operation *op, unsigned targetBitWidth) {
  auto resultTypes = op->getResultTypes();
  for (auto resType : resultTypes) {
    VectorType vecType = dyn_cast<VectorType>(resType);
    // Reject index since getElementTypeBitWidth will abort for Index types.
    if (!vecType || vecType.getElementType().isIndex())
      return false;
    // There are no dimension to fold if it is a 0-D vector.
    if (vecType.getRank() == 0)
      return false;
    unsigned trailingVecDimBitWidth =
        vecType.getShape().back() * vecType.getElementTypeBitWidth();
    if (trailingVecDimBitWidth >= targetBitWidth)
      return false;
  }
  return true;
}

static bool isLessThanOrEqualTargetBitWidth(Type t, unsigned targetBitWidth) {
  VectorType vecType = dyn_cast<VectorType>(t);
  // Reject index since getElementTypeBitWidth will abort for Index types.
  if (!vecType || vecType.getElementType().isIndex())
    return false;
  // There are no dimension to fold if it is a 0-D vector.
  if (vecType.getRank() == 0)
    return false;
  unsigned trailingVecDimBitWidth =
      vecType.getShape().back() * vecType.getElementTypeBitWidth();
  return trailingVecDimBitWidth <= targetBitWidth;
}

namespace {
struct LinearizeConstant final : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LinearizeConstant(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}
  LogicalResult
  matchAndRewrite(arith::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = constOp.getLoc();
    auto resType =
        getTypeConverter()->convertType<VectorType>(constOp.getType());

    if (resType.isScalable() && !isa<SplatElementsAttr>(constOp.getValue()))
      return rewriter.notifyMatchFailure(
          loc,
          "Cannot linearize a constant scalable vector that's not a splat");

    if (!resType)
      return rewriter.notifyMatchFailure(loc, "can't convert return type");
    if (!isLessThanTargetBitWidth(constOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          loc, "Can't flatten since targetBitWidth <= OpSize");
    auto dstElementsAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!dstElementsAttr)
      return rewriter.notifyMatchFailure(loc, "unsupported attr type");

    dstElementsAttr = dstElementsAttr.reshape(resType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, resType,
                                                   dstElementsAttr);
    return success();
  }

private:
  unsigned targetVectorBitWidth;
};

struct LinearizeVectorizable final
    : OpTraitConversionPattern<OpTrait::Vectorizable> {
  using OpTraitConversionPattern::OpTraitConversionPattern;

public:
  LinearizeVectorizable(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpTraitConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isLessThanTargetBitWidth(op, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Can't flatten since targetBitWidth <= OpSize");
    FailureOr<Operation *> newOp =
        convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return success();
  }

private:
  unsigned targetVectorBitWidth;
};

/// This pattern converts the ExtractStridedSliceOp into a ShuffleOp that works
/// on a linearized vector.
/// Following,
///   vector.extract_strided_slice %source
///         { offsets = [..], strides = [..], sizes = [..] }
/// is converted to :
///   %source_1d = vector.shape_cast %source
///   %out_1d = vector.shuffle %source_1d, %source_1d [ shuffle_indices_1d ]
///   %out_nd = vector.shape_cast %out_1d
/// `shuffle_indices_1d` is computed using the offsets and sizes of the
/// extraction.
struct LinearizeVectorExtractStridedSlice final
    : public mlir::OpConversionPattern<mlir::vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LinearizeVectorExtractStridedSlice(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstType =
        getTypeConverter()->convertType<VectorType>(extractOp.getType());
    assert(dstType && "vector type destination expected.");
    if (extractOp.getVector().getType().isScalable() || dstType.isScalable())
      return rewriter.notifyMatchFailure(extractOp,
                                         "scalable vectors are not supported.");
    if (!isLessThanTargetBitWidth(extractOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          extractOp, "Can't flatten since targetBitWidth <= OpSize");

    ArrayAttr offsets = extractOp.getOffsets();
    ArrayAttr sizes = extractOp.getSizes();
    ArrayAttr strides = extractOp.getStrides();
    if (!isConstantIntValue(strides[0], 1))
      return rewriter.notifyMatchFailure(
          extractOp, "Strided slice with stride != 1 is not supported.");
    Value srcVector = adaptor.getVector();
    // If kD offsets are specified for nD source vector (n > k), the granularity
    // of the extraction is greater than 1. In this case last (n-k) dimensions
    // form the extraction granularity.
    // Example :
    //  vector.extract_strided_slice %src {
    //      offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} :
    //      vector<4x8x8xf32> to vector<2x2x8xf32>
    // Here, extraction granularity is 8.
    int64_t extractGranularitySize = 1;
    int64_t nD = extractOp.getSourceVectorType().getRank();
    int64_t kD = (int64_t)offsets.size();
    int64_t k = kD;
    while (k < nD) {
      extractGranularitySize *= extractOp.getSourceVectorType().getShape()[k];
      ++k;
    }
    // Get total number of extracted slices.
    int64_t nExtractedSlices = 1;
    for (Attribute size : sizes) {
      nExtractedSlices *= cast<IntegerAttr>(size).getInt();
    }
    // Compute the strides of the source vector considering first k dimensions.
    llvm::SmallVector<int64_t, 4> sourceStrides(kD, extractGranularitySize);
    for (int i = kD - 2; i >= 0; --i) {
      sourceStrides[i] = sourceStrides[i + 1] *
                         extractOp.getSourceVectorType().getShape()[i + 1];
    }
    // Final shuffle indices has nExtractedSlices * extractGranularitySize
    // elements.
    llvm::SmallVector<int64_t, 4> indices(nExtractedSlices *
                                          extractGranularitySize);
    // Compute the strides of the extracted kD vector.
    llvm::SmallVector<int64_t, 4> extractedStrides(kD, 1);
    // Compute extractedStrides.
    for (int i = kD - 2; i >= 0; --i) {
      extractedStrides[i] =
          extractedStrides[i + 1] * cast<IntegerAttr>(sizes[i + 1]).getInt();
    }
    // Iterate over all extracted slices from 0 to nExtractedSlices - 1
    // and compute the multi-dimensional index and the corresponding linearized
    // index within the source vector.
    for (int64_t i = 0; i < nExtractedSlices; ++i) {
      int64_t index = i;
      // Compute the corresponding multi-dimensional index.
      llvm::SmallVector<int64_t, 4> multiDimIndex(kD, 0);
      for (int64_t j = 0; j < kD; ++j) {
        multiDimIndex[j] = (index / extractedStrides[j]);
        index -= multiDimIndex[j] * extractedStrides[j];
      }
      // Compute the corresponding linearized index in the source vector
      // i.e. shift the multiDimIndex by the offsets.
      int64_t linearizedIndex = 0;
      for (int64_t j = 0; j < kD; ++j) {
        linearizedIndex +=
            (cast<IntegerAttr>(offsets[j]).getInt() + multiDimIndex[j]) *
            sourceStrides[j];
      }
      // Fill the indices array form linearizedIndex to linearizedIndex +
      // extractGranularitySize.
      for (int64_t j = 0; j < extractGranularitySize; ++j) {
        indices[i * extractGranularitySize + j] = linearizedIndex + j;
      }
    }
    // Perform a shuffle to extract the kD vector.
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        extractOp, dstType, srcVector, srcVector, indices);
    return success();
  }

private:
  unsigned targetVectorBitWidth;
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
  using OpConversionPattern::OpConversionPattern;
  LinearizeVectorShuffle(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstType =
        getTypeConverter()->convertType<VectorType>(shuffleOp.getType());
    assert(dstType && "vector type destination expected.");
    // The assert is used because vector.shuffle does not support scalable
    // vectors.
    assert(!(shuffleOp.getV1VectorType().isScalable() ||
             shuffleOp.getV2VectorType().isScalable() ||
             dstType.isScalable()) &&
           "scalable vectors are not supported.");
    if (!isLessThanTargetBitWidth(shuffleOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          shuffleOp, "Can't flatten since targetBitWidth <= OpSize");

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
    // that needs to be shuffled to the destination vector. If shuffleSliceLen >
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

private:
  unsigned targetVectorBitWidth;
};

/// This pattern converts the ExtractOp to a ShuffleOp that works on a
/// linearized vector.
/// Following,
///   vector.extract %source [ position ]
/// is converted to :
///   %source_1d = vector.shape_cast %source
///   %out_1d = vector.shuffle %source_1d, %source_1d [ shuffle_indices_1d ]
///   %out_nd = vector.shape_cast %out_1d
/// `shuffle_indices_1d` is computed using the position of the original extract.
struct LinearizeVectorExtract final
    : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LinearizeVectorExtract(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}
  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(extractOp.getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(extractOp,
                                         "expected n-D vector type.");

    if (extractOp.getVector().getType().isScalable() ||
        cast<VectorType>(dstTy).isScalable())
      return rewriter.notifyMatchFailure(extractOp,
                                         "scalable vectors are not supported.");
    if (!isLessThanTargetBitWidth(extractOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          extractOp, "Can't flatten since targetBitWidth <= OpSize");

    // Dynamic position is not supported.
    if (extractOp.hasDynamicPosition())
      return rewriter.notifyMatchFailure(extractOp,
                                         "dynamic position is not supported.");

    llvm::ArrayRef<int64_t> shape = extractOp.getVector().getType().getShape();
    int64_t size = extractOp.getVector().getType().getNumElements();

    // Compute linearized offset.
    int64_t linearizedOffset = 0;
    llvm::ArrayRef<int64_t> offsets = extractOp.getStaticPosition();
    for (auto [i, off] : llvm::enumerate(offsets)) {
      size /= shape[i];
      linearizedOffset += offsets[i] * size;
    }

    llvm::SmallVector<int64_t, 2> indices(size);
    std::iota(indices.begin(), indices.end(), linearizedOffset);
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        extractOp, dstTy, adaptor.getVector(), adaptor.getVector(), indices);

    return success();
  }

private:
  unsigned targetVectorBitWidth;
};

/// This pattern converts the InsertOp to a ShuffleOp that works on a
/// linearized vector.
/// Following,
///   vector.insert %source %destination [ position ]
/// is converted to :
///   %source_1d = vector.shape_cast %source
///   %destination_1d = vector.shape_cast %destination
///   %out_1d = vector.shuffle %destination_1d, %source_1d [ shuffle_indices_1d
///   ] %out_nd = vector.shape_cast %out_1d
/// `shuffle_indices_1d` is computed using the position of the original insert.
struct LinearizeVectorInsert final
    : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;
  LinearizeVectorInsert(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}
  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstTy = getTypeConverter()->convertType<VectorType>(
        insertOp.getDestVectorType());
    assert(dstTy && "vector type destination expected.");
    if (insertOp.getDestVectorType().isScalable() || dstTy.isScalable())
      return rewriter.notifyMatchFailure(insertOp,
                                         "scalable vectors are not supported.");

    if (!isLessThanOrEqualTargetBitWidth(insertOp.getSourceType(),
                                         targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          insertOp, "Can't flatten since targetBitWidth < OpSize");

    // dynamic position is not supported
    if (insertOp.hasDynamicPosition())
      return rewriter.notifyMatchFailure(insertOp,
                                         "dynamic position is not supported.");
    auto srcTy = insertOp.getSourceType();
    auto srcAsVec = dyn_cast<VectorType>(srcTy);
    uint64_t srcSize = 0;
    if (srcAsVec) {
      srcSize = srcAsVec.getNumElements();
    } else {
      return rewriter.notifyMatchFailure(insertOp,
                                         "scalars are not supported.");
    }

    auto dstShape = insertOp.getDestVectorType().getShape();
    const auto dstSize = insertOp.getDestVectorType().getNumElements();
    auto dstSizeForOffsets = dstSize;

    // compute linearized offset
    int64_t linearizedOffset = 0;
    auto offsetsNd = insertOp.getStaticPosition();
    for (auto [dim, offset] : llvm::enumerate(offsetsNd)) {
      dstSizeForOffsets /= dstShape[dim];
      linearizedOffset += offset * dstSizeForOffsets;
    }

    llvm::SmallVector<int64_t, 2> indices(dstSize);
    auto origValsUntil = indices.begin();
    std::advance(origValsUntil, linearizedOffset);
    std::iota(indices.begin(), origValsUntil,
              0); // original values that remain [0, offset)
    auto newValsUntil = origValsUntil;
    std::advance(newValsUntil, srcSize);
    std::iota(origValsUntil, newValsUntil,
              dstSize); // new values [offset, offset+srcNumElements)
    std::iota(newValsUntil, indices.end(),
              linearizedOffset + srcSize); // the rest of original values
                                           // [offset+srcNumElements, end)

    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        insertOp, dstTy, adaptor.getDest(), adaptor.getSource(), indices);

    return success();
  }

private:
  unsigned targetVectorBitWidth;
};
} // namespace

void mlir::vector::populateVectorLinearizeTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, unsigned targetBitWidth) {

  typeConverter.addConversion([](VectorType type) -> std::optional<Type> {
    if (!isLinearizableVector(type))
      return type;

    return VectorType::get(type.getNumElements(), type.getElementType(),
                           type.isScalable());
  });

  auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                            Location loc) -> Value {
    if (inputs.size() != 1 || !isa<VectorType>(inputs.front().getType()) ||
        !isa<VectorType>(type))
      return nullptr;

    return builder.create<vector::ShapeCastOp>(loc, type, inputs.front());
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);
  target.markUnknownOpDynamicallyLegal(
      [=](Operation *op) -> std::optional<bool> {
        if ((isa<arith::ConstantOp>(op) ||
             op->hasTrait<OpTrait::Vectorizable>())) {
          return (isLessThanTargetBitWidth(op, targetBitWidth)
                      ? typeConverter.isLegal(op)
                      : true);
        }
        return std::nullopt;
      });

  patterns.add<LinearizeConstant, LinearizeVectorizable>(
      typeConverter, patterns.getContext(), targetBitWidth);
}

void mlir::vector::populateVectorLinearizeShuffleLikeOpsPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, unsigned int targetBitWidth) {
  target.addDynamicallyLegalOp<vector::ShuffleOp>(
      [=](vector::ShuffleOp shuffleOp) -> bool {
        return isLessThanTargetBitWidth(shuffleOp, targetBitWidth)
                   ? (typeConverter.isLegal(shuffleOp) &&
                      cast<mlir::VectorType>(shuffleOp.getResult().getType())
                              .getRank() == 1)
                   : true;
      });
  patterns.add<LinearizeVectorShuffle, LinearizeVectorExtract,
               LinearizeVectorInsert, LinearizeVectorExtractStridedSlice>(
      typeConverter, patterns.getContext(), targetBitWidth);
}
