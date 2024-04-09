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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
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
    auto dstType = getTypeConverter()->convertType(extractOp.getType());
    auto loc = extractOp.getLoc();
    if (!dstType)
      return rewriter.notifyMatchFailure(loc, "cannot convert type.");
    if (extractOp.getVector().getType().isScalable() ||
        dstType.cast<VectorType>().isScalable())
      return rewriter.notifyMatchFailure(loc,
                                         "scalable vectors are not supported.");
    if (!isLessThanTargetBitWidth(extractOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          extractOp, "Can't flatten since targetBitWidth <= OpSize");

    auto offsets = extractOp.getOffsets().getValue();
    auto sizes = extractOp.getSizes().getValue();
    auto strides = extractOp.getStrides().getValue();

    if (!isConstantIntValue(strides[0], 1))
      return rewriter.notifyMatchFailure(
          extractOp, "Strided slice with stride != 1 is not supported.");

    Value srcVector = adaptor.getVector();

    // if kD offsets are specified for nd source vector (n > k), the granularity
    // of the extraction is greater than 1. In this case last (n-k) dimensions
    // form the extraction granularity. example : %0 =
    // vector.extract_strided_slice %src { offsets = [0, 0], sizes = [2, 2],
    // strides = [1, 1]} : vector<4x8x8xf32> to vector<2x2x8xf32>
    // here, extraction granularity is 8.
    int64_t extractSliceLen = 1;
    auto n = extractOp.getSourceVectorType().getRank();
    auto k = (int64_t)offsets.size();
    if (n > k) {
      for (unsigned i = 0; i < n - k; i++) {
        extractSliceLen *= extractOp.getSourceVectorType().getShape()[i + k];
      }
    }

    // get total number of extracted slices
    int64_t nExtractedSlices = 1;
    for (auto size : sizes) {
      nExtractedSlices *= size.cast<IntegerAttr>().getInt();
    }

    // compute the strides of the source vector considering first k dimensions
    llvm::SmallVector<int64_t, 4> sourceStrides(k, extractSliceLen);
    for (int i = k - 2; i >= 0; --i) {
      sourceStrides[i] = sourceStrides[i + 1] *
                         extractOp.getSourceVectorType().getShape()[i + 1];
    }
    // final shuffle indices has nExtractedElems * extractSliceLen elements
    llvm::SmallVector<int64_t, 4> indices(nExtractedSlices * extractSliceLen);
    // compute the strides of the extracted kD vector
    llvm::SmallVector<int64_t, 4> extractedStrides(k, 1);
    // compute extractedStrides
    for (int i = k - 2; i >= 0; --i) {
      extractedStrides[i] =
          extractedStrides[i + 1] * sizes[i + 1].cast<IntegerAttr>().getInt();
    }
    // iterate over all extracted slices from 0 to nExtractedElems-1
    // and compute the multi-dimensional index and the corresponding linearized
    // index within the source vector
    for (int64_t i = 0; i < nExtractedSlices; ++i) {
      int64_t index = i;
      // compute the corresponding multi-dimensional index
      llvm::SmallVector<int64_t, 4> multiDimIndex(k, 0);
      for (int64_t j = 0; j < k; ++j) {
        multiDimIndex[j] = (index / extractedStrides[j]);
        index -= multiDimIndex[j] * extractedStrides[j];
      }
      // compute the corresponding linearized index in the source vector
      // i.e. shift the multiDimIndex by the offsets
      int64_t linearizedIndex = 0;
      for (int64_t j = 0; j < k; ++j) {
        linearizedIndex +=
            (offsets[j].cast<IntegerAttr>().getInt() + multiDimIndex[j]) *
            sourceStrides[j];
      }
      // fill the indices array form linearizedIndex to linearizedIndex +
      // sliceLen
      for (int64_t j = 0; j < extractSliceLen; ++j) {
        indices[i * extractSliceLen + j] = linearizedIndex + j;
      }
    }
    // perform a shuffle to extract the kD vector
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        extractOp, dstType, srcVector, srcVector,
        rewriter.getI64ArrayAttr(indices));

    return success();
  }

private:
  unsigned targetVectorBitWidth;
};

struct LinearizeVectorShffle final
    : public OpConversionPattern<vector::ShuffleOp> {
  using OpConversionPattern::OpConversionPattern;
  LinearizeVectorShffle(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(shuffleOp.getType());
    auto loc = shuffleOp.getLoc();
    if (!dstType)
      return rewriter.notifyMatchFailure(loc, "cannot convert type.");

    if (shuffleOp.getV1VectorType().isScalable() ||
        shuffleOp.getV2VectorType().isScalable() ||
        dstType.cast<VectorType>().isScalable())
      return rewriter.notifyMatchFailure(loc,
                                         "scalable vectors are not supported.");
    if (!isLessThanTargetBitWidth(shuffleOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          shuffleOp, "Can't flatten since targetBitWidth <= OpSize");

    auto vec1 = adaptor.getV1();
    auto vec2 = adaptor.getV2();

    int shuffleSliceLen = 1;
    int rank = shuffleOp.getV1().getType().getRank();

    // if rank > 1, we need to do the shuffle in the granularity of slices
    // instead of scalars. Size of the slice is equal to the rank-1 innermost
    // dims. Mask of the shuffle op specifies which slice to take from the
    // outermost dim.
    if (rank > 1) {
      auto shape = shuffleOp.getV1().getType().getShape();
      for (unsigned i = 1; i < shape.size(); i++) {
        shuffleSliceLen *= shape[i];
      }
    }

    auto mask = shuffleOp.getMask();
    auto totalSize = mask.size() * shuffleSliceLen;

    llvm::SmallVector<int64_t, 2> indices(totalSize);
    for (auto [i, value] :
         llvm::enumerate(mask.getAsValueRange<IntegerAttr>())) {

      int64_t v = value.getZExtValue();
      std::iota(indices.begin() + shuffleSliceLen * i,
                indices.begin() + shuffleSliceLen * (i + 1),
                shuffleSliceLen * v);
    }

    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        shuffleOp, dstType, vec1, vec2, rewriter.getI64ArrayAttr(indices));

    return success();
  }

private:
  unsigned targetVectorBitWidth;
};

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
    auto dstTy = getTypeConverter()->convertType(extractOp.getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(extractOp, "cannot convert type.");

    if (extractOp.getVector().getType().isScalable() ||
        dstTy.cast<VectorType>().isScalable())
      return rewriter.notifyMatchFailure(extractOp,
                                         "scalable vectors are not supported.");
    if (!isLessThanTargetBitWidth(extractOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          extractOp, "Can't flatten since targetBitWidth <= OpSize");

    // dynamic position is not supported
    if (extractOp.hasDynamicPosition())
      return rewriter.notifyMatchFailure(extractOp,
                                         "dynamic position is not supported.");

    auto shape = extractOp.getVector().getType().getShape();
    auto size = extractOp.getVector().getType().getNumElements();

    // compute linearized offset
    int64_t linearizedOffset = 0;
    auto offsets = extractOp.getStaticPosition();
    for (auto [i, off] : llvm::enumerate(offsets)) {
      size /= shape[i];
      linearizedOffset += offsets[i] * size;
    }

    llvm::SmallVector<int64_t, 2> indices(size);
    std::iota(indices.begin(), indices.end(), linearizedOffset);
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        extractOp, dstTy, adaptor.getVector(), adaptor.getVector(),
        rewriter.getI64ArrayAttr(indices));

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
        if (isa<vector::ShuffleOp>(op)) {
          return (isLessThanTargetBitWidth(op, targetBitWidth)
                      ? (typeConverter.isLegal(op) &&
                         op->getResult(0)
                                 .getType()
                                 .cast<mlir::VectorType>()
                                 .getRank() == 1)
                      : true);
        }
        return std::nullopt;
      });

  // target.addDynamicallyLegalOp<mlir::vector::ShuffleOp>(
  //     [=](mlir::Operation *op) {
  //       return op->getResult(0).getType().cast<mlir::VectorType>().getRank()
  //       ==
  //              1;
  //     });

  patterns.add<LinearizeConstant, LinearizeVectorizable, LinearizeVectorShffle,
               LinearizeVectorExtract, LinearizeVectorExtractStridedSlice>(
      typeConverter, patterns.getContext(), targetBitWidth);
}
