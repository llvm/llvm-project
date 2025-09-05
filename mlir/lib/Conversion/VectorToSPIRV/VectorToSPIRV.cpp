//===- VectorToSPIRV.cpp - Vector to SPIR-V Patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Vector dialect to SPIRV dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <cstdint>
#include <numeric>

using namespace mlir;

/// Returns the integer value from the first valid input element, assuming Value
/// inputs are defined by a constant index ops and Attribute inputs are integer
/// attributes.
static uint64_t getFirstIntValue(ArrayAttr attr) {
  return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
}

/// Returns the number of bits for the given scalar/vector type.
static int getNumBits(Type type) {
  // TODO: This does not take into account any memory layout or widening
  // constraints. E.g., a vector<3xi57> may report to occupy 3x57=171 bit, even
  // though in practice it will likely be stored as in a 4xi64 vector register.
  if (auto vectorType = dyn_cast<VectorType>(type))
    return vectorType.getNumElements() * vectorType.getElementTypeBitWidth();
  return type.getIntOrFloatBitWidth();
}

namespace {

struct VectorShapeCast final : public OpConversionPattern<vector::ShapeCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShapeCastOp shapeCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = getTypeConverter()->convertType(shapeCastOp.getType());
    if (!dstType)
      return failure();

    // If dstType is same as the source type or the vector size is 1, it can be
    // directly replaced by the source.
    if (dstType == adaptor.getSource().getType() ||
        shapeCastOp.getResultVectorType().getNumElements() == 1) {
      rewriter.replaceOp(shapeCastOp, adaptor.getSource());
      return success();
    }

    // Lowering for size-n vectors when n > 1 hasn't been implemented.
    return failure();
  }
};

// Convert `vector.splat` to `vector.broadcast`. There is a path from
// `vector.broadcast` to SPIRV via other patterns.
struct VectorSplatToBroadcast final
    : public OpConversionPattern<vector::SplatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::SplatOp splat, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(splat, splat.getType(),
                                                     adaptor.getInput());
    return success();
  }
};

struct VectorBitcastConvert final
    : public OpConversionPattern<vector::BitCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BitCastOp bitcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = getTypeConverter()->convertType(bitcastOp.getType());
    if (!dstType)
      return failure();

    if (dstType == adaptor.getSource().getType()) {
      rewriter.replaceOp(bitcastOp, adaptor.getSource());
      return success();
    }

    // Check that the source and destination type have the same bitwidth.
    // Depending on the target environment, we may need to emulate certain
    // types, which can cause issue with bitcast.
    Type srcType = adaptor.getSource().getType();
    if (getNumBits(dstType) != getNumBits(srcType)) {
      return rewriter.notifyMatchFailure(
          bitcastOp,
          llvm::formatv("different source ({0}) and target ({1}) bitwidth",
                        srcType, dstType));
    }

    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(bitcastOp, dstType,
                                                  adaptor.getSource());
    return success();
  }
};

struct VectorBroadcastConvert final
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(castOp.getResultVectorType());
    if (!resultType)
      return failure();

    if (isa<spirv::ScalarType>(resultType)) {
      rewriter.replaceOp(castOp, adaptor.getSource());
      return success();
    }

    SmallVector<Value, 4> source(castOp.getResultVectorType().getNumElements(),
                                 adaptor.getSource());
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(castOp, resultType,
                                                             source);
    return success();
  }
};

// SPIR-V does not have a concept of a poison index for certain instructions,
// which creates a UB hazard when lowering from otherwise equivalent Vector
// dialect instructions, because this index will be considered out-of-bounds.
// To avoid this, this function implements a dynamic sanitization that returns
// some arbitrary safe index. For power-of-two vector sizes, this uses a bitmask
// (presumably more efficient), and otherwise index 0 (always in-bounds).
static Value sanitizeDynamicIndex(ConversionPatternRewriter &rewriter,
                                  Location loc, Value dynamicIndex,
                                  int64_t kPoisonIndex, unsigned vectorSize) {
  if (llvm::isPowerOf2_32(vectorSize)) {
    Value inBoundsMask = spirv::ConstantOp::create(
        rewriter, loc, dynamicIndex.getType(),
        rewriter.getIntegerAttr(dynamicIndex.getType(), vectorSize - 1));
    return spirv::BitwiseAndOp::create(rewriter, loc, dynamicIndex,
                                       inBoundsMask);
  }
  Value poisonIndex = spirv::ConstantOp::create(
      rewriter, loc, dynamicIndex.getType(),
      rewriter.getIntegerAttr(dynamicIndex.getType(), kPoisonIndex));
  Value cmpResult =
      spirv::IEqualOp::create(rewriter, loc, dynamicIndex, poisonIndex);
  return spirv::SelectOp::create(
      rewriter, loc, cmpResult,
      spirv::ConstantOp::getZero(dynamicIndex.getType(), loc, rewriter),
      dynamicIndex);
}

struct VectorExtractOpConvert final
    : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();

    if (isa<spirv::ScalarType>(adaptor.getVector().getType())) {
      rewriter.replaceOp(extractOp, adaptor.getVector());
      return success();
    }

    if (std::optional<int64_t> id =
            getConstantIntValue(extractOp.getMixedPosition()[0])) {
      if (id == vector::ExtractOp::kPoisonIndex)
        return rewriter.notifyMatchFailure(
            extractOp,
            "Static use of poison index handled elsewhere (folded to poison)");
      rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
          extractOp, dstType, adaptor.getVector(),
          rewriter.getI32ArrayAttr(id.value()));
    } else {
      Value sanitizedIndex = sanitizeDynamicIndex(
          rewriter, extractOp.getLoc(), adaptor.getDynamicPosition()[0],
          vector::ExtractOp::kPoisonIndex,
          extractOp.getSourceVectorType().getNumElements());
      rewriter.replaceOpWithNewOp<spirv::VectorExtractDynamicOp>(
          extractOp, dstType, adaptor.getVector(), sanitizedIndex);
    }
    return success();
  }
};

struct VectorExtractStridedSliceOpConvert final
    : public OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();

    uint64_t offset = getFirstIntValue(extractOp.getOffsets());
    uint64_t size = getFirstIntValue(extractOp.getSizes());
    uint64_t stride = getFirstIntValue(extractOp.getStrides());
    if (stride != 1)
      return failure();

    Value srcVector = adaptor.getOperands().front();

    // Extract vector<1xT> case.
    if (isa<spirv::ScalarType>(dstType)) {
      rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(extractOp,
                                                             srcVector, offset);
      return success();
    }

    SmallVector<int32_t, 2> indices(size);
    std::iota(indices.begin(), indices.end(), offset);

    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
        extractOp, dstType, srcVector, srcVector,
        rewriter.getI32ArrayAttr(indices));

    return success();
  }
};

template <class SPIRVFMAOp>
struct VectorFmaOpConvert final : public OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = getTypeConverter()->convertType(fmaOp.getType());
    if (!dstType)
      return failure();
    rewriter.replaceOpWithNewOp<SPIRVFMAOp>(fmaOp, dstType, adaptor.getLhs(),
                                            adaptor.getRhs(), adaptor.getAcc());
    return success();
  }
};

struct VectorFromElementsOpConvert final
    : public OpConversionPattern<vector::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();
    ValueRange elements = adaptor.getElements();
    if (isa<spirv::ScalarType>(resultType)) {
      // In the case with a single scalar operand / single-element result,
      // pass through the scalar.
      rewriter.replaceOp(op, elements[0]);
      return success();
    }
    // SPIRVTypeConverter rejects vectors with rank > 1, so multi-dimensional
    // vector.from_elements cases should not need to be handled, only 1d.
    assert(cast<VectorType>(resultType).getRank() == 1);
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, resultType,
                                                             elements);
    return success();
  }
};

struct VectorInsertOpConvert final
    : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<VectorType>(insertOp.getValueToStoreType()))
      return rewriter.notifyMatchFailure(insertOp, "unsupported vector source");
    if (!getTypeConverter()->convertType(insertOp.getDestVectorType()))
      return rewriter.notifyMatchFailure(insertOp,
                                         "unsupported dest vector type");

    // Special case for inserting scalar values into size-1 vectors.
    if (insertOp.getValueToStoreType().isIntOrFloat() &&
        insertOp.getDestVectorType().getNumElements() == 1) {
      rewriter.replaceOp(insertOp, adaptor.getValueToStore());
      return success();
    }

    if (std::optional<int64_t> id =
            getConstantIntValue(insertOp.getMixedPosition()[0])) {
      if (id == vector::InsertOp::kPoisonIndex)
        return rewriter.notifyMatchFailure(
            insertOp,
            "Static use of poison index handled elsewhere (folded to poison)");
      rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
          insertOp, adaptor.getValueToStore(), adaptor.getDest(), id.value());
    } else {
      Value sanitizedIndex = sanitizeDynamicIndex(
          rewriter, insertOp.getLoc(), adaptor.getDynamicPosition()[0],
          vector::InsertOp::kPoisonIndex,
          insertOp.getDestVectorType().getNumElements());
      rewriter.replaceOpWithNewOp<spirv::VectorInsertDynamicOp>(
          insertOp, insertOp.getDest(), adaptor.getValueToStore(),
          sanitizedIndex);
    }
    return success();
  }
};

struct VectorInsertStridedSliceOpConvert final
    : public OpConversionPattern<vector::InsertStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertStridedSliceOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcVector = adaptor.getOperands().front();
    Value dstVector = adaptor.getOperands().back();

    uint64_t stride = getFirstIntValue(insertOp.getStrides());
    if (stride != 1)
      return failure();
    uint64_t offset = getFirstIntValue(insertOp.getOffsets());

    if (isa<spirv::ScalarType>(srcVector.getType())) {
      assert(!isa<spirv::ScalarType>(dstVector.getType()));
      rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
          insertOp, dstVector.getType(), srcVector, dstVector,
          rewriter.getI32ArrayAttr(offset));
      return success();
    }

    uint64_t totalSize = cast<VectorType>(dstVector.getType()).getNumElements();
    uint64_t insertSize =
        cast<VectorType>(srcVector.getType()).getNumElements();

    SmallVector<int32_t, 2> indices(totalSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::iota(indices.begin() + offset, indices.begin() + offset + insertSize,
              totalSize);

    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
        insertOp, dstVector.getType(), dstVector, srcVector,
        rewriter.getI32ArrayAttr(indices));

    return success();
  }
};

static SmallVector<Value> extractAllElements(
    vector::ReductionOp reduceOp, vector::ReductionOp::Adaptor adaptor,
    VectorType srcVectorType, ConversionPatternRewriter &rewriter) {
  int numElements = static_cast<int>(srcVectorType.getDimSize(0));
  SmallVector<Value> values;
  values.reserve(numElements + (adaptor.getAcc() ? 1 : 0));
  Location loc = reduceOp.getLoc();

  for (int i = 0; i < numElements; ++i) {
    values.push_back(spirv::CompositeExtractOp::create(
        rewriter, loc, srcVectorType.getElementType(), adaptor.getVector(),
        rewriter.getI32ArrayAttr({i})));
  }
  if (Value acc = adaptor.getAcc())
    values.push_back(acc);

  return values;
}

struct ReductionRewriteInfo {
  Type resultType;
  SmallVector<Value> extractedElements;
};

FailureOr<ReductionRewriteInfo> static getReductionInfo(
    vector::ReductionOp op, vector::ReductionOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, const TypeConverter &typeConverter) {
  Type resultType = typeConverter.convertType(op.getType());
  if (!resultType)
    return failure();

  auto srcVectorType = dyn_cast<VectorType>(adaptor.getVector().getType());
  if (!srcVectorType || srcVectorType.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "not a 1-D vector source");

  SmallVector<Value> extractedElements =
      extractAllElements(op, adaptor, srcVectorType, rewriter);

  return ReductionRewriteInfo{resultType, std::move(extractedElements)};
}

template <typename SPIRVUMaxOp, typename SPIRVUMinOp, typename SPIRVSMaxOp,
          typename SPIRVSMinOp>
struct VectorReductionPattern final : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp reduceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reductionInfo =
        getReductionInfo(reduceOp, adaptor, rewriter, *getTypeConverter());
    if (failed(reductionInfo))
      return failure();

    auto [resultType, extractedElements] = *reductionInfo;
    Location loc = reduceOp->getLoc();
    Value result = extractedElements.front();
    for (Value next : llvm::drop_begin(extractedElements)) {
      switch (reduceOp.getKind()) {

#define INT_AND_FLOAT_CASE(kind, iop, fop)                                     \
  case vector::CombiningKind::kind:                                            \
    if (llvm::isa<IntegerType>(resultType)) {                                  \
      result = spirv::iop::create(rewriter, loc, resultType, result, next);    \
    } else {                                                                   \
      assert(llvm::isa<FloatType>(resultType));                                \
      result = spirv::fop::create(rewriter, loc, resultType, result, next);    \
    }                                                                          \
    break

#define INT_OR_FLOAT_CASE(kind, fop)                                           \
  case vector::CombiningKind::kind:                                            \
    result = fop::create(rewriter, loc, resultType, result, next);             \
    break

        INT_AND_FLOAT_CASE(ADD, IAddOp, FAddOp);
        INT_AND_FLOAT_CASE(MUL, IMulOp, FMulOp);
        INT_OR_FLOAT_CASE(MINUI, SPIRVUMinOp);
        INT_OR_FLOAT_CASE(MINSI, SPIRVSMinOp);
        INT_OR_FLOAT_CASE(MAXUI, SPIRVUMaxOp);
        INT_OR_FLOAT_CASE(MAXSI, SPIRVSMaxOp);

      case vector::CombiningKind::AND:
      case vector::CombiningKind::OR:
      case vector::CombiningKind::XOR:
        return rewriter.notifyMatchFailure(reduceOp, "unimplemented");
      default:
        return rewriter.notifyMatchFailure(reduceOp, "not handled here");
      }
#undef INT_AND_FLOAT_CASE
#undef INT_OR_FLOAT_CASE
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

template <typename SPIRVFMaxOp, typename SPIRVFMinOp>
struct VectorReductionFloatMinMax final
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp reduceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reductionInfo =
        getReductionInfo(reduceOp, adaptor, rewriter, *getTypeConverter());
    if (failed(reductionInfo))
      return failure();

    auto [resultType, extractedElements] = *reductionInfo;
    Location loc = reduceOp->getLoc();
    Value result = extractedElements.front();
    for (Value next : llvm::drop_begin(extractedElements)) {
      switch (reduceOp.getKind()) {

#define INT_OR_FLOAT_CASE(kind, fop)                                           \
  case vector::CombiningKind::kind:                                            \
    result = fop::create(rewriter, loc, resultType, result, next);             \
    break

        INT_OR_FLOAT_CASE(MAXIMUMF, SPIRVFMaxOp);
        INT_OR_FLOAT_CASE(MINIMUMF, SPIRVFMinOp);
        INT_OR_FLOAT_CASE(MAXNUMF, SPIRVFMaxOp);
        INT_OR_FLOAT_CASE(MINNUMF, SPIRVFMinOp);

      default:
        return rewriter.notifyMatchFailure(reduceOp, "not handled here");
      }
#undef INT_OR_FLOAT_CASE
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

class VectorScalarBroadcastPattern final
    : public OpConversionPattern<vector::BroadcastOp> {
public:
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<VectorType>(op.getSourceType())) {
      return rewriter.notifyMatchFailure(
          op, "only conversion of 'broadcast from scalar' is supported");
    }
    Type dstType = getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return failure();
    if (isa<spirv::ScalarType>(dstType)) {
      rewriter.replaceOp(op, adaptor.getSource());
    } else {
      auto dstVecType = cast<VectorType>(dstType);
      SmallVector<Value, 4> source(dstVecType.getNumElements(),
                                   adaptor.getSource());
      rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, dstType,
                                                               source);
    }
    return success();
  }
};

struct VectorShuffleOpConvert final
    : public OpConversionPattern<vector::ShuffleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType oldResultType = shuffleOp.getResultVectorType();
    Type newResultType = getTypeConverter()->convertType(oldResultType);
    if (!newResultType)
      return rewriter.notifyMatchFailure(shuffleOp,
                                         "unsupported result vector type");

    auto mask = llvm::to_vector_of<int32_t>(shuffleOp.getMask());

    VectorType oldV1Type = shuffleOp.getV1VectorType();
    VectorType oldV2Type = shuffleOp.getV2VectorType();

    // When both operands and the result are SPIR-V vectors, emit a SPIR-V
    // shuffle.
    if (oldV1Type.getNumElements() > 1 && oldV2Type.getNumElements() > 1 &&
        oldResultType.getNumElements() > 1) {
      rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
          shuffleOp, newResultType, adaptor.getV1(), adaptor.getV2(),
          rewriter.getI32ArrayAttr(mask));
      return success();
    }

    // When at least one of the operands or the result becomes a scalar after
    // type conversion for SPIR-V, extract all the required elements and
    // construct the result vector.
    auto getElementAtIdx = [&rewriter, loc = shuffleOp.getLoc()](
                               Value scalarOrVec, int32_t idx) -> Value {
      if (auto vecTy = dyn_cast<VectorType>(scalarOrVec.getType()))
        return spirv::CompositeExtractOp::create(rewriter, loc, scalarOrVec,
                                                 idx);

      assert(idx == 0 && "Invalid scalar element index");
      return scalarOrVec;
    };

    int32_t numV1Elems = oldV1Type.getNumElements();
    SmallVector<Value> newOperands(mask.size());
    for (auto [shuffleIdx, newOperand] : llvm::zip_equal(mask, newOperands)) {
      Value vec = adaptor.getV1();
      int32_t elementIdx = shuffleIdx;
      if (elementIdx >= numV1Elems) {
        vec = adaptor.getV2();
        elementIdx -= numV1Elems;
      }

      newOperand = getElementAtIdx(vec, elementIdx);
    }

    // Handle the scalar result corner case.
    if (newOperands.size() == 1) {
      rewriter.replaceOp(shuffleOp, newOperands.front());
      return success();
    }

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        shuffleOp, newResultType, newOperands);
    return success();
  }
};

struct VectorInterleaveOpConvert final
    : public OpConversionPattern<vector::InterleaveOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InterleaveOp interleaveOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check the result vector type.
    VectorType oldResultType = interleaveOp.getResultVectorType();
    Type newResultType = getTypeConverter()->convertType(oldResultType);
    if (!newResultType)
      return rewriter.notifyMatchFailure(interleaveOp,
                                         "unsupported result vector type");

    // Interleave the indices.
    VectorType sourceType = interleaveOp.getSourceVectorType();
    int n = sourceType.getNumElements();

    // Input vectors of size 1 are converted to scalars by the type converter.
    // We cannot use `spirv::VectorShuffleOp` directly in this case, and need to
    // use `spirv::CompositeConstructOp`.
    if (n == 1) {
      Value newOperands[] = {adaptor.getLhs(), adaptor.getRhs()};
      rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
          interleaveOp, newResultType, newOperands);
      return success();
    }

    auto seq = llvm::seq<int64_t>(2 * n);
    auto indices = llvm::map_to_vector(
        seq, [n](int i) { return (i % 2 ? n : 0) + i / 2; });

    // Emit a SPIR-V shuffle.
    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
        interleaveOp, newResultType, adaptor.getLhs(), adaptor.getRhs(),
        rewriter.getI32ArrayAttr(indices));

    return success();
  }
};

struct VectorDeinterleaveOpConvert final
    : public OpConversionPattern<vector::DeinterleaveOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::DeinterleaveOp deinterleaveOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check the result vector type.
    VectorType oldResultType = deinterleaveOp.getResultVectorType();
    Type newResultType = getTypeConverter()->convertType(oldResultType);
    if (!newResultType)
      return rewriter.notifyMatchFailure(deinterleaveOp,
                                         "unsupported result vector type");

    Location loc = deinterleaveOp->getLoc();

    // Deinterleave the indices.
    Value sourceVector = adaptor.getSource();
    VectorType sourceType = deinterleaveOp.getSourceVectorType();
    int n = sourceType.getNumElements();

    // Output vectors of size 1 are converted to scalars by the type converter.
    // We cannot use `spirv::VectorShuffleOp` directly in this case, and need to
    // use `spirv::CompositeExtractOp`.
    if (n == 2) {
      auto elem0 = spirv::CompositeExtractOp::create(
          rewriter, loc, newResultType, sourceVector,
          rewriter.getI32ArrayAttr({0}));

      auto elem1 = spirv::CompositeExtractOp::create(
          rewriter, loc, newResultType, sourceVector,
          rewriter.getI32ArrayAttr({1}));

      rewriter.replaceOp(deinterleaveOp, {elem0, elem1});
      return success();
    }

    // Indices for `shuffleEven` (result 0).
    auto seqEven = llvm::seq<int64_t>(n / 2);
    auto indicesEven =
        llvm::map_to_vector(seqEven, [](int i) { return i * 2; });

    // Indices for `shuffleOdd` (result 1).
    auto seqOdd = llvm::seq<int64_t>(n / 2);
    auto indicesOdd =
        llvm::map_to_vector(seqOdd, [](int i) { return i * 2 + 1; });

    // Create two SPIR-V shuffles.
    auto shuffleEven = spirv::VectorShuffleOp::create(
        rewriter, loc, newResultType, sourceVector, sourceVector,
        rewriter.getI32ArrayAttr(indicesEven));

    auto shuffleOdd = spirv::VectorShuffleOp::create(
        rewriter, loc, newResultType, sourceVector, sourceVector,
        rewriter.getI32ArrayAttr(indicesOdd));

    rewriter.replaceOp(deinterleaveOp, {shuffleEven, shuffleOdd});
    return success();
  }
};

struct VectorLoadOpConverter final
    : public OpConversionPattern<vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = loadOp.getMemRefType();
    auto attr =
        dyn_cast_or_null<spirv::StorageClassAttr>(memrefType.getMemorySpace());
    if (!attr)
      return rewriter.notifyMatchFailure(
          loadOp, "expected spirv.storage_class memory space");

    const auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    auto loc = loadOp.getLoc();
    Value accessChain =
        spirv::getElementPtr(typeConverter, memrefType, adaptor.getBase(),
                             adaptor.getIndices(), loc, rewriter);
    if (!accessChain)
      return rewriter.notifyMatchFailure(
          loadOp, "failed to get memref element pointer");

    spirv::StorageClass storageClass = attr.getValue();
    auto vectorType = loadOp.getVectorType();
    // Use the converted vector type instead of original (single element vector
    // would get converted to scalar).
    auto spirvVectorType = typeConverter.convertType(vectorType);
    if (!spirvVectorType)
      return rewriter.notifyMatchFailure(loadOp, "unsupported vector type");

    auto vectorPtrType = spirv::PointerType::get(spirvVectorType, storageClass);

    std::optional<uint64_t> alignment = loadOp.getAlignment();
    if (alignment > std::numeric_limits<uint32_t>::max()) {
      return rewriter.notifyMatchFailure(loadOp,
                                         "invalid alignment requirement");
    }

    auto memoryAccess = spirv::MemoryAccess::None;
    spirv::MemoryAccessAttr memoryAccessAttr;
    IntegerAttr alignmentAttr;
    if (alignment.has_value()) {
      memoryAccess = memoryAccess | spirv::MemoryAccess::Aligned;
      memoryAccessAttr =
          spirv::MemoryAccessAttr::get(rewriter.getContext(), memoryAccess);
      alignmentAttr = rewriter.getI32IntegerAttr(alignment.value());
    }

    // For single element vectors, we don't need to bitcast the access chain to
    // the original vector type. Both is going to be the same, a pointer
    // to a scalar.
    Value castedAccessChain =
        (vectorType.getNumElements() == 1)
            ? accessChain
            : spirv::BitcastOp::create(rewriter, loc, vectorPtrType,
                                       accessChain);

    rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, spirvVectorType,
                                               castedAccessChain,
                                               memoryAccessAttr, alignmentAttr);

    return success();
  }
};

struct VectorStoreOpConverter final
    : public OpConversionPattern<vector::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = storeOp.getMemRefType();
    auto attr =
        dyn_cast_or_null<spirv::StorageClassAttr>(memrefType.getMemorySpace());
    if (!attr)
      return rewriter.notifyMatchFailure(
          storeOp, "expected spirv.storage_class memory space");

    const auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    auto loc = storeOp.getLoc();
    Value accessChain =
        spirv::getElementPtr(typeConverter, memrefType, adaptor.getBase(),
                             adaptor.getIndices(), loc, rewriter);
    if (!accessChain)
      return rewriter.notifyMatchFailure(
          storeOp, "failed to get memref element pointer");

    std::optional<uint64_t> alignment = storeOp.getAlignment();
    if (alignment > std::numeric_limits<uint32_t>::max()) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "invalid alignment requirement");
    }

    spirv::StorageClass storageClass = attr.getValue();
    auto vectorType = storeOp.getVectorType();
    auto vectorPtrType = spirv::PointerType::get(vectorType, storageClass);

    // For single element vectors, we don't need to bitcast the access chain to
    // the original vector type. Both is going to be the same, a pointer
    // to a scalar.
    Value castedAccessChain =
        (vectorType.getNumElements() == 1)
            ? accessChain
            : spirv::BitcastOp::create(rewriter, loc, vectorPtrType,
                                       accessChain);

    auto memoryAccess = spirv::MemoryAccess::None;
    spirv::MemoryAccessAttr memoryAccessAttr;
    IntegerAttr alignmentAttr;
    if (alignment.has_value()) {
      memoryAccess = memoryAccess | spirv::MemoryAccess::Aligned;
      memoryAccessAttr =
          spirv::MemoryAccessAttr::get(rewriter.getContext(), memoryAccess);
      alignmentAttr = rewriter.getI32IntegerAttr(alignment.value());
    }

    rewriter.replaceOpWithNewOp<spirv::StoreOp>(
        storeOp, castedAccessChain, adaptor.getValueToStore(), memoryAccessAttr,
        alignmentAttr);

    return success();
  }
};

struct VectorReductionToIntDotProd final
    : OpRewritePattern<vector::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(op, "combining kind is not 'add'");

    auto resultType = dyn_cast<IntegerType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result is not an integer");

    int64_t resultBitwidth = resultType.getIntOrFloatBitWidth();
    if (!llvm::is_contained({32, 64}, resultBitwidth))
      return rewriter.notifyMatchFailure(op, "unsupported integer bitwidth");

    VectorType inVecTy = op.getSourceVectorType();
    if (!llvm::is_contained({4, 3}, inVecTy.getNumElements()) ||
        inVecTy.getShape().size() != 1 || inVecTy.isScalable())
      return rewriter.notifyMatchFailure(op, "unsupported vector shape");

    auto mul = op.getVector().getDefiningOp<arith::MulIOp>();
    if (!mul)
      return rewriter.notifyMatchFailure(
          op, "reduction operand is not 'arith.muli'");

    if (succeeded(handleCase<arith::ExtSIOp, arith::ExtSIOp, spirv::SDotOp,
                             spirv::SDotAccSatOp, false>(op, mul, rewriter)))
      return success();

    if (succeeded(handleCase<arith::ExtUIOp, arith::ExtUIOp, spirv::UDotOp,
                             spirv::UDotAccSatOp, false>(op, mul, rewriter)))
      return success();

    if (succeeded(handleCase<arith::ExtSIOp, arith::ExtUIOp, spirv::SUDotOp,
                             spirv::SUDotAccSatOp, false>(op, mul, rewriter)))
      return success();

    if (succeeded(handleCase<arith::ExtUIOp, arith::ExtSIOp, spirv::SUDotOp,
                             spirv::SUDotAccSatOp, true>(op, mul, rewriter)))
      return success();

    return failure();
  }

private:
  template <typename LhsExtensionOp, typename RhsExtensionOp, typename DotOp,
            typename DotAccOp, bool SwapOperands>
  static LogicalResult handleCase(vector::ReductionOp op, arith::MulIOp mul,
                                  PatternRewriter &rewriter) {
    auto lhs = mul.getLhs().getDefiningOp<LhsExtensionOp>();
    if (!lhs)
      return failure();
    Value lhsIn = lhs.getIn();
    auto lhsInType = cast<VectorType>(lhsIn.getType());
    if (!lhsInType.getElementType().isInteger(8))
      return failure();

    auto rhs = mul.getRhs().getDefiningOp<RhsExtensionOp>();
    if (!rhs)
      return failure();
    Value rhsIn = rhs.getIn();
    auto rhsInType = cast<VectorType>(rhsIn.getType());
    if (!rhsInType.getElementType().isInteger(8))
      return failure();

    if (op.getSourceVectorType().getNumElements() == 3) {
      IntegerType i8Type = rewriter.getI8Type();
      auto v4i8Type = VectorType::get({4}, i8Type);
      Location loc = op.getLoc();
      Value zero = spirv::ConstantOp::getZero(i8Type, loc, rewriter);
      lhsIn = spirv::CompositeConstructOp::create(rewriter, loc, v4i8Type,
                                                  ValueRange{lhsIn, zero});
      rhsIn = spirv::CompositeConstructOp::create(rewriter, loc, v4i8Type,
                                                  ValueRange{rhsIn, zero});
    }

    // There's no variant of dot prod ops for unsigned LHS and signed RHS, so
    // we have to swap operands instead in that case.
    if (SwapOperands)
      std::swap(lhsIn, rhsIn);

    if (Value acc = op.getAcc()) {
      rewriter.replaceOpWithNewOp<DotAccOp>(op, op.getType(), lhsIn, rhsIn, acc,
                                            nullptr);
    } else {
      rewriter.replaceOpWithNewOp<DotOp>(op, op.getType(), lhsIn, rhsIn,
                                         nullptr);
    }

    return success();
  }
};

struct VectorReductionToFPDotProd final
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(op, "combining kind is not 'add'");

    auto resultType = getTypeConverter()->convertType<FloatType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result is not a float");

    Value vec = adaptor.getVector();
    Value acc = adaptor.getAcc();

    auto vectorType = dyn_cast<VectorType>(vec.getType());
    if (!vectorType) {
      assert(isa<FloatType>(vec.getType()) &&
             "Expected the vector to be scalarized");
      if (acc) {
        rewriter.replaceOpWithNewOp<spirv::FAddOp>(op, acc, vec);
        return success();
      }

      rewriter.replaceOp(op, vec);
      return success();
    }

    Location loc = op.getLoc();
    Value lhs;
    Value rhs;
    if (auto mul = vec.getDefiningOp<arith::MulFOp>()) {
      lhs = mul.getLhs();
      rhs = mul.getRhs();
    } else {
      // If the operand is not a mul, use a vector of ones for the dot operand
      // to just sum up all values.
      lhs = vec;
      Attribute oneAttr =
          rewriter.getFloatAttr(vectorType.getElementType(), 1.0);
      oneAttr = SplatElementsAttr::get(vectorType, oneAttr);
      rhs = spirv::ConstantOp::create(rewriter, loc, vectorType, oneAttr);
    }
    assert(lhs);
    assert(rhs);

    Value res = spirv::DotOp::create(rewriter, loc, resultType, lhs, rhs);
    if (acc)
      res = spirv::FAddOp::create(rewriter, loc, acc, res);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct VectorStepOpConvert final : OpConversionPattern<vector::StepOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::StepOp stepOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    Type dstType = typeConverter.convertType(stepOp.getType());
    if (!dstType)
      return failure();

    Location loc = stepOp.getLoc();
    int64_t numElements = stepOp.getType().getNumElements();
    auto intType =
        rewriter.getIntegerType(typeConverter.getIndexTypeBitwidth());

    // Input vectors of size 1 are converted to scalars by the type converter.
    // We just create a constant in this case.
    if (numElements == 1) {
      Value zero = spirv::ConstantOp::getZero(intType, loc, rewriter);
      rewriter.replaceOp(stepOp, zero);
      return success();
    }

    SmallVector<Value> source;
    source.reserve(numElements);
    for (int64_t i = 0; i < numElements; ++i) {
      Attribute intAttr = rewriter.getIntegerAttr(intType, i);
      Value constOp =
          spirv::ConstantOp::create(rewriter, loc, intType, intAttr);
      source.push_back(constOp);
    }
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(stepOp, dstType,
                                                             source);
    return success();
  }
};

struct VectorToElementOpConvert final
    : OpConversionPattern<vector::ToElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ToElementsOp toElementsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value> results(toElementsOp->getNumResults());
    Location loc = toElementsOp.getLoc();

    // Input vectors of size 1 are converted to scalars by the type converter.
    // We cannot use `spirv::CompositeExtractOp` directly in this case.
    // For a scalar source, the result is just the scalar itself.
    if (isa<spirv::ScalarType>(adaptor.getSource().getType())) {
      results[0] = adaptor.getSource();
      rewriter.replaceOp(toElementsOp, results);
      return success();
    }

    Type srcElementType = toElementsOp.getElements().getType().front();
    Type elementType = getTypeConverter()->convertType(srcElementType);
    if (!elementType)
      return rewriter.notifyMatchFailure(
          toElementsOp,
          llvm::formatv("failed to convert element type '{0}' to SPIR-V",
                        srcElementType));

    for (auto [idx, element] : llvm::enumerate(toElementsOp.getElements())) {
      // Create an CompositeExtract operation only for results that are not
      // dead.
      if (element.use_empty())
        continue;

      Value result = spirv::CompositeExtractOp::create(
          rewriter, loc, elementType, adaptor.getSource(),
          rewriter.getI32ArrayAttr({static_cast<int32_t>(idx)}));
      results[idx] = result;
    }

    rewriter.replaceOp(toElementsOp, results);
    return success();
  }
};

} // namespace
#define CL_INT_MAX_MIN_OPS                                                     \
  spirv::CLUMaxOp, spirv::CLUMinOp, spirv::CLSMaxOp, spirv::CLSMinOp

#define GL_INT_MAX_MIN_OPS                                                     \
  spirv::GLUMaxOp, spirv::GLUMinOp, spirv::GLSMaxOp, spirv::GLSMinOp

#define CL_FLOAT_MAX_MIN_OPS spirv::CLFMaxOp, spirv::CLFMinOp
#define GL_FLOAT_MAX_MIN_OPS spirv::GLFMaxOp, spirv::GLFMinOp

void mlir::populateVectorToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      VectorBitcastConvert, VectorBroadcastConvert, VectorExtractOpConvert,
      VectorExtractStridedSliceOpConvert, VectorFmaOpConvert<spirv::GLFmaOp>,
      VectorFmaOpConvert<spirv::CLFmaOp>, VectorFromElementsOpConvert,
      VectorToElementOpConvert, VectorInsertOpConvert,
      VectorReductionPattern<GL_INT_MAX_MIN_OPS>,
      VectorReductionPattern<CL_INT_MAX_MIN_OPS>,
      VectorReductionFloatMinMax<CL_FLOAT_MAX_MIN_OPS>,
      VectorReductionFloatMinMax<GL_FLOAT_MAX_MIN_OPS>, VectorShapeCast,
      VectorSplatToBroadcast, VectorInsertStridedSliceOpConvert,
      VectorShuffleOpConvert, VectorInterleaveOpConvert,
      VectorDeinterleaveOpConvert, VectorScalarBroadcastPattern,
      VectorLoadOpConverter, VectorStoreOpConverter, VectorStepOpConvert>(
      typeConverter, patterns.getContext(), PatternBenefit(1));

  // Make sure that the more specialized dot product pattern has higher benefit
  // than the generic one that extracts all elements.
  patterns.add<VectorReductionToFPDotProd>(typeConverter, patterns.getContext(),
                                           PatternBenefit(2));
}

void mlir::populateVectorReductionToSPIRVDotProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorReductionToIntDotProd>(patterns.getContext());
}
