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
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
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
static uint64_t getFirstIntValue(ValueRange values) {
  return values[0].getDefiningOp<arith::ConstantIndexOp>().value();
}
static uint64_t getFirstIntValue(ArrayRef<Attribute> attr) {
  return cast<IntegerAttr>(attr[0]).getInt();
}
static uint64_t getFirstIntValue(ArrayAttr attr) {
  return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
}
static uint64_t getFirstIntValue(ArrayRef<OpFoldResult> foldResults) {
  auto attr = foldResults[0].dyn_cast<Attribute>();
  if (attr)
    return getFirstIntValue(attr);

  return getFirstIntValue(ValueRange{foldResults[0].get<Value>()});
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
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        castOp, castOp.getResultVectorType(), source);
    return success();
  }
};

struct VectorExtractOpConvert final
    : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (extractOp.hasDynamicPosition())
      return failure();

    Type dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();

    if (isa<spirv::ScalarType>(adaptor.getVector().getType())) {
      rewriter.replaceOp(extractOp, adaptor.getVector());
      return success();
    }

    int32_t id = getFirstIntValue(extractOp.getMixedPosition());
    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        extractOp, adaptor.getVector(), id);
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

struct VectorInsertOpConvert final
    : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<VectorType>(insertOp.getSourceType()))
      return rewriter.notifyMatchFailure(insertOp, "unsupported vector source");
    if (!getTypeConverter()->convertType(insertOp.getDestVectorType()))
      return rewriter.notifyMatchFailure(insertOp,
                                         "unsupported dest vector type");

    // Special case for inserting scalar values into size-1 vectors.
    if (insertOp.getSourceType().isIntOrFloat() &&
        insertOp.getDestVectorType().getNumElements() == 1) {
      rewriter.replaceOp(insertOp, adaptor.getSource());
      return success();
    }

    int32_t id = getFirstIntValue(insertOp.getMixedPosition());
    rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
        insertOp, adaptor.getSource(), adaptor.getDest(), id);
    return success();
  }
};

struct VectorExtractElementOpConvert final
    : public OpConversionPattern<vector::ExtractElementOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractElementOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(extractOp.getType());
    if (!resultType)
      return failure();

    if (isa<spirv::ScalarType>(adaptor.getVector().getType())) {
      rewriter.replaceOp(extractOp, adaptor.getVector());
      return success();
    }

    APInt cstPos;
    if (matchPattern(adaptor.getPosition(), m_ConstantInt(&cstPos)))
      rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
          extractOp, resultType, adaptor.getVector(),
          rewriter.getI32ArrayAttr({static_cast<int>(cstPos.getSExtValue())}));
    else
      rewriter.replaceOpWithNewOp<spirv::VectorExtractDynamicOp>(
          extractOp, resultType, adaptor.getVector(), adaptor.getPosition());
    return success();
  }
};

struct VectorInsertElementOpConvert final
    : public OpConversionPattern<vector::InsertElementOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertElementOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type vectorType = getTypeConverter()->convertType(insertOp.getType());
    if (!vectorType)
      return failure();

    if (isa<spirv::ScalarType>(vectorType)) {
      rewriter.replaceOp(insertOp, adaptor.getSource());
      return success();
    }

    APInt cstPos;
    if (matchPattern(adaptor.getPosition(), m_ConstantInt(&cstPos)))
      rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
          insertOp, adaptor.getSource(), adaptor.getDest(),
          cstPos.getSExtValue());
    else
      rewriter.replaceOpWithNewOp<spirv::VectorInsertDynamicOp>(
          insertOp, vectorType, insertOp.getDest(), adaptor.getSource(),
          adaptor.getPosition());
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
    values.push_back(rewriter.create<spirv::CompositeExtractOp>(
        loc, srcVectorType.getElementType(), adaptor.getVector(),
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
      result = rewriter.create<spirv::iop>(loc, resultType, result, next);     \
    } else {                                                                   \
      assert(llvm::isa<FloatType>(resultType));                                \
      result = rewriter.create<spirv::fop>(loc, resultType, result, next);     \
    }                                                                          \
    break

#define INT_OR_FLOAT_CASE(kind, fop)                                           \
  case vector::CombiningKind::kind:                                            \
    result = rewriter.create<fop>(loc, resultType, result, next);              \
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
    result = rewriter.create<fop>(loc, resultType, result, next);              \
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

class VectorSplatPattern final : public OpConversionPattern<vector::SplatOp> {
public:
  using OpConversionPattern<vector::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return failure();
    if (isa<spirv::ScalarType>(dstType)) {
      rewriter.replaceOp(op, adaptor.getInput());
    } else {
      auto dstVecType = cast<VectorType>(dstType);
      SmallVector<Value, 4> source(dstVecType.getNumElements(),
                                   adaptor.getInput());
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
    auto oldResultType = shuffleOp.getResultVectorType();
    Type newResultType = getTypeConverter()->convertType(oldResultType);
    if (!newResultType)
      return rewriter.notifyMatchFailure(shuffleOp,
                                         "unsupported result vector type");

    SmallVector<int32_t, 4> mask = llvm::map_to_vector<4>(
        shuffleOp.getMask(), [](Attribute attr) -> int32_t {
          return cast<IntegerAttr>(attr).getValue().getZExtValue();
        });

    auto oldV1Type = shuffleOp.getV1VectorType();
    auto oldV2Type = shuffleOp.getV2VectorType();

    // When both operands are SPIR-V vectors, emit a SPIR-V shuffle.
    if (oldV1Type.getNumElements() > 1 && oldV2Type.getNumElements() > 1) {
      rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
          shuffleOp, newResultType, adaptor.getV1(), adaptor.getV2(),
          rewriter.getI32ArrayAttr(mask));
      return success();
    }

    // When at least one of the operands becomes a scalar after type conversion
    // for SPIR-V, extract all the required elements and construct the result
    // vector.
    auto getElementAtIdx = [&rewriter, loc = shuffleOp.getLoc()](
                               Value scalarOrVec, int32_t idx) -> Value {
      if (auto vecTy = dyn_cast<VectorType>(scalarOrVec.getType()))
        return rewriter.create<spirv::CompositeExtractOp>(loc, scalarOrVec,
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

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        shuffleOp, newResultType, newOperands);

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
    auto vectorPtrType = spirv::PointerType::get(vectorType, storageClass);
    Value castedAccessChain =
        rewriter.create<spirv::BitcastOp>(loc, vectorPtrType, accessChain);
    rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, vectorType,
                                               castedAccessChain);

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

    spirv::StorageClass storageClass = attr.getValue();
    auto vectorType = storeOp.getVectorType();
    auto vectorPtrType = spirv::PointerType::get(vectorType, storageClass);
    Value castedAccessChain =
        rewriter.create<spirv::BitcastOp>(loc, vectorPtrType, accessChain);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, castedAccessChain,
                                                adaptor.getValueToStore());

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
      lhsIn = rewriter.create<spirv::CompositeConstructOp>(
          loc, v4i8Type, ValueRange{lhsIn, zero});
      rhsIn = rewriter.create<spirv::CompositeConstructOp>(
          loc, v4i8Type, ValueRange{rhsIn, zero});
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
      rhs = rewriter.create<spirv::ConstantOp>(loc, vectorType, oneAttr);
    }
    assert(lhs);
    assert(rhs);

    Value res = rewriter.create<spirv::DotOp>(loc, resultType, lhs, rhs);
    if (acc)
      res = rewriter.create<spirv::FAddOp>(loc, acc, res);

    rewriter.replaceOp(op, res);
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

void mlir::populateVectorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<
      VectorBitcastConvert, VectorBroadcastConvert,
      VectorExtractElementOpConvert, VectorExtractOpConvert,
      VectorExtractStridedSliceOpConvert, VectorFmaOpConvert<spirv::GLFmaOp>,
      VectorFmaOpConvert<spirv::CLFmaOp>, VectorInsertElementOpConvert,
      VectorInsertOpConvert, VectorReductionPattern<GL_INT_MAX_MIN_OPS>,
      VectorReductionPattern<CL_INT_MAX_MIN_OPS>,
      VectorReductionFloatMinMax<CL_FLOAT_MAX_MIN_OPS>,
      VectorReductionFloatMinMax<GL_FLOAT_MAX_MIN_OPS>, VectorShapeCast,
      VectorInsertStridedSliceOpConvert, VectorShuffleOpConvert,
      VectorSplatPattern, VectorLoadOpConverter, VectorStoreOpConverter>(
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
