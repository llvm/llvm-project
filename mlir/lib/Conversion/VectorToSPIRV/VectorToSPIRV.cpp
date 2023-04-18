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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <numeric>

using namespace mlir;

/// Gets the first integer value from `attr`, assuming it is an integer array
/// attribute.
static uint64_t getFirstIntValue(ArrayAttr attr) {
  return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
}

/// Returns the number of bits for the given scalar/vector type.
static int getNumBits(Type type) {
  if (auto vectorType = type.dyn_cast<VectorType>())
    return vectorType.cast<ShapedType>().getSizeInBits();
  return type.getIntOrFloatBitWidth();
}

namespace {

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

    if (resultType.isa<spirv::ScalarType>()) {
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
    // Only support extracting a scalar value now.
    VectorType resultVectorType = extractOp.getType().dyn_cast<VectorType>();
    if (resultVectorType && resultVectorType.getNumElements() > 1)
      return failure();

    Type dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();

    if (adaptor.getVector().getType().isa<spirv::ScalarType>()) {
      rewriter.replaceOp(extractOp, adaptor.getVector());
      return success();
    }

    int32_t id = getFirstIntValue(extractOp.getPosition());
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
    if (dstType.isa<spirv::ScalarType>()) {
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
    // Special case for inserting scalar values into size-1 vectors.
    if (insertOp.getSourceType().isIntOrFloat() &&
        insertOp.getDestVectorType().getNumElements() == 1) {
      rewriter.replaceOp(insertOp, adaptor.getSource());
      return success();
    }

    if (insertOp.getSourceType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(insertOp.getDestVectorType()))
      return failure();
    int32_t id = getFirstIntValue(insertOp.getPosition());
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

    if (adaptor.getVector().getType().isa<spirv::ScalarType>()) {
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

    if (vectorType.isa<spirv::ScalarType>()) {
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

    if (srcVector.getType().isa<spirv::ScalarType>()) {
      assert(!dstVector.getType().isa<spirv::ScalarType>());
      rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
          insertOp, dstVector.getType(), srcVector, dstVector,
          rewriter.getI32ArrayAttr(offset));
      return success();
    }

    uint64_t totalSize =
        dstVector.getType().cast<VectorType>().getNumElements();
    uint64_t insertSize =
        srcVector.getType().cast<VectorType>().getNumElements();

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

template <class SPIRVFMaxOp, class SPIRVFMinOp, class SPIRVUMaxOp,
          class SPIRVUMinOp, class SPIRVSMaxOp, class SPIRVSMinOp>
struct VectorReductionPattern final
    : public OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp reduceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(reduceOp.getType());
    if (!resultType)
      return failure();

    auto srcVectorType = adaptor.getVector().getType().dyn_cast<VectorType>();
    if (!srcVectorType || srcVectorType.getRank() != 1)
      return rewriter.notifyMatchFailure(reduceOp, "not 1-D vector source");

    // Extract all elements.
    int numElements = srcVectorType.getDimSize(0);
    SmallVector<Value, 4> values;
    values.reserve(numElements + (adaptor.getAcc() != nullptr));
    Location loc = reduceOp.getLoc();
    for (int i = 0; i < numElements; ++i) {
      values.push_back(rewriter.create<spirv::CompositeExtractOp>(
          loc, srcVectorType.getElementType(), adaptor.getVector(),
          rewriter.getI32ArrayAttr({i})));
    }
    if (Value acc = adaptor.getAcc())
      values.push_back(acc);

    // Reduce them.
    Value result = values.front();
    for (Value next : llvm::ArrayRef(values).drop_front()) {
      switch (reduceOp.getKind()) {

#define INT_AND_FLOAT_CASE(kind, iop, fop)                                     \
  case vector::CombiningKind::kind:                                            \
    if (resultType.isa<IntegerType>()) {                                       \
      result = rewriter.create<spirv::iop>(loc, resultType, result, next);     \
    } else {                                                                   \
      assert(resultType.isa<FloatType>());                                     \
      result = rewriter.create<spirv::fop>(loc, resultType, result, next);     \
    }                                                                          \
    break

#define INT_OR_FLOAT_CASE(kind, fop)                                           \
  case vector::CombiningKind::kind:                                            \
    result = rewriter.create<fop>(loc, resultType, result, next);              \
    break

        INT_AND_FLOAT_CASE(ADD, IAddOp, FAddOp);
        INT_AND_FLOAT_CASE(MUL, IMulOp, FMulOp);

        INT_OR_FLOAT_CASE(MAXF, SPIRVFMaxOp);
        INT_OR_FLOAT_CASE(MINF, SPIRVFMinOp);
        INT_OR_FLOAT_CASE(MINUI, SPIRVUMinOp);
        INT_OR_FLOAT_CASE(MINSI, SPIRVSMinOp);
        INT_OR_FLOAT_CASE(MAXUI, SPIRVUMaxOp);
        INT_OR_FLOAT_CASE(MAXSI, SPIRVSMaxOp);

      case vector::CombiningKind::AND:
      case vector::CombiningKind::OR:
      case vector::CombiningKind::XOR:
        return rewriter.notifyMatchFailure(reduceOp, "unimplemented");
      }
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
    if (dstType.isa<spirv::ScalarType>()) {
      rewriter.replaceOp(op, adaptor.getInput());
    } else {
      auto dstVecType = dstType.cast<VectorType>();
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
    if (!spirv::CompositeType::isValid(oldResultType))
      return failure();
    Type newResultType = getTypeConverter()->convertType(oldResultType);

    auto oldSourceType = shuffleOp.getV1VectorType();
    if (oldSourceType.getNumElements() > 1) {
      SmallVector<int32_t, 4> components = llvm::to_vector<4>(
          llvm::map_range(shuffleOp.getMask(), [](Attribute attr) -> int32_t {
            return attr.cast<IntegerAttr>().getValue().getZExtValue();
          }));
      rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
          shuffleOp, newResultType, adaptor.getV1(), adaptor.getV2(),
          rewriter.getI32ArrayAttr(components));
      return success();
    }

    SmallVector<Value, 2> oldOperands = {adaptor.getV1(), adaptor.getV2()};
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(oldResultType.getNumElements());
    for (const APInt &i : shuffleOp.getMask().getAsValueRange<IntegerAttr>()) {
      newOperands.push_back(oldOperands[i.getZExtValue()]);
    }
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        shuffleOp, newResultType, newOperands);

    return success();
  }
};

struct VectorReductionToDotProd final : OpRewritePattern<vector::ReductionOp> {
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

} // namespace
#define CL_MAX_MIN_OPS                                                         \
  spirv::CLFMaxOp, spirv::CLFMinOp, spirv::CLUMaxOp, spirv::CLUMinOp,          \
      spirv::CLSMaxOp, spirv::CLSMinOp

#define GL_MAX_MIN_OPS                                                         \
  spirv::GLFMaxOp, spirv::GLFMinOp, spirv::GLUMaxOp, spirv::GLUMinOp,          \
      spirv::GLSMaxOp, spirv::GLSMinOp

void mlir::populateVectorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<
      VectorBitcastConvert, VectorBroadcastConvert,
      VectorExtractElementOpConvert, VectorExtractOpConvert,
      VectorExtractStridedSliceOpConvert, VectorFmaOpConvert<spirv::GLFmaOp>,
      VectorFmaOpConvert<spirv::CLFmaOp>, VectorInsertElementOpConvert,
      VectorInsertOpConvert, VectorReductionPattern<GL_MAX_MIN_OPS>,
      VectorReductionPattern<CL_MAX_MIN_OPS>, VectorInsertStridedSliceOpConvert,
      VectorShuffleOpConvert, VectorSplatPattern>(typeConverter,
                                                  patterns.getContext());
}

void mlir::populateVectorReductionToSPIRVDotProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorReductionToDotProd>(patterns.getContext());
}
