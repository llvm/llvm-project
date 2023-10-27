//===- MathToSPIRV.cpp - Math to SPIR-V Patterns --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Math dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "math-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Creates a 32-bit scalar/vector integer constant. Returns nullptr if the
/// given type is not a 32-bit scalar/vector type.
static Value getScalarOrVectorI32Constant(Type type, int value,
                                          OpBuilder &builder, Location loc) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (!vectorType.getElementType().isInteger(32))
      return nullptr;
    SmallVector<int> values(vectorType.getNumElements(), value);
    return builder.create<spirv::ConstantOp>(loc, type,
                                             builder.getI32VectorAttr(values));
  }
  if (type.isInteger(32))
    return builder.create<spirv::ConstantOp>(loc, type,
                                             builder.getI32IntegerAttr(value));

  return nullptr;
}

/// Check if the type is supported by math-to-spirv conversion. We expect to
/// only see scalars and vectors at this point, with higher-level types already
/// lowered.
static bool isSupportedSourceType(Type originalType) {
  if (originalType.isIntOrIndexOrFloat())
    return true;

  if (auto vecTy = dyn_cast<VectorType>(originalType)) {
    if (!vecTy.getElementType().isIntOrIndexOrFloat())
      return false;
    if (vecTy.isScalable())
      return false;
    if (vecTy.getRank() > 1)
      return false;

    return true;
  }

  return false;
}

/// Check if all `sourceOp` types are supported by math-to-spirv conversion.
/// Notify of a match failure othwerise and return a `failure` result.
/// This is intended to simplify type checks in `OpConversionPattern`s.
static LogicalResult checkSourceOpTypes(ConversionPatternRewriter &rewriter,
                                        Operation *sourceOp) {
  auto allTypes = llvm::to_vector(sourceOp->getOperandTypes());
  llvm::append_range(allTypes, sourceOp->getResultTypes());

  for (Type ty : allTypes) {
    if (!isSupportedSourceType(ty)) {
      return rewriter.notifyMatchFailure(
          sourceOp,
          llvm::formatv(
              "unsupported source type for Math to SPIR-V conversion: {0}",
              ty));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {
/// Converts elementwise unary, binary, and ternary standard operations to
/// SPIR-V operations. Checks that source `Op` types are supported.
template <typename Op, typename SPIRVOp>
struct CheckedElementwiseOpPattern final
    : public spirv::ElementwiseOpPattern<Op, SPIRVOp> {
  using BasePattern = typename spirv::ElementwiseOpPattern<Op, SPIRVOp>;
  using BasePattern::BasePattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (LogicalResult res = checkSourceOpTypes(rewriter, op); failed(res))
      return res;

    return BasePattern::matchAndRewrite(op, adaptor, rewriter);
  }
};

/// Converts math.copysign to SPIR-V ops.
struct CopySignPattern final : public OpConversionPattern<math::CopySignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CopySignOp copySignOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (LogicalResult res = checkSourceOpTypes(rewriter, copySignOp);
        failed(res))
      return res;

    Type type = getTypeConverter()->convertType(copySignOp.getType());
    if (!type)
      return failure();

    FloatType floatType;
    if (auto scalarType = dyn_cast<FloatType>(copySignOp.getType())) {
      floatType = scalarType;
    } else if (auto vectorType = dyn_cast<VectorType>(copySignOp.getType())) {
      floatType = cast<FloatType>(vectorType.getElementType());
    } else {
      return failure();
    }

    Location loc = copySignOp.getLoc();
    int bitwidth = floatType.getWidth();
    Type intType = rewriter.getIntegerType(bitwidth);
    uint64_t intValue = uint64_t(1) << (bitwidth - 1);

    Value signMask = rewriter.create<spirv::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, intValue));
    Value valueMask = rewriter.create<spirv::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, intValue - 1u));

    if (auto vectorType = dyn_cast<VectorType>(type)) {
      assert(vectorType.getRank() == 1);
      int count = vectorType.getNumElements();
      intType = VectorType::get(count, intType);

      SmallVector<Value> signSplat(count, signMask);
      signMask =
          rewriter.create<spirv::CompositeConstructOp>(loc, intType, signSplat);

      SmallVector<Value> valueSplat(count, valueMask);
      valueMask = rewriter.create<spirv::CompositeConstructOp>(loc, intType,
                                                               valueSplat);
    }

    Value lhsCast =
        rewriter.create<spirv::BitcastOp>(loc, intType, adaptor.getLhs());
    Value rhsCast =
        rewriter.create<spirv::BitcastOp>(loc, intType, adaptor.getRhs());

    Value value = rewriter.create<spirv::BitwiseAndOp>(
        loc, intType, ValueRange{lhsCast, valueMask});
    Value sign = rewriter.create<spirv::BitwiseAndOp>(
        loc, intType, ValueRange{rhsCast, signMask});

    Value result = rewriter.create<spirv::BitwiseOrOp>(loc, intType,
                                                       ValueRange{value, sign});
    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(copySignOp, type, result);
    return success();
  }
};

/// Converts math.ctlz to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for counting leading zeros. If
/// Shader capability is supported, we can leverage GL FindUMsb to calculate
/// it.
struct CountLeadingZerosPattern final
    : public OpConversionPattern<math::CountLeadingZerosOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CountLeadingZerosOp countOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (LogicalResult res = checkSourceOpTypes(rewriter, countOp); failed(res))
      return res;

    Type type = getTypeConverter()->convertType(countOp.getType());
    if (!type)
      return failure();

    // We can only support 32-bit integer types for now.
    unsigned bitwidth = 0;
    if (isa<IntegerType>(type))
      bitwidth = type.getIntOrFloatBitWidth();
    if (auto vectorType = dyn_cast<VectorType>(type))
      bitwidth = vectorType.getElementTypeBitWidth();
    if (bitwidth != 32)
      return failure();

    Location loc = countOp.getLoc();
    Value input = adaptor.getOperand();
    Value val1 = getScalarOrVectorI32Constant(type, 1, rewriter, loc);
    Value val31 = getScalarOrVectorI32Constant(type, 31, rewriter, loc);
    Value val32 = getScalarOrVectorI32Constant(type, 32, rewriter, loc);

    Value msb = rewriter.create<spirv::GLFindUMsbOp>(loc, input);
    // We need to subtract from 31 given that the index returned by GLSL
    // FindUMsb is counted from the least significant bit. Theoretically this
    // also gives the correct result even if the integer has all zero bits, in
    // which case GL FindUMsb would return -1.
    Value subMsb = rewriter.create<spirv::ISubOp>(loc, val31, msb);
    // However, certain Vulkan implementations have driver bugs for the corner
    // case where the input is zero. And.. it can be smart to optimize a select
    // only involving the corner case. So separately compute the result when the
    // input is either zero or one.
    Value subInput = rewriter.create<spirv::ISubOp>(loc, val32, input);
    Value cmp = rewriter.create<spirv::ULessThanEqualOp>(loc, input, val1);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(countOp, cmp, subInput,
                                                 subMsb);
    return success();
  }
};

/// Converts math.expm1 to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for exp(x)-1. Explicitly lower to
/// these operations.
template <typename ExpOp>
struct ExpM1OpPattern final : public OpConversionPattern<math::ExpM1Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpM1Op operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 1);
    if (LogicalResult res = checkSourceOpTypes(rewriter, operation);
        failed(res))
      return res;

    Location loc = operation.getLoc();
    Type type = this->getTypeConverter()->convertType(operation.getType());
    if (!type)
      return failure();

    Value exp = rewriter.create<ExpOp>(loc, type, adaptor.getOperand());
    auto one = spirv::ConstantOp::getOne(type, loc, rewriter);
    rewriter.replaceOpWithNewOp<spirv::FSubOp>(operation, exp, one);
    return success();
  }
};

/// Converts math.log1p to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for log(1+x). Explicitly lower to
/// these operations.
template <typename LogOp>
struct Log1pOpPattern final : public OpConversionPattern<math::Log1pOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 1);
    if (LogicalResult res = checkSourceOpTypes(rewriter, operation);
        failed(res))
      return res;

    Location loc = operation.getLoc();
    Type type = this->getTypeConverter()->convertType(operation.getType());
    if (!type)
      return failure();

    auto one = spirv::ConstantOp::getOne(type, operation.getLoc(), rewriter);
    Value onePlus =
        rewriter.create<spirv::FAddOp>(loc, one, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<LogOp>(operation, type, onePlus);
    return success();
  }
};

/// Converts math.powf to SPIRV-Ops.
struct PowFOpPattern final : public OpConversionPattern<math::PowFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::PowFOp powfOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (LogicalResult res = checkSourceOpTypes(rewriter, powfOp); failed(res))
      return res;

    Type dstType = getTypeConverter()->convertType(powfOp.getType());
    if (!dstType)
      return failure();

    // Get the scalar float type.
    FloatType scalarFloatType;
    if (auto scalarType = dyn_cast<FloatType>(powfOp.getType())) {
      scalarFloatType = scalarType;
    } else if (auto vectorType = dyn_cast<VectorType>(powfOp.getType())) {
      scalarFloatType = cast<FloatType>(vectorType.getElementType());
    } else {
      return failure();
    }

    // Get int type of the same shape as the float type.
    Type scalarIntType = rewriter.getIntegerType(32);
    Type intType = scalarIntType;
    if (auto vectorType = dyn_cast<VectorType>(adaptor.getRhs().getType())) {
      auto shape = vectorType.getShape();
      intType = VectorType::get(shape, scalarIntType);
    }

    // Per GL Pow extended instruction spec:
    // "Result is undefined if x < 0. Result is undefined if x = 0 and y <= 0."
    Location loc = powfOp.getLoc();
    Value zero =
        spirv::ConstantOp::getZero(adaptor.getLhs().getType(), loc, rewriter);
    Value lessThan =
        rewriter.create<spirv::FOrdLessThanOp>(loc, adaptor.getLhs(), zero);
    Value abs = rewriter.create<spirv::GLFAbsOp>(loc, adaptor.getLhs());

    // TODO: The following just forcefully casts y into an integer value in
    // order to properly propagate the sign, assuming integer y cases. It
    // doesn't cover other cases and should be fixed.

    // Cast exponent to integer and calculate exponent % 2 != 0.
    Value intRhs =
        rewriter.create<spirv::ConvertFToSOp>(loc, intType, adaptor.getRhs());
    Value intOne = spirv::ConstantOp::getOne(intType, loc, rewriter);
    Value bitwiseAndOne =
        rewriter.create<spirv::BitwiseAndOp>(loc, intRhs, intOne);
    Value isOdd = rewriter.create<spirv::IEqualOp>(loc, bitwiseAndOne, intOne);

    // calculate pow based on abs(lhs)^rhs.
    Value pow = rewriter.create<spirv::GLPowOp>(loc, abs, adaptor.getRhs());
    Value negate = rewriter.create<spirv::FNegateOp>(loc, pow);
    // if the exponent is odd and lhs < 0, negate the result.
    Value shouldNegate =
        rewriter.create<spirv::LogicalAndOp>(loc, lessThan, isOdd);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(powfOp, shouldNegate, negate,
                                                 pow);
    return success();
  }
};

/// Converts math.round to GLSL SPIRV extended ops.
struct RoundOpPattern final : public OpConversionPattern<math::RoundOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RoundOp roundOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (LogicalResult res = checkSourceOpTypes(rewriter, roundOp); failed(res))
      return res;

    Location loc = roundOp.getLoc();
    Value operand = roundOp.getOperand();
    Type ty = operand.getType();
    Type ety = getElementTypeOrSelf(ty);

    auto zero = spirv::ConstantOp::getZero(ty, loc, rewriter);
    auto one = spirv::ConstantOp::getOne(ty, loc, rewriter);
    Value half;
    if (VectorType vty = dyn_cast<VectorType>(ty)) {
      half = rewriter.create<spirv::ConstantOp>(
          loc, vty,
          DenseElementsAttr::get(vty,
                                 rewriter.getFloatAttr(ety, 0.5).getValue()));
    } else {
      half = rewriter.create<spirv::ConstantOp>(
          loc, ty, rewriter.getFloatAttr(ety, 0.5));
    }

    auto abs = rewriter.create<spirv::GLFAbsOp>(loc, operand);
    auto floor = rewriter.create<spirv::GLFloorOp>(loc, abs);
    auto sub = rewriter.create<spirv::FSubOp>(loc, abs, floor);
    auto greater =
        rewriter.create<spirv::FOrdGreaterThanEqualOp>(loc, sub, half);
    auto select = rewriter.create<spirv::SelectOp>(loc, greater, one, zero);
    auto add = rewriter.create<spirv::FAddOp>(loc, floor, select);
    rewriter.replaceOpWithNewOp<math::CopySignOp>(roundOp, add, operand);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateMathToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  // Core patterns
  patterns.add<CopySignPattern>(typeConverter, patterns.getContext());

  // GLSL patterns
  patterns
      .add<CountLeadingZerosPattern, Log1pOpPattern<spirv::GLLogOp>,
           ExpM1OpPattern<spirv::GLExpOp>, PowFOpPattern, RoundOpPattern,
           CheckedElementwiseOpPattern<math::AbsFOp, spirv::GLFAbsOp>,
           CheckedElementwiseOpPattern<math::AbsIOp, spirv::GLSAbsOp>,
           CheckedElementwiseOpPattern<math::CeilOp, spirv::GLCeilOp>,
           CheckedElementwiseOpPattern<math::CosOp, spirv::GLCosOp>,
           CheckedElementwiseOpPattern<math::ExpOp, spirv::GLExpOp>,
           CheckedElementwiseOpPattern<math::FloorOp, spirv::GLFloorOp>,
           CheckedElementwiseOpPattern<math::FmaOp, spirv::GLFmaOp>,
           CheckedElementwiseOpPattern<math::LogOp, spirv::GLLogOp>,
           CheckedElementwiseOpPattern<math::RoundEvenOp, spirv::GLRoundEvenOp>,
           CheckedElementwiseOpPattern<math::RsqrtOp, spirv::GLInverseSqrtOp>,
           CheckedElementwiseOpPattern<math::SinOp, spirv::GLSinOp>,
           CheckedElementwiseOpPattern<math::SqrtOp, spirv::GLSqrtOp>,
           CheckedElementwiseOpPattern<math::TanhOp, spirv::GLTanhOp>>(
          typeConverter, patterns.getContext());

  // OpenCL patterns
  patterns.add<Log1pOpPattern<spirv::CLLogOp>, ExpM1OpPattern<spirv::CLExpOp>,
               CheckedElementwiseOpPattern<math::AbsFOp, spirv::CLFAbsOp>,
               CheckedElementwiseOpPattern<math::AbsIOp, spirv::CLSAbsOp>,
               CheckedElementwiseOpPattern<math::CeilOp, spirv::CLCeilOp>,
               CheckedElementwiseOpPattern<math::CosOp, spirv::CLCosOp>,
               CheckedElementwiseOpPattern<math::ErfOp, spirv::CLErfOp>,
               CheckedElementwiseOpPattern<math::ExpOp, spirv::CLExpOp>,
               CheckedElementwiseOpPattern<math::FloorOp, spirv::CLFloorOp>,
               CheckedElementwiseOpPattern<math::FmaOp, spirv::CLFmaOp>,
               CheckedElementwiseOpPattern<math::LogOp, spirv::CLLogOp>,
               CheckedElementwiseOpPattern<math::PowFOp, spirv::CLPowOp>,
               CheckedElementwiseOpPattern<math::RoundEvenOp, spirv::CLRintOp>,
               CheckedElementwiseOpPattern<math::RoundOp, spirv::CLRoundOp>,
               CheckedElementwiseOpPattern<math::RsqrtOp, spirv::CLRsqrtOp>,
               CheckedElementwiseOpPattern<math::SinOp, spirv::CLSinOp>,
               CheckedElementwiseOpPattern<math::SqrtOp, spirv::CLSqrtOp>,
               CheckedElementwiseOpPattern<math::TanhOp, spirv::CLTanhOp>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir
