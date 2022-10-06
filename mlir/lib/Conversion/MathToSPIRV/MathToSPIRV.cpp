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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "math-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Creates a 32-bit scalar/vector integer constant. Returns nullptr if the
/// given type is not a 32-bit scalar/vector type.
static Value getScalarOrVectorI32Constant(Type type, int value,
                                          OpBuilder &builder, Location loc) {
  if (auto vectorType = type.dyn_cast<VectorType>()) {
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

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {
/// Converts math.copysign to SPIR-V ops.
class CopySignPattern final : public OpConversionPattern<math::CopySignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CopySignOp copySignOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(copySignOp.getType());
    if (!type)
      return failure();

    FloatType floatType;
    if (auto scalarType = copySignOp.getType().dyn_cast<FloatType>()) {
      floatType = scalarType;
    } else if (auto vectorType = copySignOp.getType().dyn_cast<VectorType>()) {
      floatType = vectorType.getElementType().cast<FloatType>();
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

    if (auto vectorType = copySignOp.getType().dyn_cast<VectorType>()) {
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
class CountLeadingZerosPattern final
    : public OpConversionPattern<math::CountLeadingZerosOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CountLeadingZerosOp countOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(countOp.getType());
    if (!type)
      return failure();

    // We can only support 32-bit integer types for now.
    unsigned bitwidth = 0;
    if (type.isa<IntegerType>())
      bitwidth = type.getIntOrFloatBitWidth();
    if (auto vectorType = type.dyn_cast<VectorType>())
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
    Location loc = operation.getLoc();
    auto type = this->getTypeConverter()->convertType(operation.getType());
    auto exp = rewriter.create<ExpOp>(loc, type, adaptor.getOperand());
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
    Location loc = operation.getLoc();
    auto type = this->getTypeConverter()->convertType(operation.getType());
    auto one = spirv::ConstantOp::getOne(type, operation.getLoc(), rewriter);
    auto onePlus =
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
    auto dstType = getTypeConverter()->convertType(powfOp.getType());
    if (!dstType)
      return failure();

    // Per GL Pow extended instruction spec:
    // "Result is undefined if x < 0. Result is undefined if x = 0 and y <= 0."
    Location loc = powfOp.getLoc();
    Value zero =
        spirv::ConstantOp::getZero(adaptor.getLhs().getType(), loc, rewriter);
    Value lessThan =
        rewriter.create<spirv::FOrdLessThanOp>(loc, adaptor.getLhs(), zero);
    Value abs = rewriter.create<spirv::GLFAbsOp>(loc, adaptor.getLhs());
    Value pow = rewriter.create<spirv::GLPowOp>(loc, abs, adaptor.getRhs());
    Value negate = rewriter.create<spirv::FNegateOp>(loc, pow);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(powfOp, lessThan, negate, pow);
    return success();
  }
};

/// Converts math.round to GLSL SPIRV extended ops.
struct RoundOpPattern final : public OpConversionPattern<math::RoundOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RoundOp roundOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = roundOp.getLoc();
    auto operand = roundOp.getOperand();
    auto ty = operand.getType();
    auto ety = getElementTypeOrSelf(ty);

    auto zero = spirv::ConstantOp::getZero(ty, loc, rewriter);
    auto one = spirv::ConstantOp::getOne(ty, loc, rewriter);
    Value half;
    if (VectorType vty = ty.dyn_cast<VectorType>()) {
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
           spirv::ElementwiseOpPattern<math::AbsFOp, spirv::GLFAbsOp>,
           spirv::ElementwiseOpPattern<math::AbsIOp, spirv::GLSAbsOp>,
           spirv::ElementwiseOpPattern<math::CeilOp, spirv::GLCeilOp>,
           spirv::ElementwiseOpPattern<math::CosOp, spirv::GLCosOp>,
           spirv::ElementwiseOpPattern<math::ExpOp, spirv::GLExpOp>,
           spirv::ElementwiseOpPattern<math::FloorOp, spirv::GLFloorOp>,
           spirv::ElementwiseOpPattern<math::FmaOp, spirv::GLFmaOp>,
           spirv::ElementwiseOpPattern<math::LogOp, spirv::GLLogOp>,
           spirv::ElementwiseOpPattern<math::RsqrtOp, spirv::GLInverseSqrtOp>,
           spirv::ElementwiseOpPattern<math::SinOp, spirv::GLSinOp>,
           spirv::ElementwiseOpPattern<math::SqrtOp, spirv::GLSqrtOp>,
           spirv::ElementwiseOpPattern<math::TanhOp, spirv::GLTanhOp>>(
          typeConverter, patterns.getContext());

  // OpenCL patterns
  patterns.add<Log1pOpPattern<spirv::CLLogOp>, ExpM1OpPattern<spirv::CLExpOp>,
               spirv::ElementwiseOpPattern<math::AbsFOp, spirv::CLFAbsOp>,
               spirv::ElementwiseOpPattern<math::CeilOp, spirv::CLCeilOp>,
               spirv::ElementwiseOpPattern<math::CosOp, spirv::CLCosOp>,
               spirv::ElementwiseOpPattern<math::ErfOp, spirv::CLErfOp>,
               spirv::ElementwiseOpPattern<math::ExpOp, spirv::CLExpOp>,
               spirv::ElementwiseOpPattern<math::FloorOp, spirv::CLFloorOp>,
               spirv::ElementwiseOpPattern<math::FmaOp, spirv::CLFmaOp>,
               spirv::ElementwiseOpPattern<math::LogOp, spirv::CLLogOp>,
               spirv::ElementwiseOpPattern<math::PowFOp, spirv::CLPowOp>,
               spirv::ElementwiseOpPattern<math::RoundOp, spirv::CLRoundOp>,
               spirv::ElementwiseOpPattern<math::RsqrtOp, spirv::CLRsqrtOp>,
               spirv::ElementwiseOpPattern<math::SinOp, spirv::CLSinOp>,
               spirv::ElementwiseOpPattern<math::SqrtOp, spirv::CLSqrtOp>,
               spirv::ElementwiseOpPattern<math::TanhOp, spirv::CLTanhOp>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir
