//===- ArithToEmitC.cpp - Arith to EmitC Patterns ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the Arith dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Tools/PDLL/AST/Types.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
class ArithConstantOpConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp arithConst,
                  arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        arithConst, arithConst.getType(), adaptor.getValue());
    return success();
  }
};

class CmpIOpConversion : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  bool needsUnsignedCmp(arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge:
      return false;
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::uge:
      return true;
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  emitc::CmpPredicate toEmitCPred(arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::eq:
      return emitc::CmpPredicate::eq;
    case arith::CmpIPredicate::ne:
      return emitc::CmpPredicate::ne;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return emitc::CmpPredicate::lt;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return emitc::CmpPredicate::le;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return emitc::CmpPredicate::gt;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return emitc::CmpPredicate::ge;
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = adaptor.getLhs().getType();
    if (!isa_and_nonnull<IntegerType, IndexType>(type)) {
      return rewriter.notifyMatchFailure(op, "expected integer or index type");
    }

    bool needsUnsigned = needsUnsignedCmp(op.getPredicate());
    emitc::CmpPredicate pred = toEmitCPred(op.getPredicate());
    Type arithmeticType = type;
    if (type.isUnsignedInteger() != needsUnsigned) {
      arithmeticType = rewriter.getIntegerType(type.getIntOrFloatBitWidth(),
                                               /*isSigned=*/!needsUnsigned);
    }
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (arithmeticType != type) {
      lhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    lhs);
      rhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    rhs);
    }
    rewriter.replaceOpWithNewOp<emitc::CmpOp>(op, op.getType(), pred, lhs, rhs);
    return success();
  }
};

template <typename ArithOp, bool castToUnsigned>
class CastConversion : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type opReturnType = this->getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(opReturnType))
      return rewriter.notifyMatchFailure(op, "expected integer result type");

    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "CastConversion only supports unary ops");
    }

    Type operandType = adaptor.getIn().getType();
    if (!isa_and_nonnull<IntegerType>(operandType))
      return rewriter.notifyMatchFailure(op, "expected integer operand type");

    // Signed (sign-extending) casts from i1 are not supported.
    if (operandType.isInteger(1) && !castToUnsigned)
      return rewriter.notifyMatchFailure(op,
                                         "operation not supported on i1 type");

    // to-i1 conversions: arith semantics want truncation, whereas (bool)(v) is
    // equivalent to (v != 0). Implementing as (bool)(v & 0x01) gives
    // truncation.
    if (opReturnType.isInteger(1)) {
      auto constOne = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), operandType, rewriter.getIntegerAttr(operandType, 1));
      auto oneAndOperand = rewriter.create<emitc::BitwiseAndOp>(
          op.getLoc(), operandType, adaptor.getIn(), constOne);
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, opReturnType,
                                                 oneAndOperand);
      return success();
    }

    bool isTruncation = operandType.getIntOrFloatBitWidth() >
                        opReturnType.getIntOrFloatBitWidth();
    bool doUnsigned = castToUnsigned || isTruncation;

    Type castType = opReturnType;
    // If the op is a ui variant and the type wanted as
    // return type isn't unsigned, we need to issue an unsigned type to do
    // the conversion.
    if (castType.isUnsignedInteger() != doUnsigned) {
      castType = rewriter.getIntegerType(opReturnType.getIntOrFloatBitWidth(),
                                         /*isSigned=*/!doUnsigned);
    }

    Value actualOp = adaptor.getIn();
    // Adapt the signedness of the operand if necessary
    if (operandType.isUnsignedInteger() != doUnsigned) {
      Type correctSignednessType =
          rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/!doUnsigned);
      actualOp = rewriter.template create<emitc::CastOp>(
          op.getLoc(), correctSignednessType, actualOp);
    }

    auto result = rewriter.template create<emitc::CastOp>(op.getLoc(), castType,
                                                          actualOp);

    // Cast to the expected output type
    if (castType != opReturnType) {
      result = rewriter.template create<emitc::CastOp>(op.getLoc(),
                                                       opReturnType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ArithOp>
class UnsignedCastConversion : public CastConversion<ArithOp, true> {
  using CastConversion<ArithOp, true>::CastConversion;
};

template <typename ArithOp>
class SignedCastConversion : public CastConversion<ArithOp, false> {
  using CastConversion<ArithOp, false>::CastConversion;
};

template <typename ArithOp, typename EmitCOp>
class ArithOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp arithOp, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.template replaceOpWithNewOp<EmitCOp>(arithOp, arithOp.getType(),
                                                  adaptor.getOperands());

    return success();
  }
};

template <typename ArithOp, typename EmitCOp>
class IntegerOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType, IndexType>(type)) {
      return rewriter.notifyMatchFailure(op, "expected integer type");
    }

    if (type.isInteger(1)) {
      // arith expects wrap-around arithmethic, which doesn't happen on `bool`.
      return rewriter.notifyMatchFailure(op, "i1 type is not implemented");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type arithmeticType = type;
    if ((type.isSignlessInteger() || type.isSignedInteger()) &&
        !bitEnumContainsAll(op.getOverflowFlags(),
                            arith::IntegerOverflowFlags::nsw)) {
      // If the C type is signed and the op doesn't guarantee "No Signed Wrap",
      // we compute in unsigned integers to avoid UB.
      arithmeticType = rewriter.getIntegerType(type.getIntOrFloatBitWidth(),
                                               /*isSigned=*/false);
    }
    if (arithmeticType != type) {
      lhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    lhs);
      rhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    rhs);
    }

    Value result = rewriter.template create<EmitCOp>(op.getLoc(),
                                                     arithmeticType, lhs, rhs);

    if (arithmeticType != type) {
      result =
          rewriter.template create<emitc::CastOp>(op.getLoc(), type, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp selectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type dstType = getTypeConverter()->convertType(selectOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(selectOp, "type conversion failed");

    if (!adaptor.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          selectOp,
          "can only be converted if condition is a scalar of type i1");

    rewriter.replaceOpWithNewOp<emitc::ConditionalOp>(selectOp, dstType,
                                                      adaptor.getOperands());

    return success();
  }
};

// Floating-point to integer conversions.
template <typename CastOp>
class FtoICastOpConversion : public OpConversionPattern<CastOp> {
public:
  FtoICastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type operandType = adaptor.getIn().getType();
    if (!emitc::isSupportedFloatType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    // Float-to-i1 casts are not supported: any value with 0 < value < 1 must be
    // truncated to 0, whereas a boolean conversion would return true.
    if (!emitc::isSupportedIntegerType(dstType) || dstType.isInteger(1))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    // Convert to unsigned if it's the "ui" variant
    // Signless is interpreted as signed, so no need to cast for "si"
    Type actualResultType = dstType;
    if (isa<arith::FPToUIOp>(castOp)) {
      actualResultType =
          rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/false);
    }

    Value result = rewriter.create<emitc::CastOp>(
        castOp.getLoc(), actualResultType, adaptor.getOperands());

    if (isa<arith::FPToUIOp>(castOp)) {
      result = rewriter.create<emitc::CastOp>(castOp.getLoc(), dstType, result);
    }
    rewriter.replaceOp(castOp, result);

    return success();
  }
};

// Integer to floating-point conversions.
template <typename CastOp>
class ItoFCastOpConversion : public OpConversionPattern<CastOp> {
public:
  ItoFCastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Vectors in particular are not supported
    Type operandType = adaptor.getIn().getType();
    if (!emitc::isSupportedIntegerType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    if (!emitc::isSupportedFloatType(dstType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    // Convert to unsigned if it's the "ui" variant
    // Signless is interpreted as signed, so no need to cast for "si"
    Type actualOperandType = operandType;
    if (isa<arith::UIToFPOp>(castOp)) {
      actualOperandType =
          rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/false);
    }
    Value fpCastOperand = adaptor.getIn();
    if (actualOperandType != operandType) {
      fpCastOperand = rewriter.template create<emitc::CastOp>(
          castOp.getLoc(), actualOperandType, fpCastOperand);
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(castOp, dstType, fpCastOperand);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateArithToEmitCPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // clang-format off
  patterns.add<
    ArithConstantOpConversionPattern,
    ArithOpConversion<arith::AddFOp, emitc::AddOp>,
    ArithOpConversion<arith::DivFOp, emitc::DivOp>,
    ArithOpConversion<arith::DivSIOp, emitc::DivOp>,
    ArithOpConversion<arith::MulFOp, emitc::MulOp>,
    ArithOpConversion<arith::RemSIOp, emitc::RemOp>,
    ArithOpConversion<arith::SubFOp, emitc::SubOp>,
    IntegerOpConversion<arith::AddIOp, emitc::AddOp>,
    IntegerOpConversion<arith::MulIOp, emitc::MulOp>,
    IntegerOpConversion<arith::SubIOp, emitc::SubOp>,
    CmpIOpConversion,
    SelectOpConversion,
    // Truncation is guaranteed for unsigned types.
    UnsignedCastConversion<arith::TruncIOp>,
    SignedCastConversion<arith::ExtSIOp>,
    UnsignedCastConversion<arith::ExtUIOp>,
    ItoFCastOpConversion<arith::SIToFPOp>,
    ItoFCastOpConversion<arith::UIToFPOp>,
    FtoICastOpConversion<arith::FPToSIOp>,
    FtoICastOpConversion<arith::FPToUIOp>
  >(typeConverter, ctx);
  // clang-format on
}
