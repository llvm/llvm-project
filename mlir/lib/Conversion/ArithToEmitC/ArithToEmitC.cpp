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
    ArithOpConversion<arith::MulFOp, emitc::MulOp>,
    ArithOpConversion<arith::SubFOp, emitc::SubOp>,
    IntegerOpConversion<arith::AddIOp, emitc::AddOp>,
    IntegerOpConversion<arith::MulIOp, emitc::MulOp>,
    IntegerOpConversion<arith::SubIOp, emitc::SubOp>,
    CmpIOpConversion,
    SelectOpConversion
  >(typeConverter, ctx);
  // clang-format on
}
