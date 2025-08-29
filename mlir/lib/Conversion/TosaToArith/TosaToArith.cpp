//===- TosaToArith.cpp - Lowering Tosa to Arith Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Arith dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace tosa;

namespace {

class ConstOpConverter : public OpRewritePattern<tosa::ConstOp> {
public:
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValues());
    return success();
  }
};

Type matchContainerType(Type element, Type container) {
  if (auto shapedTy = dyn_cast<ShapedType>(container))
    return shapedTy.clone(element);

  return element;
}

TypedAttr getConstantAttr(Type type, int64_t value, PatternRewriter &rewriter) {
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    Type eTy = shapedTy.getElementType();
    APInt valueInt(eTy.getIntOrFloatBitWidth(), value, /*isSigned=*/true);
    return DenseIntElementsAttr::get(shapedTy, valueInt);
  }

  return rewriter.getIntegerAttr(type, value);
}

Value getConstantValue(Location loc, Type type, int64_t value,
                       PatternRewriter &rewriter) {
  return arith::ConstantOp::create(rewriter, loc,
                                   getConstantAttr(type, value, rewriter));
}

// This converts the TOSA ApplyScale operator to a set of arithmetic ops,
// using 64-bit operations to perform the necessary multiply, bias, and shift.
class ApplyScaleGenericOpConverter
    : public OpRewritePattern<tosa::ApplyScaleOp> {
public:
  using OpRewritePattern<tosa::ApplyScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ApplyScaleOp op,
                                PatternRewriter &rewriter) const final {
    RoundingMode roundingMode = op.getRoundingMode();
    if (roundingMode != RoundingMode::DOUBLE_ROUND &&
        roundingMode != RoundingMode::SINGLE_ROUND) {
      return failure();
    }

    Location loc = op.getLoc();
    Value value = op.getValue();
    Value multiplier32 = op.getMultiplier();

    Type resultTy = op.getType();
    Type valueTy = value.getType();
    Type i32Ty = matchContainerType(rewriter.getI32Type(), resultTy);
    Type i64Ty = matchContainerType(rewriter.getI64Type(), resultTy);

    Value zero = getConstantValue(loc, valueTy, 0, rewriter);
    Value one64 = getConstantValue(loc, i64Ty, 1, rewriter);
    Value thirtyOne32 = getConstantValue(loc, i32Ty, 31, rewriter);

    Value shift32 = arith::ExtUIOp::create(rewriter, loc, i32Ty, op.getShift());

    // Compute the multiplication in 64-bits then select the high / low parts.
    Value value64 = value;
    if (getElementTypeOrSelf(valueTy) != rewriter.getI64Type())
      value64 = arith::ExtSIOp::create(rewriter, loc, i64Ty, value);
    Value multiplier64 =
        arith::ExtSIOp::create(rewriter, loc, i64Ty, multiplier32);
    Value multiply64 =
        arith::MulIOp::create(rewriter, loc, value64, multiplier64);

    // Apply normal rounding.
    Value shift64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, shift32);
    Value round = arith::ShLIOp::create(rewriter, loc, one64, shift64);
    round = arith::ShRUIOp::create(rewriter, loc, round, one64);
    multiply64 = arith::AddIOp::create(rewriter, loc, multiply64, round);

    // Apply double rounding if necessary.
    if (op.getRoundingMode() == RoundingMode::DOUBLE_ROUND) {
      int64_t roundInt = 1 << 30;
      Value roundUp = getConstantValue(loc, i64Ty, roundInt, rewriter);
      Value roundDown = getConstantValue(loc, i64Ty, -roundInt, rewriter);
      Value positive = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sge, value, zero);
      Value dir =
          arith::SelectOp::create(rewriter, loc, positive, roundUp, roundDown);
      Value val = arith::AddIOp::create(rewriter, loc, dir, multiply64);
      Value valid = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sgt, shift32, thirtyOne32);
      multiply64 =
          arith::SelectOp::create(rewriter, loc, valid, val, multiply64);
    }

    Value result64 = arith::ShRSIOp::create(rewriter, loc, multiply64, shift64);
    Value result32 = arith::TruncIOp::create(rewriter, loc, i32Ty, result64);

    rewriter.replaceOp(op, result32);
    return success();
  }
};

class ApplyScale32BitOpConverter : public OpRewritePattern<tosa::ApplyScaleOp> {
public:
  using OpRewritePattern<tosa::ApplyScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ApplyScaleOp op,
                                PatternRewriter &rewriter) const final {
    RoundingMode roundingMode = op.getRoundingMode();
    if (roundingMode != RoundingMode::DOUBLE_ROUND &&
        roundingMode != RoundingMode::SINGLE_ROUND) {
      return failure();
    }

    Location loc = op.getLoc();

    Type resultTy = op.getType();
    Type i32Ty = matchContainerType(rewriter.getI32Type(), resultTy);

    Value value = op.getValue();
    if (getElementTypeOrSelf(value.getType()).getIntOrFloatBitWidth() > 32) {
      return failure();
    }

    Value value32 = op.getValue();
    Value multiplier32 = op.getMultiplier();
    Value shift32 = arith::ExtUIOp::create(rewriter, loc, i32Ty, op.getShift());

    // Constants used during the scaling operation.
    Value zero32 = getConstantValue(loc, i32Ty, 0, rewriter);
    Value one32 = getConstantValue(loc, i32Ty, 1, rewriter);
    Value two32 = getConstantValue(loc, i32Ty, 2, rewriter);
    Value thirty32 = getConstantValue(loc, i32Ty, 30, rewriter);
    Value thirtyTwo32 = getConstantValue(loc, i32Ty, 32, rewriter);

    // Compute the multiplication in 64-bits then select the high / low parts.
    // Grab out the high/low of the computation
    auto value64 =
        arith::MulSIExtendedOp::create(rewriter, loc, value32, multiplier32);
    Value low32 = value64.getLow();
    Value high32 = value64.getHigh();

    // Determine the direction and amount to shift the high bits.
    Value shiftOver32 = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sge, shift32, thirtyTwo32);
    Value roundHighBits = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sgt, shift32, thirtyTwo32);

    Value shiftHighL =
        arith::SubIOp::create(rewriter, loc, thirtyTwo32, shift32);
    Value shiftHighR =
        arith::SubIOp::create(rewriter, loc, shift32, thirtyTwo32);

    shiftHighL =
        arith::SelectOp::create(rewriter, loc, shiftOver32, zero32, shiftHighL);
    shiftHighR =
        arith::SelectOp::create(rewriter, loc, shiftOver32, shiftHighR, zero32);

    // Conditionally perform our double round.
    if (op.getRoundingMode() == RoundingMode::DOUBLE_ROUND) {
      Value negOne32 = getConstantValue(loc, i32Ty, -1, rewriter);
      Value valuePositive = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sge, value32, zero32);

      Value roundDir = arith::SelectOp::create(rewriter, loc, valuePositive,
                                               one32, negOne32);
      roundDir =
          arith::SelectOp::create(rewriter, loc, shiftOver32, roundDir, zero32);

      Value shiftLow = arith::ShRUIOp::create(rewriter, loc, low32, thirty32);
      Value rounded = arith::AddIOp::create(rewriter, loc, shiftLow, roundDir);
      Value carry = arith::ShRSIOp::create(rewriter, loc, rounded, two32);

      Value shiftRound =
          arith::ShLIOp::create(rewriter, loc, roundDir, thirty32);

      low32 = arith::AddIOp::create(rewriter, loc, low32, shiftRound);
      high32 = arith::AddIOp::create(rewriter, loc, high32, carry);
    }

    // Conditionally apply rounding in the low bits.
    {
      Value shiftSubOne = arith::SubIOp::create(rewriter, loc, shift32, one32);
      Value roundBit = arith::ShLIOp::create(rewriter, loc, one32, shiftSubOne);
      roundBit = arith::SelectOp::create(rewriter, loc, roundHighBits, zero32,
                                         roundBit);

      Value newLow32 = arith::AddIOp::create(rewriter, loc, low32, roundBit);
      Value wasRounded = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ugt, low32, newLow32);
      low32 = newLow32;

      Value rounded32 =
          arith::ExtUIOp::create(rewriter, loc, i32Ty, wasRounded);
      high32 = arith::AddIOp::create(rewriter, loc, high32, rounded32);
    }

    // Conditionally apply rounding in the high bits.
    {
      Value shiftSubOne =
          arith::SubIOp::create(rewriter, loc, shiftHighR, one32);
      Value roundBit = arith::ShLIOp::create(rewriter, loc, one32, shiftSubOne);
      roundBit = arith::SelectOp::create(rewriter, loc, roundHighBits, roundBit,
                                         zero32);
      high32 = arith::AddIOp::create(rewriter, loc, high32, roundBit);
    }

    // Combine the correct high/low bits into the final rescale result.
    high32 = arith::ShLIOp::create(rewriter, loc, high32, shiftHighL);
    high32 = arith::ShRSIOp::create(rewriter, loc, high32, shiftHighR);
    low32 = arith::ShRUIOp::create(rewriter, loc, low32, shift32);
    low32 = arith::SelectOp::create(rewriter, loc, shiftOver32, zero32, low32);

    // Apply the rounding behavior and shift to the final alignment.
    Value result = arith::AddIOp::create(rewriter, loc, low32, high32);

    // Truncate if necessary.
    if (!getElementTypeOrSelf(resultTy).isInteger(32)) {
      result = arith::TruncIOp::create(rewriter, loc, resultTy, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToArithConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<ConstOpConverter>(patterns->getContext());
}

void mlir::tosa::populateTosaRescaleToArithConversionPatterns(
    RewritePatternSet *patterns, bool include32Bit) {
  patterns->add<ApplyScaleGenericOpConverter>(patterns->getContext(), 100);
  if (include32Bit) {
    patterns->add<ApplyScale32BitOpConverter>(patterns->getContext(), 200);
  }
}
