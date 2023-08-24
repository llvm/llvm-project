//===- ExpandTanh.cpp - Code to perform expanding tanh op -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of tanh op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Create a float constant.
static Value createFloatConst(Location loc, Type type, double value,
                              OpBuilder &b) {
  auto attr = b.getFloatAttr(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return b.create<arith::ConstantOp>(loc,
                                       DenseElementsAttr::get(shapedTy, attr));
  }

  return b.create<arith::ConstantOp>(loc, attr);
}

/// Create a float constant.
static Value createIntConst(Location loc, Type type, int64_t value,
                            OpBuilder &b) {
  auto attr = b.getIntegerAttr(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return b.create<arith::ConstantOp>(loc,
                                       DenseElementsAttr::get(shapedTy, attr));
  }

  return b.create<arith::ConstantOp>(loc, attr);
}

static Value createTruncatedFPValue(Value operand, ImplicitLocOpBuilder &b) {
  Type opType = operand.getType();
  Type i64Ty = b.getI64Type();
  if (auto shapedTy = dyn_cast<ShapedType>(opType))
    i64Ty = shapedTy.clone(i64Ty);
  Value fixedConvert = b.create<arith::FPToSIOp>(i64Ty, operand);
  Value fpFixedConvert = b.create<arith::SIToFPOp>(opType, fixedConvert);
  // The truncation does not preserve the sign when the truncated
  // value is -0. So here the sign is copied again.
  return b.create<math::CopySignOp>(fpFixedConvert, operand);
}

/// Expands tanh op into
///   1) 1-exp^{-2x} / 1+exp^{-2x}, if x => 0
///   2) exp^{2x}-1 / exp^{2x}+1  , if x < 0
static LogicalResult convertTanhOp(math::TanhOp op, PatternRewriter &rewriter) {
  auto floatType = op.getOperand().getType();
  Location loc = op.getLoc();
  Value one = createFloatConst(loc, floatType, 1.0, rewriter);
  Value two = createFloatConst(loc, floatType, 2.0, rewriter);
  Value doubledX = rewriter.create<arith::MulFOp>(loc, op.getOperand(), two);

  // Case 1: tanh(x) = 1-exp^{-2x} / 1+exp^{-2x}
  Value negDoubledX = rewriter.create<arith::NegFOp>(loc, doubledX);
  Value exp2x = rewriter.create<math::ExpOp>(loc, negDoubledX);
  Value dividend = rewriter.create<arith::SubFOp>(loc, one, exp2x);
  Value divisor = rewriter.create<arith::AddFOp>(loc, one, exp2x);
  Value positiveRes = rewriter.create<arith::DivFOp>(loc, dividend, divisor);

  // Case 2: tanh(x) = exp^{2x}-1 / exp^{2x}+1
  exp2x = rewriter.create<math::ExpOp>(loc, doubledX);
  dividend = rewriter.create<arith::SubFOp>(loc, exp2x, one);
  divisor = rewriter.create<arith::AddFOp>(loc, exp2x, one);
  Value negativeRes = rewriter.create<arith::DivFOp>(loc, dividend, divisor);

  // tanh(x) = x >= 0 ? positiveRes : negativeRes
  Value zero = createFloatConst(loc, floatType, 0.0, rewriter);
  Value cmpRes = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                                op.getOperand(), zero);
  rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmpRes, positiveRes,
                                               negativeRes);
  return success();
}

// Converts math.tan to math.sin, math.cos, and arith.divf.
static LogicalResult convertTanOp(math::TanOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type type = operand.getType();
  Value sin = b.create<math::SinOp>(type, operand);
  Value cos = b.create<math::CosOp>(type, operand);
  Value div = b.create<arith::DivFOp>(type, sin, cos);
  rewriter.replaceOp(op, div);
  return success();
}

static LogicalResult convertFmaFOp(math::FmaOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operandA = op.getOperand(0);
  Value operandB = op.getOperand(1);
  Value operandC = op.getOperand(2);
  Type type = op.getType();
  Value mult = b.create<arith::MulFOp>(type, operandA, operandB);
  Value add = b.create<arith::AddFOp>(type, mult, operandC);
  rewriter.replaceOp(op, add);
  return success();
}

// Converts a floorf() function to the following:
// floorf(float x) ->
//     y = (float)(int) x
//     if (x < 0) then incr = -1 else incr = 0
//     y = y + incr    <= replace this op with the floorf op.
static LogicalResult convertFloorOp(math::FloorOp op,
                                    PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();
  Value fpFixedConvert = createTruncatedFPValue(operand, b);

  // Creating constants for later use.
  Value zero = createFloatConst(op->getLoc(), opType, 0.00, rewriter);
  Value negOne = createFloatConst(op->getLoc(), opType, -1.00, rewriter);

  Value negCheck =
      b.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, operand, zero);
  Value incrValue =
      b.create<arith::SelectOp>(op->getLoc(), negCheck, negOne, zero);
  Value ret = b.create<arith::AddFOp>(opType, fpFixedConvert, incrValue);
  rewriter.replaceOp(op, ret);
  return success();
}

// Converts a ceilf() function to the following:
// ceilf(float x) ->
//      y = (float)(int) x
//      if (x > y) then incr = 1 else incr = 0
//      y = y + incr   <= replace this op with the ceilf op.
static LogicalResult convertCeilOp(math::CeilOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();
  Value fpFixedConvert = createTruncatedFPValue(operand, b);

  // Creating constants for later use.
  Value zero = createFloatConst(op->getLoc(), opType, 0.00, rewriter);
  Value one = createFloatConst(op->getLoc(), opType, 1.00, rewriter);

  Value gtCheck = b.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, operand,
                                          fpFixedConvert);
  Value incrValue = b.create<arith::SelectOp>(op->getLoc(), gtCheck, one, zero);

  Value ret = b.create<arith::AddFOp>(opType, fpFixedConvert, incrValue);
  rewriter.replaceOp(op, ret);
  return success();
}
// Converts  Powf(float a, float b) (meaning a^b) to exp^(b * ln(a))
static LogicalResult convertPowfOp(math::PowFOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operandA = op.getOperand(0);
  Value operandB = op.getOperand(1);
  Type opType = operandA.getType();

  Value logA = b.create<math::LogOp>(opType, operandA);
  Value mult = b.create<arith::MulFOp>(opType, logA, operandB);
  Value expResult = b.create<math::ExpOp>(opType, mult);
  rewriter.replaceOp(op, expResult);
  return success();
}

// exp2f(float x) -> exp(x * ln(2))
//   Proof: Let's say 2^x = y
//   ln(2^x) = ln(y)
//   x * ln(2) = ln(y) => e ^(x*ln(2)) = y
static LogicalResult convertExp2fOp(math::Exp2Op op,
                                    PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();
  Value ln2 = createFloatConst(op->getLoc(), opType, llvm::numbers::ln2, b);
  Value mult = b.create<arith::MulFOp>(opType, operand, ln2);
  Value exp = b.create<math::ExpOp>(op->getLoc(), mult);
  rewriter.replaceOp(op, exp);
  return success();
}

static LogicalResult convertRoundOp(math::RoundOp op,
                                    PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();
  Type opEType = getElementTypeOrSelf(opType);

  if (!opEType.isF32()) {
    return rewriter.notifyMatchFailure(op, "not a round of f32.");
  }

  Type i32Ty = b.getI32Type();
  if (auto shapedTy = dyn_cast<ShapedType>(opType))
    i32Ty = shapedTy.clone(i32Ty);

  Value half = createFloatConst(loc, opType, 0.5, b);
  Value c23 = createIntConst(loc, i32Ty, 23, b);
  Value c127 = createIntConst(loc, i32Ty, 127, b);
  Value expMask = createIntConst(loc, i32Ty, (1 << 8) - 1, b);

  Value incrValue = b.create<math::CopySignOp>(half, operand);
  Value add = b.create<arith::AddFOp>(opType, operand, incrValue);
  Value fpFixedConvert = createTruncatedFPValue(add, b);

  // There are three cases where adding 0.5 to the value and truncating by
  // converting to an i64 does not result in the correct behavior:
  //
  // 1. Special values: +-inf and +-nan
  //     Casting these special values to i64 has undefined behavior. To identify
  //     these values, we use the fact that these values are the only float
  //     values with the maximum possible biased exponent.
  //
  // 2. Large values: 2^23 <= |x| <= INT_64_MAX
  //     Adding 0.5 to a float larger than or equal to 2^23 results in precision
  //     errors that sometimes round the value up and sometimes round the value
  //     down. For example:
  //         8388608.0 + 0.5 = 8388608.0
  //         8388609.0 + 0.5 = 8388610.0
  //
  // 3. Very large values: |x| > INT_64_MAX
  //     Casting to i64 a value greater than the max i64 value will overflow the
  //     i64 leading to wrong outputs.
  //
  // All three cases satisfy the property `biasedExp >= 23`.
  Value operandBitcast = b.create<arith::BitcastOp>(i32Ty, operand);
  Value operandExp = b.create<arith::AndIOp>(
      b.create<arith::ShRUIOp>(operandBitcast, c23), expMask);
  Value operandBiasedExp = b.create<arith::SubIOp>(operandExp, c127);
  Value isSpecialValOrLargeVal =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::sge, operandBiasedExp, c23);

  Value result = b.create<arith::SelectOp>(isSpecialValOrLargeVal, operand,
                                           fpFixedConvert);
  rewriter.replaceOp(op, result);
  return success();
}

// Converts math.ctlz to scf and arith operations. This is done
// by performing a binary search on the bits.
static LogicalResult convertCtlzOp(math::CountLeadingZerosOp op,
                                   PatternRewriter &rewriter) {
  auto operand = op.getOperand();
  auto operandTy = operand.getType();
  auto eTy = getElementTypeOrSelf(operandTy);
  Location loc = op.getLoc();

  int32_t bitwidth = eTy.getIntOrFloatBitWidth();
  if (bitwidth > 64)
    return failure();

  uint64_t allbits = -1;
  if (bitwidth < 64) {
    allbits = allbits >> (64 - bitwidth);
  }

  Value x = operand;
  Value count = createIntConst(loc, operandTy, 0, rewriter);
  for (int32_t bw = bitwidth; bw > 1; bw = bw / 2) {
    auto half = bw / 2;
    auto bits = createIntConst(loc, operandTy, half, rewriter);
    auto mask = createIntConst(loc, operandTy, allbits >> half, rewriter);

    Value pred =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, x, mask);
    Value add = rewriter.create<arith::AddIOp>(loc, count, bits);
    Value shift = rewriter.create<arith::ShLIOp>(loc, x, bits);

    x = rewriter.create<arith::SelectOp>(loc, pred, shift, x);
    count = rewriter.create<arith::SelectOp>(loc, pred, add, count);
  }

  Value zero = createIntConst(loc, operandTy, 0, rewriter);
  Value pred = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              operand, zero);

  Value bwval = createIntConst(loc, operandTy, bitwidth, rewriter);
  Value sel = rewriter.create<arith::SelectOp>(loc, pred, bwval, count);
  rewriter.replaceOp(op, sel);
  return success();
}

// Convert `math.roundeven` into `math.round` + arith ops
static LogicalResult convertRoundEvenOp(math::RoundEvenOp op,
                                        PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);
  auto operand = op.getOperand();
  Type operandTy = operand.getType();
  Type resultTy = op.getType();
  Type operandETy = getElementTypeOrSelf(operandTy);
  Type resultETy = getElementTypeOrSelf(resultTy);

  if (!isa<FloatType>(operandETy) || !isa<FloatType>(resultETy)) {
    return rewriter.notifyMatchFailure(op, "not a roundeven of f16 or f32.");
  }

  Type fTy = operandTy;
  Type iTy = rewriter.getIntegerType(operandETy.getIntOrFloatBitWidth());
  if (auto shapedTy = dyn_cast<ShapedType>(fTy)) {
    iTy = shapedTy.clone(iTy);
  }

  unsigned bitWidth = operandETy.getIntOrFloatBitWidth();
  // The width returned by getFPMantissaWidth includes the integer bit.
  unsigned mantissaWidth =
      llvm::cast<FloatType>(operandETy).getFPMantissaWidth() - 1;
  unsigned exponentWidth = bitWidth - mantissaWidth - 1;

  // The names of the variables correspond to f32.
  // f64: 1 bit sign | 11 bits exponent | 52 bits mantissa.
  // f32: 1 bit sign | 8 bits exponent  | 23 bits mantissa.
  // f16: 1 bit sign | 5 bits exponent  | 10 bits mantissa.
  Value c1Float = createFloatConst(loc, fTy, 1.0, b);
  Value c0 = createIntConst(loc, iTy, 0, b);
  Value c1 = createIntConst(loc, iTy, 1, b);
  Value cNeg1 = createIntConst(loc, iTy, -1, b);
  Value c23 = createIntConst(loc, iTy, mantissaWidth, b);
  Value c31 = createIntConst(loc, iTy, bitWidth - 1, b);
  Value c127 = createIntConst(loc, iTy, (1ull << (exponentWidth - 1)) - 1, b);
  Value c2To22 = createIntConst(loc, iTy, 1ull << (mantissaWidth - 1), b);
  Value c23Mask = createIntConst(loc, iTy, (1ull << mantissaWidth) - 1, b);
  Value expMask = createIntConst(loc, iTy, (1ull << exponentWidth) - 1, b);

  Value operandBitcast = b.create<arith::BitcastOp>(iTy, operand);
  Value round = b.create<math::RoundOp>(operand);
  Value roundBitcast = b.create<arith::BitcastOp>(iTy, round);

  // Get biased exponents for operand and round(operand)
  Value operandExp = b.create<arith::AndIOp>(
      b.create<arith::ShRUIOp>(operandBitcast, c23), expMask);
  Value operandBiasedExp = b.create<arith::SubIOp>(operandExp, c127);
  Value roundExp = b.create<arith::AndIOp>(
      b.create<arith::ShRUIOp>(roundBitcast, c23), expMask);
  Value roundBiasedExp = b.create<arith::SubIOp>(roundExp, c127);

  auto safeShiftRight = [&](Value x, Value shift) -> Value {
    // Clamp shift to valid range [0, bitwidth - 1] to avoid undefined behavior
    Value clampedShift = b.create<arith::MaxSIOp>(shift, c0);
    clampedShift = b.create<arith::MinSIOp>(clampedShift, c31);
    return b.create<arith::ShRUIOp>(x, clampedShift);
  };

  auto maskMantissa = [&](Value mantissa,
                          Value mantissaMaskRightShift) -> Value {
    Value shiftedMantissaMask = safeShiftRight(c23Mask, mantissaMaskRightShift);
    return b.create<arith::AndIOp>(mantissa, shiftedMantissaMask);
  };

  // A whole number `x`, such that `|x| != 1`, is even if the mantissa, ignoring
  // the leftmost `clamp(biasedExp - 1, 0, 23)` bits, is zero. Large numbers
  // with `biasedExp > 23` (numbers where there is not enough precision to store
  // decimals) are always even, and they satisfy the even condition trivially
  // since the mantissa without all its bits is zero. The even condition
  // is also true for +-0, since they have `biasedExp = -127` and the entire
  // mantissa is zero. The case of +-1 has to be handled separately. Here
  // we identify these values by noting that +-1 are the only whole numbers with
  // `biasedExp == 0`.
  //
  // The special values +-inf and +-nan also satisfy the same property that
  // whole non-unit even numbers satisfy. In particular, the special values have
  // `biasedExp > 23`, so they get treated as large numbers with no room for
  // decimals, which are always even.
  Value roundBiasedExpEq0 =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, roundBiasedExp, c0);
  Value roundBiasedExpMinus1 = b.create<arith::SubIOp>(roundBiasedExp, c1);
  Value roundMaskedMantissa = maskMantissa(roundBitcast, roundBiasedExpMinus1);
  Value roundIsNotEvenOrSpecialVal = b.create<arith::CmpIOp>(
      arith::CmpIPredicate::ne, roundMaskedMantissa, c0);
  roundIsNotEvenOrSpecialVal =
      b.create<arith::OrIOp>(roundIsNotEvenOrSpecialVal, roundBiasedExpEq0);

  // A value `x` with `0 <= biasedExp < 23`, is halfway between two consecutive
  // integers if the bit at index `biasedExp` starting from the left in the
  // mantissa is 1 and all the bits to the right are zero. Values with
  // `biasedExp >= 23` don't have decimals, so they are never halfway. The
  // values +-0.5 are the only halfway values that have `biasedExp == -1 < 0`,
  // so these are handled separately. In particular, if `biasedExp == -1`, the
  // value is halfway if the entire mantissa is zero.
  Value operandBiasedExpEqNeg1 = b.create<arith::CmpIOp>(
      arith::CmpIPredicate::eq, operandBiasedExp, cNeg1);
  Value expectedOperandMaskedMantissa = b.create<arith::SelectOp>(
      operandBiasedExpEqNeg1, c0, safeShiftRight(c2To22, operandBiasedExp));
  Value operandMaskedMantissa = maskMantissa(operandBitcast, operandBiasedExp);
  Value operandIsHalfway =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, operandMaskedMantissa,
                              expectedOperandMaskedMantissa);
  // Ensure `biasedExp` is in the valid range for half values.
  Value operandBiasedExpGeNeg1 = b.create<arith::CmpIOp>(
      arith::CmpIPredicate::sge, operandBiasedExp, cNeg1);
  Value operandBiasedExpLt23 =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, operandBiasedExp, c23);
  operandIsHalfway =
      b.create<arith::AndIOp>(operandIsHalfway, operandBiasedExpLt23);
  operandIsHalfway =
      b.create<arith::AndIOp>(operandIsHalfway, operandBiasedExpGeNeg1);

  // Adjust rounded operand with `round(operand) - sign(operand)` to correct the
  // case where `round` rounded in the opposite direction of `roundeven`.
  Value sign = b.create<math::CopySignOp>(c1Float, operand);
  Value roundShifted = b.create<arith::SubFOp>(round, sign);
  // If the rounded value is even or a special value, we default to the behavior
  // of `math.round`.
  Value needsShift =
      b.create<arith::AndIOp>(roundIsNotEvenOrSpecialVal, operandIsHalfway);
  Value result = b.create<arith::SelectOp>(needsShift, roundShifted, round);
  // The `x - sign` adjustment does not preserve the sign when we are adjusting
  // the value -1 to -0. So here the sign is copied again to ensure that -0.5 is
  // rounded to -0.0.
  result = b.create<math::CopySignOp>(result, operand);
  rewriter.replaceOp(op, result);
  return success();
}

void mlir::populateExpandCtlzPattern(RewritePatternSet &patterns) {
  patterns.add(convertCtlzOp);
}

void mlir::populateExpandTanPattern(RewritePatternSet &patterns) {
  patterns.add(convertTanOp);
}

void mlir::populateExpandTanhPattern(RewritePatternSet &patterns) {
  patterns.add(convertTanhOp);
}

void mlir::populateExpandFmaFPattern(RewritePatternSet &patterns) {
  patterns.add(convertFmaFOp);
}

void mlir::populateExpandCeilFPattern(RewritePatternSet &patterns) {
  patterns.add(convertCeilOp);
}

void mlir::populateExpandExp2FPattern(RewritePatternSet &patterns) {
  patterns.add(convertExp2fOp);
}

void mlir::populateExpandPowFPattern(RewritePatternSet &patterns) {
  patterns.add(convertPowfOp);
}

void mlir::populateExpandRoundFPattern(RewritePatternSet &patterns) {
  patterns.add(convertRoundOp);
}

void mlir::populateExpandFloorFPattern(RewritePatternSet &patterns) {
  patterns.add(convertFloorOp);
}

void mlir::populateExpandRoundEvenPattern(RewritePatternSet &patterns) {
  patterns.add(convertRoundEvenOp);
}
