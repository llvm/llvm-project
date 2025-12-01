//===- ExpandPatterns.cpp - Code to expand various math operations. -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of various math operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::math {
#define GEN_PASS_DEF_MATHEXPANDOPSPASS
#include "mlir/Dialect/Math/Transforms/Passes.h.inc"
} // namespace mlir::math

/// Create a float constant.
static Value createFloatConst(Location loc, Type type, APFloat value,
                              OpBuilder &b) {
  bool losesInfo = false;
  auto eltType = getElementTypeOrSelf(type);
  // Convert double to the given `FloatType` with round-to-nearest-ties-to-even.
  value.convert(cast<FloatType>(eltType).getFloatSemantics(),
                APFloat::rmNearestTiesToEven, &losesInfo);
  auto attr = b.getFloatAttr(eltType, value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return arith::ConstantOp::create(b, loc,
                                     DenseElementsAttr::get(shapedTy, attr));
  }

  return arith::ConstantOp::create(b, loc, attr);
}

static Value createFloatConst(Location loc, Type type, double value,
                              OpBuilder &b) {
  return createFloatConst(loc, type, APFloat(value), b);
}

/// Create an integer constant.
static Value createIntConst(Location loc, Type type, int64_t value,
                            OpBuilder &b) {
  auto attr = b.getIntegerAttr(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return arith::ConstantOp::create(b, loc,
                                     DenseElementsAttr::get(shapedTy, attr));
  }

  return arith::ConstantOp::create(b, loc, attr);
}

static Value createTruncatedFPValue(Value operand, ImplicitLocOpBuilder &b) {
  Type opType = operand.getType();
  Type i64Ty = b.getI64Type();
  if (auto shapedTy = dyn_cast<ShapedType>(opType))
    i64Ty = shapedTy.clone(i64Ty);
  Value fixedConvert = arith::FPToSIOp::create(b, i64Ty, operand);
  Value fpFixedConvert = arith::SIToFPOp::create(b, opType, fixedConvert);
  // The truncation does not preserve the sign when the truncated
  // value is -0. So here the sign is copied again.
  return math::CopySignOp::create(b, fpFixedConvert, operand);
}

// sinhf(float x) -> (exp(x) - exp(-x)) / 2
static LogicalResult convertSinhOp(math::SinhOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();

  Value exp = math::ExpOp::create(b, operand);
  Value neg = arith::NegFOp::create(b, operand);
  Value nexp = math::ExpOp::create(b, neg);
  Value sub = arith::SubFOp::create(b, exp, nexp);
  Value half = createFloatConst(op->getLoc(), opType, 0.5, rewriter);
  Value res = arith::MulFOp::create(b, sub, half);
  rewriter.replaceOp(op, res);
  return success();
}

// coshf(float x) -> (exp(x) + exp(-x)) / 2
static LogicalResult convertCoshOp(math::CoshOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();

  Value exp = math::ExpOp::create(b, operand);
  Value neg = arith::NegFOp::create(b, operand);
  Value nexp = math::ExpOp::create(b, neg);
  Value add = arith::AddFOp::create(b, exp, nexp);
  Value half = createFloatConst(op->getLoc(), opType, 0.5, rewriter);
  Value res = arith::MulFOp::create(b, add, half);
  rewriter.replaceOp(op, res);
  return success();
}

/// Expands tanh op into
/// 1-exp^{-2x} / 1+exp^{-2x}
/// To avoid overflow we exploit the reflection symmetry `tanh(-x) = -tanh(x)`.
/// We compute a "signs" value which is -1 if input is negative and +1 if input
/// is positive.  Then multiply the input by this value, guaranteeing that the
/// result is positive, which also guarantees `exp^{-2x * sign(x)}` is in (0,
/// 1]. Expand the computation on the input `x * sign(x)`, then multiply the
/// result by `sign(x)` to retain sign of the real result.
static LogicalResult convertTanhOp(math::TanhOp op, PatternRewriter &rewriter) {
  auto floatType = op.getOperand().getType();
  Location loc = op.getLoc();
  Value zero = createFloatConst(loc, floatType, 0.0, rewriter);
  Value one = createFloatConst(loc, floatType, 1.0, rewriter);
  Value negTwo = createFloatConst(loc, floatType, -2.0, rewriter);

  // Compute sign(x) = cast<float_type>(x < 0) * (-2) + 1
  Value isNegative = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLT, op.getOperand(), zero);
  Value isNegativeFloat =
      arith::UIToFPOp::create(rewriter, loc, floatType, isNegative);
  Value isNegativeTimesNegTwo =
      arith::MulFOp::create(rewriter, loc, isNegativeFloat, negTwo);
  Value sign = arith::AddFOp::create(rewriter, loc, isNegativeTimesNegTwo, one);

  // Normalize input to positive value: y = sign(x) * x
  Value positiveX = arith::MulFOp::create(rewriter, loc, sign, op.getOperand());

  // Decompose on normalized input
  Value negDoubledX = arith::MulFOp::create(rewriter, loc, negTwo, positiveX);
  Value exp2x = math::ExpOp::create(rewriter, loc, negDoubledX);
  Value dividend = arith::SubFOp::create(rewriter, loc, one, exp2x);
  Value divisor = arith::AddFOp::create(rewriter, loc, one, exp2x);
  Value positiveRes = arith::DivFOp::create(rewriter, loc, dividend, divisor);

  // Multiply result by sign(x) to retain signs from negative inputs
  rewriter.replaceOpWithNewOp<arith::MulFOp>(op, sign, positiveRes);

  return success();
}

// Converts math.tan to math.sin, math.cos, and arith.divf.
static LogicalResult convertTanOp(math::TanOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type type = operand.getType();
  Value sin = math::SinOp::create(b, type, operand);
  Value cos = math::CosOp::create(b, type, operand);
  Value div = arith::DivFOp::create(b, type, sin, cos);
  rewriter.replaceOp(op, div);
  return success();
}

// asinh(float x) -> log(x + sqrt(x**2 + 1))
static LogicalResult convertAsinhOp(math::AsinhOp op,
                                    PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();

  Value one = createFloatConst(op->getLoc(), opType, 1.0, rewriter);
  Value fma = math::FmaOp::create(b, operand, operand, one);
  Value sqrt = math::SqrtOp::create(b, fma);
  Value add = arith::AddFOp::create(b, operand, sqrt);
  Value res = math::LogOp::create(b, add);
  rewriter.replaceOp(op, res);
  return success();
}

// acosh(float x) -> log(x + sqrt(x**2 - 1))
static LogicalResult convertAcoshOp(math::AcoshOp op,
                                    PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();

  Value negOne = createFloatConst(op->getLoc(), opType, -1.0, rewriter);
  Value fma = math::FmaOp::create(b, operand, operand, negOne);
  Value sqrt = math::SqrtOp::create(b, fma);
  Value add = arith::AddFOp::create(b, operand, sqrt);
  Value res = math::LogOp::create(b, add);
  rewriter.replaceOp(op, res);
  return success();
}

// atanh(float x) -> log((1 + x) / (1 - x)) / 2
static LogicalResult convertAtanhOp(math::AtanhOp op,
                                    PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();

  Value one = createFloatConst(op->getLoc(), opType, 1.0, rewriter);
  Value add = arith::AddFOp::create(b, operand, one);
  Value neg = arith::NegFOp::create(b, operand);
  Value sub = arith::AddFOp::create(b, neg, one);
  Value div = arith::DivFOp::create(b, add, sub);
  Value log = math::LogOp::create(b, div);
  Value half = createFloatConst(op->getLoc(), opType, 0.5, rewriter);
  Value res = arith::MulFOp::create(b, log, half);
  rewriter.replaceOp(op, res);
  return success();
}

static LogicalResult convertFmaFOp(math::FmaOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operandA = op.getOperand(0);
  Value operandB = op.getOperand(1);
  Value operandC = op.getOperand(2);
  Type type = op.getType();
  Value mult = arith::MulFOp::create(b, type, operandA, operandB);
  Value add = arith::AddFOp::create(b, type, mult, operandC);
  rewriter.replaceOp(op, add);
  return success();
}

// Converts a ceilf() function to the following:
// ceilf(float x) ->
//      y = (float)(int) x
//      if (x > y) then incr = 1 else incr = 0
//      y = y + incr   <= replace this op with the ceilf op.
static LogicalResult convertCeilOp(math::CeilOp op, PatternRewriter &rewriter) {
  // Creating constants assumes the static shaped type.
  auto shapedType = dyn_cast<ShapedType>(op.getType());
  if (shapedType && !shapedType.hasStaticShape())
    return failure();

  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();
  Value fpFixedConvert = createTruncatedFPValue(operand, b);

  // Creating constants for later use.
  Value zero = createFloatConst(op->getLoc(), opType, 0.00, rewriter);
  Value one = createFloatConst(op->getLoc(), opType, 1.00, rewriter);

  Value gtCheck = arith::CmpFOp::create(b, arith::CmpFPredicate::OGT, operand,
                                        fpFixedConvert);
  Value incrValue =
      arith::SelectOp::create(b, op->getLoc(), gtCheck, one, zero);

  Value ret = arith::AddFOp::create(b, opType, fpFixedConvert, incrValue);
  rewriter.replaceOp(op, ret);
  return success();
}

// Convert `math.fpowi` to a series of `arith.mulf` operations.
// If the power is negative, we divide one by the result.
// If both the base and power are zero, the result is 1.
// In the case of non constant power, we convert the operation to `math.powf`.
static LogicalResult convertFPowIOp(math::FPowIOp op,
                                    PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value base = op.getOperand(0);
  Value power = op.getOperand(1);
  Type baseType = base.getType();

  auto convertFPowItoPowf = [&]() -> LogicalResult {
    Value castPowerToFp =
        arith::SIToFPOp::create(rewriter, op.getLoc(), baseType, power);
    Value res = math::PowFOp::create(rewriter, op.getLoc(), baseType, base,
                                     castPowerToFp);
    rewriter.replaceOp(op, res);
    return success();
  };

  Attribute cstAttr;
  if (!matchPattern(power, m_Constant(&cstAttr)))
    return convertFPowItoPowf();

  APInt value;
  if (!matchPattern(cstAttr, m_ConstantInt(&value)))
    return convertFPowItoPowf();

  int64_t powerInt = value.getSExtValue();
  bool isNegative = powerInt < 0;
  int64_t absPower = std::abs(powerInt);
  Value one = createFloatConst(op->getLoc(), baseType, 1.00, rewriter);
  Value res = createFloatConst(op->getLoc(), baseType, 1.00, rewriter);

  while (absPower > 0) {
    if (absPower & 1)
      res = arith::MulFOp::create(b, baseType, base, res);
    absPower >>= 1;
    base = arith::MulFOp::create(b, baseType, base, base);
  }

  // Make sure not to introduce UB in case of negative power.
  if (isNegative) {
    auto &sem = dyn_cast<mlir::FloatType>(getElementTypeOrSelf(baseType))
                    .getFloatSemantics();
    Value zero =
        createFloatConst(op->getLoc(), baseType,
                         APFloat::getZero(sem, /*Negative=*/false), rewriter);
    Value negZero =
        createFloatConst(op->getLoc(), baseType,
                         APFloat::getZero(sem, /*Negative=*/true), rewriter);
    Value posInfinity =
        createFloatConst(op->getLoc(), baseType,
                         APFloat::getInf(sem, /*Negative=*/false), rewriter);
    Value negInfinity =
        createFloatConst(op->getLoc(), baseType,
                         APFloat::getInf(sem, /*Negative=*/true), rewriter);
    Value zeroEqCheck =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, res, zero);
    Value negZeroEqCheck =
        arith::CmpFOp::create(b, arith::CmpFPredicate::OEQ, res, negZero);
    res = arith::DivFOp::create(b, baseType, one, res);
    res =
        arith::SelectOp::create(b, op->getLoc(), zeroEqCheck, posInfinity, res);
    res = arith::SelectOp::create(b, op->getLoc(), negZeroEqCheck, negInfinity,
                                  res);
  }

  rewriter.replaceOp(op, res);
  return success();
}

// Converts Powf(float a, float b) (meaning a^b) to exp^(b * ln(a))
// Some special cases where b is constant are handled separately:
// when b == 0, or |b| == 0.5, 1.0, or 2.0.
static LogicalResult convertPowfOp(math::PowFOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operandA = op.getOperand(0);
  Value operandB = op.getOperand(1);
  auto typeA = operandA.getType();
  auto typeB = operandB.getType();

  auto &sem =
      cast<mlir::FloatType>(getElementTypeOrSelf(typeB)).getFloatSemantics();
  APFloat valueB(sem);
  auto mulf = [&](Value x, Value y) -> Value {
    return arith::MulFOp::create(b, x, y);
  };
  if (matchPattern(operandB, m_ConstantFloat(&valueB))) {
    if (valueB.isZero()) {
      // a^0 -> 1
      Value one = createFloatConst(op->getLoc(), typeA, 1.0, rewriter);
      rewriter.replaceOp(op, one);
      return success();
    }
    if (valueB.isExactlyValue(1.0)) {
      // a^1 -> a
      rewriter.replaceOp(op, operandA);
      return success();
    }
    if (valueB.isExactlyValue(-1.0)) {
      // a^(-1) -> 1 / a
      Value one = createFloatConst(op->getLoc(), typeA, 1.0, rewriter);
      Value div = arith::DivFOp::create(b, one, operandA);
      rewriter.replaceOp(op, div);
      return success();
    }
    if (valueB.isExactlyValue(0.5)) {
      // a^(1/2) -> sqrt(a)
      Value sqrt = math::SqrtOp::create(b, operandA);
      rewriter.replaceOp(op, sqrt);
      return success();
    }
    if (valueB.isExactlyValue(-0.5)) {
      // a^(-1/2) -> 1 / sqrt(a)
      Value rsqrt = math::RsqrtOp::create(b, operandA);
      rewriter.replaceOp(op, rsqrt);
      return success();
    }
    if (valueB.isExactlyValue(2.0)) {
      // a^2 -> a * a
      rewriter.replaceOp(op, mulf(operandA, operandA));
      return success();
    }
    if (valueB.isExactlyValue(-2.0)) {
      // a^(-2) -> 1 / (a * a)
      Value one =
          createFloatConst(op->getLoc(), operandA.getType(), 1.0, rewriter);
      Value div = arith::DivFOp::create(b, one, mulf(operandA, operandA));
      rewriter.replaceOp(op, div);
      return success();
    }
    if (valueB.isExactlyValue(3.0)) {
      rewriter.replaceOp(op, mulf(mulf(operandA, operandA), operandA));
      return success();
    }
  }

  Value logA = math::LogOp::create(b, operandA);
  Value mult = arith::MulFOp::create(b, operandB, logA);
  Value expResult = math::ExpOp::create(b, mult);
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
  Value mult = arith::MulFOp::create(b, opType, operand, ln2);
  Value exp = math::ExpOp::create(b, op->getLoc(), mult);
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

  Value incrValue = math::CopySignOp::create(b, half, operand);
  Value add = arith::AddFOp::create(b, opType, operand, incrValue);
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
  Value operandBitcast = arith::BitcastOp::create(b, i32Ty, operand);
  Value operandExp = arith::AndIOp::create(
      b, arith::ShRUIOp::create(b, operandBitcast, c23), expMask);
  Value operandBiasedExp = arith::SubIOp::create(b, operandExp, c127);
  Value isSpecialValOrLargeVal = arith::CmpIOp::create(
      b, arith::CmpIPredicate::sge, operandBiasedExp, c23);

  Value result = arith::SelectOp::create(b, isSpecialValOrLargeVal, operand,
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

    Value pred = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ule,
                                       x, mask);
    Value add = arith::AddIOp::create(rewriter, loc, count, bits);
    Value shift = arith::ShLIOp::create(rewriter, loc, x, bits);

    x = arith::SelectOp::create(rewriter, loc, pred, shift, x);
    count = arith::SelectOp::create(rewriter, loc, pred, add, count);
  }

  Value zero = createIntConst(loc, operandTy, 0, rewriter);
  Value pred = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     operand, zero);

  Value bwval = createIntConst(loc, operandTy, bitwidth, rewriter);
  Value sel = arith::SelectOp::create(rewriter, loc, pred, bwval, count);
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

  Value operandBitcast = arith::BitcastOp::create(b, iTy, operand);
  Value round = math::RoundOp::create(b, operand);
  Value roundBitcast = arith::BitcastOp::create(b, iTy, round);

  // Get biased exponents for operand and round(operand)
  Value operandExp = arith::AndIOp::create(
      b, arith::ShRUIOp::create(b, operandBitcast, c23), expMask);
  Value operandBiasedExp = arith::SubIOp::create(b, operandExp, c127);
  Value roundExp = arith::AndIOp::create(
      b, arith::ShRUIOp::create(b, roundBitcast, c23), expMask);
  Value roundBiasedExp = arith::SubIOp::create(b, roundExp, c127);

  auto safeShiftRight = [&](Value x, Value shift) -> Value {
    // Clamp shift to valid range [0, bitwidth - 1] to avoid undefined behavior
    Value clampedShift = arith::MaxSIOp::create(b, shift, c0);
    clampedShift = arith::MinSIOp::create(b, clampedShift, c31);
    return arith::ShRUIOp::create(b, x, clampedShift);
  };

  auto maskMantissa = [&](Value mantissa,
                          Value mantissaMaskRightShift) -> Value {
    Value shiftedMantissaMask = safeShiftRight(c23Mask, mantissaMaskRightShift);
    return arith::AndIOp::create(b, mantissa, shiftedMantissaMask);
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
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, roundBiasedExp, c0);
  Value roundBiasedExpMinus1 = arith::SubIOp::create(b, roundBiasedExp, c1);
  Value roundMaskedMantissa = maskMantissa(roundBitcast, roundBiasedExpMinus1);
  Value roundIsNotEvenOrSpecialVal = arith::CmpIOp::create(
      b, arith::CmpIPredicate::ne, roundMaskedMantissa, c0);
  roundIsNotEvenOrSpecialVal =
      arith::OrIOp::create(b, roundIsNotEvenOrSpecialVal, roundBiasedExpEq0);

  // A value `x` with `0 <= biasedExp < 23`, is halfway between two consecutive
  // integers if the bit at index `biasedExp` starting from the left in the
  // mantissa is 1 and all the bits to the right are zero. Values with
  // `biasedExp >= 23` don't have decimals, so they are never halfway. The
  // values +-0.5 are the only halfway values that have `biasedExp == -1 < 0`,
  // so these are handled separately. In particular, if `biasedExp == -1`, the
  // value is halfway if the entire mantissa is zero.
  Value operandBiasedExpEqNeg1 = arith::CmpIOp::create(
      b, arith::CmpIPredicate::eq, operandBiasedExp, cNeg1);
  Value expectedOperandMaskedMantissa = arith::SelectOp::create(
      b, operandBiasedExpEqNeg1, c0, safeShiftRight(c2To22, operandBiasedExp));
  Value operandMaskedMantissa = maskMantissa(operandBitcast, operandBiasedExp);
  Value operandIsHalfway =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, operandMaskedMantissa,
                            expectedOperandMaskedMantissa);
  // Ensure `biasedExp` is in the valid range for half values.
  Value operandBiasedExpGeNeg1 = arith::CmpIOp::create(
      b, arith::CmpIPredicate::sge, operandBiasedExp, cNeg1);
  Value operandBiasedExpLt23 = arith::CmpIOp::create(
      b, arith::CmpIPredicate::slt, operandBiasedExp, c23);
  operandIsHalfway =
      arith::AndIOp::create(b, operandIsHalfway, operandBiasedExpLt23);
  operandIsHalfway =
      arith::AndIOp::create(b, operandIsHalfway, operandBiasedExpGeNeg1);

  // Adjust rounded operand with `round(operand) - sign(operand)` to correct the
  // case where `round` rounded in the opposite direction of `roundeven`.
  Value sign = math::CopySignOp::create(b, c1Float, operand);
  Value roundShifted = arith::SubFOp::create(b, round, sign);
  // If the rounded value is even or a special value, we default to the behavior
  // of `math.round`.
  Value needsShift =
      arith::AndIOp::create(b, roundIsNotEvenOrSpecialVal, operandIsHalfway);
  Value result = arith::SelectOp::create(b, needsShift, roundShifted, round);
  // The `x - sign` adjustment does not preserve the sign when we are adjusting
  // the value -1 to -0. So here the sign is copied again to ensure that -0.5 is
  // rounded to -0.0.
  result = math::CopySignOp::create(b, result, operand);
  rewriter.replaceOp(op, result);
  return success();
}

// Convert `math.rsqrt` into `arith.divf` + `math.sqrt`
static LogicalResult convertRsqrtOp(math::RsqrtOp op,
                                    PatternRewriter &rewriter) {

  auto operand = op.getOperand();
  auto operandTy = operand.getType();
  // Operand type must be shatic shaped type to create const float.
  auto shapedOperandType = dyn_cast<ShapedType>(operandTy);
  if (shapedOperandType && !shapedOperandType.hasStaticShape())
    return failure();

  auto eTy = getElementTypeOrSelf(operandTy);
  if (!isa<FloatType>(eTy))
    return failure();

  Location loc = op->getLoc();
  auto constOneFloat = createFloatConst(loc, operandTy, 1.0, rewriter);
  auto sqrtOp = math::SqrtOp::create(rewriter, loc, operand);
  rewriter.replaceOpWithNewOp<arith::DivFOp>(op, constOneFloat, sqrtOp);
  return success();
}

// Convert `math.clampf` into `arith.minimumf` + `arith.maximumf`
static LogicalResult convertClampfOp(math::ClampFOp op,
                                     PatternRewriter &rewriter) {
  auto minOp = arith::MinimumFOp::create(rewriter, op.getLoc(), op.getValue(),
                                         op.getMin(), op.getFastmath());
  rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, minOp, op.getMax(),
                                                 op.getFastmath());
  return success();
}

void mlir::math::populateExpansionPatterns(RewritePatternSet &patterns,
                                           ArrayRef<StringRef> opMnemonics) {
  auto filter = [&](StringRef name) {
    // This should be a static assert and `consume_front` take a twine, but none
    // is currently possible. TODO: augment `StringRef::consume_front` and make
    // `getDialectNamespace` use `std::string_view`.
    assert("math" == MathDialect::getDialectNamespace());
    name.consume_front("math.");
    return opMnemonics.empty() || (llvm::count(opMnemonics, name) > 0);
  };
  if (filter(CountLeadingZerosOp::getOperationName()))
    patterns.add(convertCtlzOp);
  if (filter(SinhOp::getOperationName()))
    patterns.add(convertSinhOp);
  if (filter(CoshOp::getOperationName()))
    patterns.add(convertCoshOp);
  if (filter(TanOp::getOperationName()))
    patterns.add(convertTanOp);
  if (filter(TanhOp::getOperationName()))
    patterns.add(convertTanhOp);
  if (filter(AsinhOp::getOperationName()))
    patterns.add(convertAsinhOp);
  if (filter(AcoshOp::getOperationName()))
    patterns.add(convertAcoshOp);
  if (filter(AtanhOp::getOperationName()))
    patterns.add(convertAtanhOp);
  if (filter(FmaOp::getOperationName()))
    patterns.add(convertFmaFOp);
  if (filter(CeilOp::getOperationName()))
    patterns.add(convertCeilOp);
  if (filter(Exp2Op::getOperationName()))
    patterns.add(convertExp2fOp);
  if (filter(PowFOp::getOperationName()))
    patterns.add(convertPowfOp);
  if (filter(FPowIOp::getOperationName()))
    patterns.add(convertFPowIOp);
  if (filter(RoundOp::getOperationName()))
    patterns.add(convertRoundOp);
  if (filter(RoundEvenOp::getOperationName()))
    patterns.add(convertRoundEvenOp);
  if (filter(RsqrtOp::getOperationName()))
    patterns.add(convertRsqrtOp);
  if (filter(ClampFOp::getOperationName()))
    patterns.add(convertClampfOp);
}

//===----------------------------------------------------------------------===//
// MathExpandOpsPass pass
//===----------------------------------------------------------------------===//
namespace {
struct MathExpandOpsPass final
    : math::impl::MathExpandOpsPassBase<MathExpandOpsPass> {
  using MathExpandOpsPassBase::MathExpandOpsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    SmallVector<StringRef> mnemonics =
        llvm::to_vector_of<StringRef>(opMnemonics);
    math::populateExpansionPatterns(patterns, mnemonics);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
