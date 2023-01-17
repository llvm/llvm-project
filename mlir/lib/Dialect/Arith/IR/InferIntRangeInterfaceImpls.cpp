//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for arith -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::arith;

/// Function that evaluates the result of doing something on arithmetic
/// constants and returns std::nullopt on overflow.
using ConstArithFn =
    function_ref<std::optional<APInt>(const APInt &, const APInt &)>;

/// Return the maxmially wide signed or unsigned range for a given bitwidth.

/// Compute op(minLeft, minRight) and op(maxLeft, maxRight) if possible,
/// If either computation overflows, make the result unbounded.
static ConstantIntRanges computeBoundsBy(ConstArithFn op, const APInt &minLeft,
                                         const APInt &minRight,
                                         const APInt &maxLeft,
                                         const APInt &maxRight, bool isSigned) {
  std::optional<APInt> maybeMin = op(minLeft, minRight);
  std::optional<APInt> maybeMax = op(maxLeft, maxRight);
  if (maybeMin && maybeMax)
    return ConstantIntRanges::range(*maybeMin, *maybeMax, isSigned);
  return ConstantIntRanges::maxRange(minLeft.getBitWidth());
}

/// Compute the minimum and maximum of `(op(l, r) for l in lhs for r in rhs)`,
/// ignoring unbounded values. Returns the maximal range if `op` overflows.
static ConstantIntRanges minMaxBy(ConstArithFn op, ArrayRef<APInt> lhs,
                                  ArrayRef<APInt> rhs, bool isSigned) {
  unsigned width = lhs[0].getBitWidth();
  APInt min =
      isSigned ? APInt::getSignedMaxValue(width) : APInt::getMaxValue(width);
  APInt max =
      isSigned ? APInt::getSignedMinValue(width) : APInt::getZero(width);
  for (const APInt &left : lhs) {
    for (const APInt &right : rhs) {
      std::optional<APInt> maybeThisResult = op(left, right);
      if (!maybeThisResult)
        return ConstantIntRanges::maxRange(width);
      APInt result = std::move(*maybeThisResult);
      min = (isSigned ? result.slt(min) : result.ult(min)) ? result : min;
      max = (isSigned ? result.sgt(max) : result.ugt(max)) ? result : max;
    }
  }
  return ConstantIntRanges::range(min, max, isSigned);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRange) {
  auto constAttr = getValue().dyn_cast_or_null<IntegerAttr>();
  if (constAttr) {
    const APInt &value = constAttr.getValue();
    setResultRange(getResult(), ConstantIntRanges::constant(value));
  }
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

void arith::AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];
  ConstArithFn uadd = [](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = a.uadd_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstArithFn sadd = [](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = a.sadd_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };

  ConstantIntRanges urange = computeBoundsBy(
      uadd, lhs.umin(), rhs.umin(), lhs.umax(), rhs.umax(), /*isSigned=*/false);
  ConstantIntRanges srange = computeBoundsBy(
      sadd, lhs.smin(), rhs.smin(), lhs.smax(), rhs.smax(), /*isSigned=*/true);
  setResultRange(getResult(), urange.intersection(srange));
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

void arith::SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  ConstArithFn usub = [](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = a.usub_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstArithFn ssub = [](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = a.ssub_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstantIntRanges urange = computeBoundsBy(
      usub, lhs.umin(), rhs.umax(), lhs.umax(), rhs.umin(), /*isSigned=*/false);
  ConstantIntRanges srange = computeBoundsBy(
      ssub, lhs.smin(), rhs.smax(), lhs.smax(), rhs.smin(), /*isSigned=*/true);
  setResultRange(getResult(), urange.intersection(srange));
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

void arith::MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  ConstArithFn umul = [](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = a.umul_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstArithFn smul = [](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = a.smul_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };

  ConstantIntRanges urange =
      minMaxBy(umul, {lhs.umin(), lhs.umax()}, {rhs.umin(), rhs.umax()},
               /*isSigned=*/false);
  ConstantIntRanges srange =
      minMaxBy(smul, {lhs.smin(), lhs.smax()}, {rhs.smin(), rhs.smax()},
               /*isSigned=*/true);

  setResultRange(getResult(), urange.intersection(srange));
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

/// Fix up division results (ex. for ceiling and floor), returning an APInt
/// if there has been no overflow
using DivisionFixupFn = function_ref<std::optional<APInt>(
    const APInt &lhs, const APInt &rhs, const APInt &result)>;

static ConstantIntRanges inferDivUIRange(const ConstantIntRanges &lhs,
                                         const ConstantIntRanges &rhs,
                                         DivisionFixupFn fixup) {
  const APInt &lhsMin = lhs.umin(), &lhsMax = lhs.umax(), &rhsMin = rhs.umin(),
              &rhsMax = rhs.umax();

  if (!rhsMin.isZero()) {
    auto udiv = [&fixup](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
      return fixup(a, b, a.udiv(b));
    };
    return minMaxBy(udiv, {lhsMin, lhsMax}, {rhsMin, rhsMax},
                    /*isSigned=*/false);
  }
  // Otherwise, it's possible we might divide by 0.
  return ConstantIntRanges::maxRange(rhsMin.getBitWidth());
}

void arith::DivUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferDivUIRange(argRanges[0], argRanges[1],
                                 [](const APInt &lhs, const APInt &rhs,
                                    const APInt &result) { return result; }));
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

static ConstantIntRanges inferDivSIRange(const ConstantIntRanges &lhs,
                                         const ConstantIntRanges &rhs,
                                         DivisionFixupFn fixup) {
  const APInt &lhsMin = lhs.smin(), &lhsMax = lhs.smax(), &rhsMin = rhs.smin(),
              &rhsMax = rhs.smax();
  bool canDivide = rhsMin.isStrictlyPositive() || rhsMax.isNegative();

  if (canDivide) {
    auto sdiv = [&fixup](const APInt &a,
                         const APInt &b) -> std::optional<APInt> {
      bool overflowed = false;
      APInt result = a.sdiv_ov(b, overflowed);
      return overflowed ? std::optional<APInt>() : fixup(a, b, result);
    };
    return minMaxBy(sdiv, {lhsMin, lhsMax}, {rhsMin, rhsMax},
                    /*isSigned=*/true);
  }
  return ConstantIntRanges::maxRange(rhsMin.getBitWidth());
}

void arith::DivSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferDivSIRange(argRanges[0], argRanges[1],
                                 [](const APInt &lhs, const APInt &rhs,
                                    const APInt &result) { return result; }));
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivUIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  DivisionFixupFn ceilDivUIFix =
      [](const APInt &lhs, const APInt &rhs,
         const APInt &result) -> std::optional<APInt> {
    if (!lhs.urem(rhs).isZero()) {
      bool overflowed = false;
      APInt corrected =
          result.uadd_ov(APInt(result.getBitWidth(), 1), overflowed);
      return overflowed ? std::optional<APInt>() : corrected;
    }
    return result;
  };
  setResultRange(getResult(), inferDivUIRange(lhs, rhs, ceilDivUIFix));
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  DivisionFixupFn ceilDivSIFix =
      [](const APInt &lhs, const APInt &rhs,
         const APInt &result) -> std::optional<APInt> {
    if (!lhs.srem(rhs).isZero() && lhs.isNonNegative() == rhs.isNonNegative()) {
      bool overflowed = false;
      APInt corrected =
          result.sadd_ov(APInt(result.getBitWidth(), 1), overflowed);
      return overflowed ? std::optional<APInt>() : corrected;
    }
    return result;
  };
  setResultRange(getResult(), inferDivSIRange(lhs, rhs, ceilDivSIFix));
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

void arith::FloorDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  DivisionFixupFn floorDivSIFix =
      [](const APInt &lhs, const APInt &rhs,
         const APInt &result) -> std::optional<APInt> {
    if (!lhs.srem(rhs).isZero() && lhs.isNonNegative() != rhs.isNonNegative()) {
      bool overflowed = false;
      APInt corrected =
          result.ssub_ov(APInt(result.getBitWidth(), 1), overflowed);
      return overflowed ? std::optional<APInt>() : corrected;
    }
    return result;
  };
  setResultRange(getResult(), inferDivSIRange(lhs, rhs, floorDivSIFix));
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

void arith::RemUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];
  const APInt &rhsMin = rhs.umin(), &rhsMax = rhs.umax();

  unsigned width = rhsMin.getBitWidth();
  APInt umin = APInt::getZero(width);
  APInt umax = APInt::getMaxValue(width);

  if (!rhsMin.isZero()) {
    umax = rhsMax - 1;
    // Special case: sweeping out a contiguous range in N/[modulus]
    if (rhsMin == rhsMax) {
      const APInt &lhsMin = lhs.umin(), &lhsMax = lhs.umax();
      if ((lhsMax - lhsMin).ult(rhsMax)) {
        APInt minRem = lhsMin.urem(rhsMax);
        APInt maxRem = lhsMax.urem(rhsMax);
        if (minRem.ule(maxRem)) {
          umin = minRem;
          umax = maxRem;
        }
      }
    }
  }
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(umin, umax));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

void arith::RemSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];
  const APInt &lhsMin = lhs.smin(), &lhsMax = lhs.smax(), &rhsMin = rhs.smin(),
              &rhsMax = rhs.smax();

  unsigned width = rhsMax.getBitWidth();
  APInt smin = APInt::getSignedMinValue(width);
  APInt smax = APInt::getSignedMaxValue(width);
  // No bounds if zero could be a divisor.
  bool canBound = (rhsMin.isStrictlyPositive() || rhsMax.isNegative());
  if (canBound) {
    APInt maxDivisor = rhsMin.isStrictlyPositive() ? rhsMax : rhsMin.abs();
    bool canNegativeDividend = lhsMin.isNegative();
    bool canPositiveDividend = lhsMax.isStrictlyPositive();
    APInt zero = APInt::getZero(maxDivisor.getBitWidth());
    APInt maxPositiveResult = maxDivisor - 1;
    APInt minNegativeResult = -maxPositiveResult;
    smin = canNegativeDividend ? minNegativeResult : zero;
    smax = canPositiveDividend ? maxPositiveResult : zero;
    // Special case: sweeping out a contiguous range in N/[modulus].
    if (rhsMin == rhsMax) {
      if ((lhsMax - lhsMin).ult(maxDivisor)) {
        APInt minRem = lhsMin.srem(maxDivisor);
        APInt maxRem = lhsMax.srem(maxDivisor);
        if (minRem.sle(maxRem)) {
          smin = minRem;
          smax = maxRem;
        }
      }
    }
  }
  setResultRange(getResult(), ConstantIntRanges::fromSigned(smin, smax));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

/// "Widen" bounds - if 0bvvvvv??? <= a <= 0bvvvvv???,
/// relax the bounds to 0bvvvvv000 <= a <= 0bvvvvv111, where vvvvv are the bits
/// that both bonuds have in common. This gives us a consertive approximation
/// for what values can be passed to bitwise operations.
static std::tuple<APInt, APInt>
widenBitwiseBounds(const ConstantIntRanges &bound) {
  APInt leftVal = bound.umin(), rightVal = bound.umax();
  unsigned bitwidth = leftVal.getBitWidth();
  unsigned differingBits = bitwidth - (leftVal ^ rightVal).countLeadingZeros();
  leftVal.clearLowBits(differingBits);
  rightVal.setLowBits(differingBits);
  return std::make_tuple(std::move(leftVal), std::move(rightVal));
}

void arith::AndIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto [lhsZeros, lhsOnes] = widenBitwiseBounds(argRanges[0]);
  auto [rhsZeros, rhsOnes] = widenBitwiseBounds(argRanges[1]);
  auto andi = [](const APInt &a, const APInt &b) -> std::optional<APInt> {
    return a & b;
  };
  setResultRange(getResult(),
                 minMaxBy(andi, {lhsZeros, lhsOnes}, {rhsZeros, rhsOnes},
                          /*isSigned=*/false));
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  auto [lhsZeros, lhsOnes] = widenBitwiseBounds(argRanges[0]);
  auto [rhsZeros, rhsOnes] = widenBitwiseBounds(argRanges[1]);
  auto ori = [](const APInt &a, const APInt &b) -> std::optional<APInt> {
    return a | b;
  };
  setResultRange(getResult(),
                 minMaxBy(ori, {lhsZeros, lhsOnes}, {rhsZeros, rhsOnes},
                          /*isSigned=*/false));
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

void arith::XOrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto [lhsZeros, lhsOnes] = widenBitwiseBounds(argRanges[0]);
  auto [rhsZeros, rhsOnes] = widenBitwiseBounds(argRanges[1]);
  auto xori = [](const APInt &a, const APInt &b) -> std::optional<APInt> {
    return a ^ b;
  };
  setResultRange(getResult(),
                 minMaxBy(xori, {lhsZeros, lhsOnes}, {rhsZeros, rhsOnes},
                          /*isSigned=*/false));
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

void arith::MaxSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &smin = lhs.smin().sgt(rhs.smin()) ? lhs.smin() : rhs.smin();
  const APInt &smax = lhs.smax().sgt(rhs.smax()) ? lhs.smax() : rhs.smax();
  setResultRange(getResult(), ConstantIntRanges::fromSigned(smin, smax));
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

void arith::MaxUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &umin = lhs.umin().ugt(rhs.umin()) ? lhs.umin() : rhs.umin();
  const APInt &umax = lhs.umax().ugt(rhs.umax()) ? lhs.umax() : rhs.umax();
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(umin, umax));
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

void arith::MinSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &smin = lhs.smin().slt(rhs.smin()) ? lhs.smin() : rhs.smin();
  const APInt &smax = lhs.smax().slt(rhs.smax()) ? lhs.smax() : rhs.smax();
  setResultRange(getResult(), ConstantIntRanges::fromSigned(smin, smax));
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

void arith::MinUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &umin = lhs.umin().ult(rhs.umin()) ? lhs.umin() : rhs.umin();
  const APInt &umax = lhs.umax().ult(rhs.umax()) ? lhs.umax() : rhs.umax();
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(umin, umax));
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

static ConstantIntRanges extUIRange(const ConstantIntRanges &range,
                                    Type destType) {
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);
  APInt umin = range.umin().zext(destWidth);
  APInt umax = range.umax().zext(destWidth);
  return ConstantIntRanges::fromUnsigned(umin, umax);
}

void arith::ExtUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  Type destType = getResult().getType();
  setResultRange(getResult(), extUIRange(argRanges[0], destType));
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

static ConstantIntRanges extSIRange(const ConstantIntRanges &range,
                                    Type destType) {
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);
  APInt smin = range.smin().sext(destWidth);
  APInt smax = range.smax().sext(destWidth);
  return ConstantIntRanges::fromSigned(smin, smax);
}

void arith::ExtSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  Type destType = getResult().getType();
  setResultRange(getResult(), extSIRange(argRanges[0], destType));
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

static ConstantIntRanges truncIRange(const ConstantIntRanges &range,
                                     Type destType) {
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);
  // If you truncate the first four bytes in [0xaaaabbbb, 0xccccbbbb],
  // the range of the resulting value is not contiguous ind includes 0.
  // Ex. If you truncate [256, 258] from i16 to i8, you validly get [0, 2],
  // but you can't truncate [255, 257] similarly.
  bool hasUnsignedRollover =
      range.umin().lshr(destWidth) != range.umax().lshr(destWidth);
  APInt umin = hasUnsignedRollover ? APInt::getZero(destWidth)
                                   : range.umin().trunc(destWidth);
  APInt umax = hasUnsignedRollover ? APInt::getMaxValue(destWidth)
                                   : range.umax().trunc(destWidth);

  // Signed post-truncation rollover will not occur when either:
  // - The high parts of the min and max, plus the sign bit, are the same
  // - The high halves + sign bit of the min and max are either all 1s or all 0s
  //  and you won't create a [positive, negative] range by truncating.
  // For example, you can truncate the ranges [256, 258]_i16 to [0, 2]_i8
  // but not [255, 257]_i16 to a range of i8s. You can also truncate
  // [-256, -256]_i16 to [-2, 0]_i8, but not [-257, -255]_i16.
  // You can also truncate [-130, 0]_i16 to i8 because -130_i16 (0xff7e)
  // will truncate to 0x7e, which is greater than 0
  APInt sminHighPart = range.smin().ashr(destWidth - 1);
  APInt smaxHighPart = range.smax().ashr(destWidth - 1);
  bool hasSignedOverflow =
      (sminHighPart != smaxHighPart) &&
      !(sminHighPart.isAllOnes() &&
        (smaxHighPart.isAllOnes() || smaxHighPart.isZero())) &&
      !(sminHighPart.isZero() && smaxHighPart.isZero());
  APInt smin = hasSignedOverflow ? APInt::getSignedMinValue(destWidth)
                                 : range.smin().trunc(destWidth);
  APInt smax = hasSignedOverflow ? APInt::getSignedMaxValue(destWidth)
                                 : range.smax().trunc(destWidth);
  return {umin, umax, smin, smax};
}

void arith::TruncIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRange) {
  Type destType = getResult().getType();
  setResultRange(getResult(), truncIRange(argRanges[0], destType));
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

void arith::IndexCastOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extSIRange(argRanges[0], destType));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncIRange(argRanges[0], destType));
  else
    setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

void arith::IndexCastUIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extUIRange(argRanges[0], destType));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncIRange(argRanges[0], destType));
  else
    setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

bool isStaticallyTrue(arith::CmpIPredicate pred, const ConstantIntRanges &lhs,
                      const ConstantIntRanges &rhs) {
  switch (pred) {
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::slt:
    return (applyCmpPredicate(pred, lhs.smax(), rhs.smin()));
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ult:
    return applyCmpPredicate(pred, lhs.umax(), rhs.umin());
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::sgt:
    return applyCmpPredicate(pred, lhs.smin(), rhs.smax());
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ugt:
    return applyCmpPredicate(pred, lhs.umin(), rhs.umax());
  case arith::CmpIPredicate::eq: {
    std::optional<APInt> lhsConst = lhs.getConstantValue();
    std::optional<APInt> rhsConst = rhs.getConstantValue();
    return lhsConst && rhsConst && lhsConst == rhsConst;
  }
  case arith::CmpIPredicate::ne: {
    // While equality requires that there is an interpration of the preceeding
    // computations that produces equal constants, whether that be signed or
    // unsigned, statically determining inequality requires that neither
    // interpretation produce potentially overlapping ranges.
    bool sne = isStaticallyTrue(CmpIPredicate::slt, lhs, rhs) ||
               isStaticallyTrue(CmpIPredicate::sgt, lhs, rhs);
    bool une = isStaticallyTrue(CmpIPredicate::ult, lhs, rhs) ||
               isStaticallyTrue(CmpIPredicate::ugt, lhs, rhs);
    return sne && une;
  }
  }
  return false;
}

void arith::CmpIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  arith::CmpIPredicate pred = getPredicate();
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnesValue(1);
  if (isStaticallyTrue(pred, lhs, rhs))
    min = max;
  else if (isStaticallyTrue(invertPredicate(pred), lhs, rhs))
    max = min;

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void arith::SelectOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRange) {
  std::optional<APInt> mbCondVal = argRanges[0].getConstantValue();

  if (mbCondVal) {
    if (mbCondVal->isZero())
      setResultRange(getResult(), argRanges[2]);
    else
      setResultRange(getResult(), argRanges[1]);
    return;
  }
  setResultRange(getResult(), argRanges[1].rangeUnion(argRanges[2]));
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

void arith::ShLIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];
  ConstArithFn shl = [](const APInt &l,
                        const APInt &r) -> std::optional<APInt> {
    return r.uge(r.getBitWidth()) ? std::optional<APInt>() : l.shl(r);
  };
  ConstantIntRanges urange =
      minMaxBy(shl, {lhs.umin(), lhs.umax()}, {rhs.umin(), rhs.umax()},
               /*isSigned=*/false);
  ConstantIntRanges srange =
      minMaxBy(shl, {lhs.smin(), lhs.smax()}, {rhs.umin(), rhs.umax()},
               /*isSigned=*/true);
  setResultRange(getResult(), urange.intersection(srange));
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

void arith::ShRUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  ConstArithFn lshr = [](const APInt &l,
                         const APInt &r) -> std::optional<APInt> {
    return r.uge(r.getBitWidth()) ? std::optional<APInt>() : l.lshr(r);
  };
  setResultRange(getResult(), minMaxBy(lshr, {lhs.umin(), lhs.umax()},
                                       {rhs.umin(), rhs.umax()},
                                       /*isSigned=*/false));
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

void arith::ShRSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  ConstArithFn ashr = [](const APInt &l,
                         const APInt &r) -> std::optional<APInt> {
    return r.uge(r.getBitWidth()) ? std::optional<APInt>() : l.ashr(r);
  };

  setResultRange(getResult(),
                 minMaxBy(ashr, {lhs.smin(), lhs.smax()},
                          {rhs.umin(), rhs.umax()}, /*isSigned=*/true));
}
