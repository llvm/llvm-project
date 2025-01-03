//===- InferIntRangeCommon.cpp - Inference for common ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementations of range inference for operations that are
// common to both the `arith` and `index` dialects to facilitate reuse.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

#include "mlir/Interfaces/InferIntRangeInterface.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Debug.h"

#include <iterator>
#include <optional>

using namespace mlir;

#define DEBUG_TYPE "int-range-analysis"

//===----------------------------------------------------------------------===//
// General utilities
//===----------------------------------------------------------------------===//

/// Function that evaluates the result of doing something on arithmetic
/// constants and returns std::nullopt on overflow.
using ConstArithFn =
    function_ref<std::optional<APInt>(const APInt &, const APInt &)>;
using ConstArithStdFn =
    std::function<std::optional<APInt>(const APInt &, const APInt &)>;

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
// Ext, trunc, index op handling
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferIndexOp(const InferRangeFn &inferFn,
                             ArrayRef<ConstantIntRanges> argRanges,
                             intrange::CmpMode mode) {
  ConstantIntRanges sixtyFour = inferFn(argRanges);
  SmallVector<ConstantIntRanges, 2> truncated;
  llvm::transform(argRanges, std::back_inserter(truncated),
                  [](const ConstantIntRanges &range) {
                    return truncRange(range, /*destWidth=*/indexMinWidth);
                  });
  ConstantIntRanges thirtyTwo = inferFn(truncated);
  ConstantIntRanges thirtyTwoAsSixtyFour =
      extRange(thirtyTwo, /*destWidth=*/indexMaxWidth);
  ConstantIntRanges sixtyFourAsThirtyTwo =
      truncRange(sixtyFour, /*destWidth=*/indexMinWidth);

  LLVM_DEBUG(llvm::dbgs() << "Index handling: 64-bit result = " << sixtyFour
                          << " 32-bit = " << thirtyTwo << "\n");
  bool truncEqual = false;
  switch (mode) {
  case intrange::CmpMode::Both:
    truncEqual = (thirtyTwo == sixtyFourAsThirtyTwo);
    break;
  case intrange::CmpMode::Signed:
    truncEqual = (thirtyTwo.smin() == sixtyFourAsThirtyTwo.smin() &&
                  thirtyTwo.smax() == sixtyFourAsThirtyTwo.smax());
    break;
  case intrange::CmpMode::Unsigned:
    truncEqual = (thirtyTwo.umin() == sixtyFourAsThirtyTwo.umin() &&
                  thirtyTwo.umax() == sixtyFourAsThirtyTwo.umax());
    break;
  }
  if (truncEqual)
    // Returing the 64-bit result preserves more information.
    return sixtyFour;
  ConstantIntRanges merged = sixtyFour.rangeUnion(thirtyTwoAsSixtyFour);
  return merged;
}

ConstantIntRanges mlir::intrange::extRange(const ConstantIntRanges &range,
                                           unsigned int destWidth) {
  APInt umin = range.umin().zext(destWidth);
  APInt umax = range.umax().zext(destWidth);
  APInt smin = range.smin().sext(destWidth);
  APInt smax = range.smax().sext(destWidth);
  return {umin, umax, smin, smax};
}

ConstantIntRanges mlir::intrange::extUIRange(const ConstantIntRanges &range,
                                             unsigned destWidth) {
  APInt umin = range.umin().zext(destWidth);
  APInt umax = range.umax().zext(destWidth);
  return ConstantIntRanges::fromUnsigned(umin, umax);
}

ConstantIntRanges mlir::intrange::extSIRange(const ConstantIntRanges &range,
                                             unsigned destWidth) {
  APInt smin = range.smin().sext(destWidth);
  APInt smax = range.smax().sext(destWidth);
  return ConstantIntRanges::fromSigned(smin, smax);
}

ConstantIntRanges mlir::intrange::truncRange(const ConstantIntRanges &range,
                                             unsigned int destWidth) {
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

//===----------------------------------------------------------------------===//
// Addition
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferAdd(ArrayRef<ConstantIntRanges> argRanges,
                         OverflowFlags ovfFlags) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  ConstArithStdFn uadd = [=](const APInt &a,
                             const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nuw)
                       ? a.uadd_sat(b)
                       : a.uadd_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstArithStdFn sadd = [=](const APInt &a,
                             const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nsw)
                       ? a.sadd_sat(b)
                       : a.sadd_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };

  ConstantIntRanges urange = computeBoundsBy(
      uadd, lhs.umin(), rhs.umin(), lhs.umax(), rhs.umax(), /*isSigned=*/false);
  ConstantIntRanges srange = computeBoundsBy(
      sadd, lhs.smin(), rhs.smin(), lhs.smax(), rhs.smax(), /*isSigned=*/true);
  return urange.intersection(srange);
}

//===----------------------------------------------------------------------===//
// Subtraction
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferSub(ArrayRef<ConstantIntRanges> argRanges,
                         OverflowFlags ovfFlags) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  ConstArithStdFn usub = [=](const APInt &a,
                             const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nuw)
                       ? a.usub_sat(b)
                       : a.usub_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstArithStdFn ssub = [=](const APInt &a,
                             const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nsw)
                       ? a.ssub_sat(b)
                       : a.ssub_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstantIntRanges urange = computeBoundsBy(
      usub, lhs.umin(), rhs.umax(), lhs.umax(), rhs.umin(), /*isSigned=*/false);
  ConstantIntRanges srange = computeBoundsBy(
      ssub, lhs.smin(), rhs.smax(), lhs.smax(), rhs.smin(), /*isSigned=*/true);
  return urange.intersection(srange);
}

//===----------------------------------------------------------------------===//
// Multiplication
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferMul(ArrayRef<ConstantIntRanges> argRanges,
                         OverflowFlags ovfFlags) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  ConstArithStdFn umul = [=](const APInt &a,
                             const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nuw)
                       ? a.umul_sat(b)
                       : a.umul_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstArithStdFn smul = [=](const APInt &a,
                             const APInt &b) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nsw)
                       ? a.smul_sat(b)
                       : a.smul_ov(b, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };

  ConstantIntRanges urange =
      minMaxBy(umul, {lhs.umin(), lhs.umax()}, {rhs.umin(), rhs.umax()},
               /*isSigned=*/false);
  ConstantIntRanges srange =
      minMaxBy(smul, {lhs.smin(), lhs.smax()}, {rhs.smin(), rhs.smax()},
               /*isSigned=*/true);
  return urange.intersection(srange);
}

//===----------------------------------------------------------------------===//
// DivU, CeilDivU (Unsigned division)
//===----------------------------------------------------------------------===//

/// Fix up division results (ex. for ceiling and floor), returning an APInt
/// if there has been no overflow
using DivisionFixupFn = function_ref<std::optional<APInt>(
    const APInt &lhs, const APInt &rhs, const APInt &result)>;

static ConstantIntRanges inferDivURange(const ConstantIntRanges &lhs,
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

  APInt umin = APInt::getZero(rhsMin.getBitWidth());
  if (lhsMin.uge(rhsMax) && !rhsMax.isZero())
    umin = lhsMin.udiv(rhsMax);

  // X u/ Y u<= X.
  APInt umax = lhsMax;
  return ConstantIntRanges::fromUnsigned(umin, umax);
}

ConstantIntRanges
mlir::intrange::inferDivU(ArrayRef<ConstantIntRanges> argRanges) {
  return inferDivURange(argRanges[0], argRanges[1],
                        [](const APInt &lhs, const APInt &rhs,
                           const APInt &result) { return result; });
}

ConstantIntRanges
mlir::intrange::inferCeilDivU(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  auto ceilDivUIFix = [](const APInt &lhs, const APInt &rhs,
                         const APInt &result) -> std::optional<APInt> {
    if (!lhs.urem(rhs).isZero()) {
      bool overflowed = false;
      APInt corrected =
          result.uadd_ov(APInt(result.getBitWidth(), 1), overflowed);
      return overflowed ? std::optional<APInt>() : corrected;
    }
    return result;
  };
  return inferDivURange(lhs, rhs, ceilDivUIFix);
}

//===----------------------------------------------------------------------===//
// DivS, CeilDivS, FloorDivS (Signed division)
//===----------------------------------------------------------------------===//

static ConstantIntRanges inferDivSRange(const ConstantIntRanges &lhs,
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

ConstantIntRanges
mlir::intrange::inferDivS(ArrayRef<ConstantIntRanges> argRanges) {
  return inferDivSRange(argRanges[0], argRanges[1],
                        [](const APInt &lhs, const APInt &rhs,
                           const APInt &result) { return result; });
}

ConstantIntRanges
mlir::intrange::inferCeilDivS(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  auto ceilDivSIFix = [](const APInt &lhs, const APInt &rhs,
                         const APInt &result) -> std::optional<APInt> {
    if (!lhs.srem(rhs).isZero() && lhs.isNonNegative() == rhs.isNonNegative()) {
      bool overflowed = false;
      APInt corrected =
          result.sadd_ov(APInt(result.getBitWidth(), 1), overflowed);
      return overflowed ? std::optional<APInt>() : corrected;
    }
    // Special case where the usual implementation of ceilDiv causes
    // INT_MIN / [positive number] to be positive. This doesn't match the
    // definition of signed ceiling division mathematically, but it prevents
    // inconsistent constant-folding results. This arises because (-int_min) is
    // still negative, so -(-int_min / b) is -(int_min / b), which is
    // positive See #115293.
    if (lhs.isMinSignedValue() && rhs.sgt(1)) {
      return -result;
    }
    return result;
  };
  ConstantIntRanges result = inferDivSRange(lhs, rhs, ceilDivSIFix);
  if (lhs.smin().isMinSignedValue() && lhs.smax().sgt(lhs.smin())) {
    // If lhs range includes INT_MIN and lhs is not a single value, we can
    // suddenly wrap to positive val, skipping entire negative range, add
    // [INT_MIN + 1, smax()] range to the result to handle this.
    auto newLhs = ConstantIntRanges::fromSigned(lhs.smin() + 1, lhs.smax());
    result = result.rangeUnion(inferDivSRange(newLhs, rhs, ceilDivSIFix));
  }
  return result;
}

ConstantIntRanges
mlir::intrange::inferFloorDivS(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  auto floorDivSIFix = [](const APInt &lhs, const APInt &rhs,
                          const APInt &result) -> std::optional<APInt> {
    if (!lhs.srem(rhs).isZero() && lhs.isNonNegative() != rhs.isNonNegative()) {
      bool overflowed = false;
      APInt corrected =
          result.ssub_ov(APInt(result.getBitWidth(), 1), overflowed);
      return overflowed ? std::optional<APInt>() : corrected;
    }
    return result;
  };
  return inferDivSRange(lhs, rhs, floorDivSIFix);
}

//===----------------------------------------------------------------------===//
// Signed remainder (RemS)
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferRemS(ArrayRef<ConstantIntRanges> argRanges) {
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
  return ConstantIntRanges::fromSigned(smin, smax);
}

//===----------------------------------------------------------------------===//
// Unsigned remainder (RemU)
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferRemU(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];
  const APInt &rhsMin = rhs.umin(), &rhsMax = rhs.umax();

  unsigned width = rhsMin.getBitWidth();
  APInt umin = APInt::getZero(width);
  // Remainder can't be larger than either of its arguments.
  APInt umax = llvm::APIntOps::umin((rhsMax - 1), lhs.umax());

  if (!rhsMin.isZero()) {
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
  return ConstantIntRanges::fromUnsigned(umin, umax);
}

//===----------------------------------------------------------------------===//
// Max and min (MaxS, MaxU, MinS, MinU)
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferMaxS(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &smin = lhs.smin().sgt(rhs.smin()) ? lhs.smin() : rhs.smin();
  const APInt &smax = lhs.smax().sgt(rhs.smax()) ? lhs.smax() : rhs.smax();
  return ConstantIntRanges::fromSigned(smin, smax);
}

ConstantIntRanges
mlir::intrange::inferMaxU(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &umin = lhs.umin().ugt(rhs.umin()) ? lhs.umin() : rhs.umin();
  const APInt &umax = lhs.umax().ugt(rhs.umax()) ? lhs.umax() : rhs.umax();
  return ConstantIntRanges::fromUnsigned(umin, umax);
}

ConstantIntRanges
mlir::intrange::inferMinS(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &smin = lhs.smin().slt(rhs.smin()) ? lhs.smin() : rhs.smin();
  const APInt &smax = lhs.smax().slt(rhs.smax()) ? lhs.smax() : rhs.smax();
  return ConstantIntRanges::fromSigned(smin, smax);
}

ConstantIntRanges
mlir::intrange::inferMinU(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  const APInt &umin = lhs.umin().ult(rhs.umin()) ? lhs.umin() : rhs.umin();
  const APInt &umax = lhs.umax().ult(rhs.umax()) ? lhs.umax() : rhs.umax();
  return ConstantIntRanges::fromUnsigned(umin, umax);
}

//===----------------------------------------------------------------------===//
// Bitwise operators (And, Or, Xor)
//===----------------------------------------------------------------------===//

/// "Widen" bounds - if 0bvvvvv??? <= a <= 0bvvvvv???,
/// relax the bounds to 0bvvvvv000 <= a <= 0bvvvvv111, where vvvvv are the bits
/// that both bonuds have in common. This gives us a consertive approximation
/// for what values can be passed to bitwise operations.
static std::tuple<APInt, APInt>
widenBitwiseBounds(const ConstantIntRanges &bound) {
  APInt leftVal = bound.umin(), rightVal = bound.umax();
  unsigned bitwidth = leftVal.getBitWidth();
  unsigned differingBits = bitwidth - (leftVal ^ rightVal).countl_zero();
  leftVal.clearLowBits(differingBits);
  rightVal.setLowBits(differingBits);
  return std::make_tuple(std::move(leftVal), std::move(rightVal));
}

ConstantIntRanges
mlir::intrange::inferAnd(ArrayRef<ConstantIntRanges> argRanges) {
  auto [lhsZeros, lhsOnes] = widenBitwiseBounds(argRanges[0]);
  auto [rhsZeros, rhsOnes] = widenBitwiseBounds(argRanges[1]);
  auto andi = [](const APInt &a, const APInt &b) -> std::optional<APInt> {
    return a & b;
  };
  return minMaxBy(andi, {lhsZeros, lhsOnes}, {rhsZeros, rhsOnes},
                  /*isSigned=*/false);
}

ConstantIntRanges
mlir::intrange::inferOr(ArrayRef<ConstantIntRanges> argRanges) {
  auto [lhsZeros, lhsOnes] = widenBitwiseBounds(argRanges[0]);
  auto [rhsZeros, rhsOnes] = widenBitwiseBounds(argRanges[1]);
  auto ori = [](const APInt &a, const APInt &b) -> std::optional<APInt> {
    return a | b;
  };
  return minMaxBy(ori, {lhsZeros, lhsOnes}, {rhsZeros, rhsOnes},
                  /*isSigned=*/false);
}

/// Get bitmask of all bits which can change while iterating in
/// [bound.umin(), bound.umax()].
static APInt getVaryingBitsMask(const ConstantIntRanges &bound) {
  APInt leftVal = bound.umin(), rightVal = bound.umax();
  unsigned bitwidth = leftVal.getBitWidth();
  unsigned differingBits = bitwidth - (leftVal ^ rightVal).countl_zero();
  return APInt::getLowBitsSet(bitwidth, differingBits);
}

ConstantIntRanges
mlir::intrange::inferXor(ArrayRef<ConstantIntRanges> argRanges) {
  // Construct mask of varying bits for both ranges, xor values and then replace
  // masked bits with 0s and 1s to get min and max values respectively.
  ConstantIntRanges lhs = argRanges[0], rhs = argRanges[1];
  APInt mask = getVaryingBitsMask(lhs) | getVaryingBitsMask(rhs);
  APInt res = lhs.umin() ^ rhs.umin();
  APInt min = res & ~mask;
  APInt max = res | mask;
  return ConstantIntRanges::fromUnsigned(min, max);
}

//===----------------------------------------------------------------------===//
// Shifts (Shl, ShrS, ShrU)
//===----------------------------------------------------------------------===//

ConstantIntRanges
mlir::intrange::inferShl(ArrayRef<ConstantIntRanges> argRanges,
                         OverflowFlags ovfFlags) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];
  const APInt &rhsUMin = rhs.umin(), &rhsUMax = rhs.umax();

  // The signed/unsigned overflow behavior of shl by `rhs` matches a mul with
  // 2^rhs.
  ConstArithStdFn ushl = [=](const APInt &l,
                             const APInt &r) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nuw)
                       ? l.ushl_sat(r)
                       : l.ushl_ov(r, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };
  ConstArithStdFn sshl = [=](const APInt &l,
                             const APInt &r) -> std::optional<APInt> {
    bool overflowed = false;
    APInt result = any(ovfFlags & OverflowFlags::Nsw)
                       ? l.sshl_sat(r)
                       : l.sshl_ov(r, overflowed);
    return overflowed ? std::optional<APInt>() : result;
  };

  ConstantIntRanges urange =
      minMaxBy(ushl, {lhs.umin(), lhs.umax()}, {rhsUMin, rhsUMax},
               /*isSigned=*/false);
  ConstantIntRanges srange =
      minMaxBy(sshl, {lhs.smin(), lhs.smax()}, {rhsUMin, rhsUMax},
               /*isSigned=*/true);
  return urange.intersection(srange);
}

ConstantIntRanges
mlir::intrange::inferShrS(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  auto ashr = [](const APInt &l, const APInt &r) -> std::optional<APInt> {
    return r.uge(r.getBitWidth()) ? std::optional<APInt>() : l.ashr(r);
  };

  return minMaxBy(ashr, {lhs.smin(), lhs.smax()}, {rhs.umin(), rhs.umax()},
                  /*isSigned=*/true);
}

ConstantIntRanges
mlir::intrange::inferShrU(ArrayRef<ConstantIntRanges> argRanges) {
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  auto lshr = [](const APInt &l, const APInt &r) -> std::optional<APInt> {
    return r.uge(r.getBitWidth()) ? std::optional<APInt>() : l.lshr(r);
  };
  return minMaxBy(lshr, {lhs.umin(), lhs.umax()}, {rhs.umin(), rhs.umax()},
                  /*isSigned=*/false);
}

//===----------------------------------------------------------------------===//
// Comparisons (Cmp)
//===----------------------------------------------------------------------===//

static intrange::CmpPredicate invertPredicate(intrange::CmpPredicate pred) {
  switch (pred) {
  case intrange::CmpPredicate::eq:
    return intrange::CmpPredicate::ne;
  case intrange::CmpPredicate::ne:
    return intrange::CmpPredicate::eq;
  case intrange::CmpPredicate::slt:
    return intrange::CmpPredicate::sge;
  case intrange::CmpPredicate::sle:
    return intrange::CmpPredicate::sgt;
  case intrange::CmpPredicate::sgt:
    return intrange::CmpPredicate::sle;
  case intrange::CmpPredicate::sge:
    return intrange::CmpPredicate::slt;
  case intrange::CmpPredicate::ult:
    return intrange::CmpPredicate::uge;
  case intrange::CmpPredicate::ule:
    return intrange::CmpPredicate::ugt;
  case intrange::CmpPredicate::ugt:
    return intrange::CmpPredicate::ule;
  case intrange::CmpPredicate::uge:
    return intrange::CmpPredicate::ult;
  }
  llvm_unreachable("unknown cmp predicate value");
}

static bool isStaticallyTrue(intrange::CmpPredicate pred,
                             const ConstantIntRanges &lhs,
                             const ConstantIntRanges &rhs) {
  switch (pred) {
  case intrange::CmpPredicate::sle:
    return lhs.smax().sle(rhs.smin());
  case intrange::CmpPredicate::slt:
    return lhs.smax().slt(rhs.smin());
  case intrange::CmpPredicate::ule:
    return lhs.umax().ule(rhs.umin());
  case intrange::CmpPredicate::ult:
    return lhs.umax().ult(rhs.umin());
  case intrange::CmpPredicate::sge:
    return lhs.smin().sge(rhs.smax());
  case intrange::CmpPredicate::sgt:
    return lhs.smin().sgt(rhs.smax());
  case intrange::CmpPredicate::uge:
    return lhs.umin().uge(rhs.umax());
  case intrange::CmpPredicate::ugt:
    return lhs.umin().ugt(rhs.umax());
  case intrange::CmpPredicate::eq: {
    std::optional<APInt> lhsConst = lhs.getConstantValue();
    std::optional<APInt> rhsConst = rhs.getConstantValue();
    return lhsConst && rhsConst && lhsConst == rhsConst;
  }
  case intrange::CmpPredicate::ne: {
    // While equality requires that there is an interpration of the preceeding
    // computations that produces equal constants, whether that be signed or
    // unsigned, statically determining inequality requires that neither
    // interpretation produce potentially overlapping ranges.
    bool sne = isStaticallyTrue(intrange::CmpPredicate::slt, lhs, rhs) ||
               isStaticallyTrue(intrange::CmpPredicate::sgt, lhs, rhs);
    bool une = isStaticallyTrue(intrange::CmpPredicate::ult, lhs, rhs) ||
               isStaticallyTrue(intrange::CmpPredicate::ugt, lhs, rhs);
    return sne && une;
  }
  }
  return false;
}

std::optional<bool> mlir::intrange::evaluatePred(CmpPredicate pred,
                                                 const ConstantIntRanges &lhs,
                                                 const ConstantIntRanges &rhs) {
  if (isStaticallyTrue(pred, lhs, rhs))
    return true;
  if (isStaticallyTrue(invertPredicate(pred), lhs, rhs))
    return false;
  return std::nullopt;
}
