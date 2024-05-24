//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for arith -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

#include <optional>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::index;
using namespace mlir::intrange;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

void ConstantOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                   SetIntRangeFn setResultRange) {
  const APInt &value = getValue();
  setResultRange(getResult(), ConstantIntRanges::constant(value));
}

void BoolConstantOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  bool value = getValue();
  APInt asInt(/*numBits=*/1, value);
  setResultRange(getResult(), ConstantIntRanges::constant(asInt));
}

//===----------------------------------------------------------------------===//
// Arithmec operations. All of these operations will have their results inferred
// using both the 64-bit values and truncated 32-bit values of their inputs,
// with the results being the union of those inferences, except where the
// truncation of the 64-bit result is equal to the 32-bit result (at which time
// we take the 64-bit result).
//===----------------------------------------------------------------------===//

// Some arithmetic inference functions allow specifying special overflow / wrap
// behavior. We do not require this for the IndexOps and use this helper to call
// the inference function without any `OverflowFlags`.
static std::function<ConstantIntRanges(ArrayRef<ConstantIntRanges>)>
inferWithoutOverflowFlags(InferRangeWithOvfFlagsFn inferWithOvfFn) {
  return [inferWithOvfFn](
             ArrayRef<ConstantIntRanges> argRanges) -> ConstantIntRanges {
    return inferWithOvfFn(argRanges, OverflowFlags::None);
  };
}

void AddOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                              SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferWithoutOverflowFlags(inferAdd), ranges,
                            CmpMode::Both);
      });

  setResultRange(getResult(), infer(argRanges));
}

void SubOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                              SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferWithoutOverflowFlags(inferSub), ranges,
                            CmpMode::Both);
      });

  setResultRange(getResult(), infer(argRanges));
}

void MulOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                              SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferWithoutOverflowFlags(inferMul), ranges,
                            CmpMode::Both);
      });

  setResultRange(getResult(), infer(argRanges));
}

void DivUOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferWithoutOverflowFlags(inferSub), ranges,
                            CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void DivSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferDivS, ranges, CmpMode::Signed);
      });

  setResultRange(getResult(), infer(argRanges));
}

void CeilDivUOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                   SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferCeilDivU, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void CeilDivSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                   SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferCeilDivS, ranges, CmpMode::Signed);
      });

  setResultRange(getResult(), infer(argRanges));
}

void FloorDivSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                    SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferFloorDivS, ranges, CmpMode::Signed);
      });

  return setResultRange(getResult(), infer(argRanges));
}

void RemSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferRemS, ranges, CmpMode::Signed);
      });

  setResultRange(getResult(), infer(argRanges));
}

void RemUOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferRemU, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void MaxSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferMaxS, ranges, CmpMode::Signed);
      });

  setResultRange(getResult(), infer(argRanges));
}

void MaxUOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferMaxU, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void MinSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferMinS, ranges, CmpMode::Signed);
      });

  setResultRange(getResult(), infer(argRanges));
}

void MinUOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferMinU, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void ShlOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                              SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferWithoutOverflowFlags(inferShl), ranges,
                            CmpMode::Both);
      });

  setResultRange(getResult(), infer(argRanges));
}

void ShrSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferShrS, ranges, CmpMode::Signed);
      });

  setResultRange(getResult(), infer(argRanges));
}

void ShrUOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                               SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferShrU, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void AndOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                              SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferAnd, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void OrOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                             SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferOr, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

void XOrOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                              SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([](ArrayRef<ConstantIntRanges> ranges) {
        return inferIndexOp(inferXor, ranges, CmpMode::Unsigned);
      });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// Casts
//===----------------------------------------------------------------------===//

static ConstantIntRanges makeLikeDest(const ConstantIntRanges &range,
                                      unsigned srcWidth, unsigned destWidth,
                                      bool isSigned) {
  if (srcWidth < destWidth)
    return isSigned ? extSIRange(range, destWidth)
                    : extUIRange(range, destWidth);
  if (srcWidth > destWidth)
    return truncRange(range, destWidth);
  return range;
}

// When casting to `index`, we will take the union of the possible fixed-width
// casts.
static ConstantIntRanges inferIndexCast(const ConstantIntRanges &range,
                                        Type sourceType, Type destType,
                                        bool isSigned) {
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);
  if (sourceType.isIndex())
    return makeLikeDest(range, srcWidth, destWidth, isSigned);
  // We are casting to indexs, so use the union of the 32-bit and 64-bit casts
  ConstantIntRanges storageRange =
      makeLikeDest(range, srcWidth, destWidth, isSigned);
  ConstantIntRanges minWidthRange =
      makeLikeDest(range, srcWidth, indexMinWidth, isSigned);
  ConstantIntRanges minWidthExt = extRange(minWidthRange, destWidth);
  ConstantIntRanges ret = storageRange.rangeUnion(minWidthExt);
  return ret;
}

void CastSOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([this](ArrayRef<ConstantIntRanges> ranges) {
        Type sourceType = getOperand().getType();
        Type destType = getResult().getType();

        return inferIndexCast(ranges[0], sourceType, destType,
                              /*isSigned=*/true);
      });

  setResultRange(getResult(), infer(argRanges));
}

void CastUOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([this](ArrayRef<ConstantIntRanges> ranges) {
        Type sourceType = getOperand().getType();
        Type destType = getResult().getType();

        return inferIndexCast(ranges[0], sourceType, destType,
                              /*isSigned=*/false);
      });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

void CmpOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                              SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([this](ArrayRef<ConstantIntRanges> ranges) {
        index::IndexCmpPredicate indexPred = getPred();
        intrange::CmpPredicate pred =
            static_cast<intrange::CmpPredicate>(indexPred);
        const ConstantIntRanges &lhs = ranges[0], &rhs = ranges[1];

        APInt min = APInt::getZero(1);
        APInt max = APInt::getAllOnes(1);

        std::optional<bool> truthValue64 =
            intrange::evaluatePred(pred, lhs, rhs);

        ConstantIntRanges lhsTrunc = truncRange(lhs, indexMinWidth),
                          rhsTrunc = truncRange(rhs, indexMinWidth);
        std::optional<bool> truthValue32 =
            intrange::evaluatePred(pred, lhsTrunc, rhsTrunc);

        if (truthValue64 == truthValue32) {
          if (truthValue64.has_value() && *truthValue64)
            min = max;
          else if (truthValue64.has_value() && !(*truthValue64))
            max = min;
        }

        return ConstantIntRanges::fromUnsigned(min, max);
      });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// SizeOf, which is bounded between the two supported bitwidth (32 and 64).
//===----------------------------------------------------------------------===//

void SizeOfOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                 SetIntRangeFn setResultRange) {
  unsigned storageWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  APInt min(/*numBits=*/storageWidth, indexMinWidth);
  APInt max(/*numBits=*/storageWidth, indexMaxWidth);
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}
