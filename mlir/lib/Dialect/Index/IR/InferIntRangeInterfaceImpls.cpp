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

#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::index;
using namespace mlir::intrange;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

void ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
  const APInt &value = getValue();
  setResultRange(getResult(), ConstantIntRanges::constant(value));
}

void BoolConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
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
  return [inferWithOvfFn](ArrayRef<ConstantIntRanges> argRanges) {
    return inferWithOvfFn(argRanges, OverflowFlags::None);
  };
}

void AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferIndexOp(inferWithoutOverflowFlags(inferAdd),
                                           argRanges, CmpMode::Both));
}

void SubOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferIndexOp(inferWithoutOverflowFlags(inferSub),
                                           argRanges, CmpMode::Both));
}

void MulOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferIndexOp(inferWithoutOverflowFlags(inferMul),
                                           argRanges, CmpMode::Both));
}

void DivUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferDivU, argRanges, CmpMode::Unsigned));
}

void DivSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferDivS, argRanges, CmpMode::Signed));
}

void CeilDivUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferCeilDivU, argRanges, CmpMode::Unsigned));
}

void CeilDivSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferCeilDivS, argRanges, CmpMode::Signed));
}

void FloorDivSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRange) {
  return setResultRange(
      getResult(), inferIndexOp(inferFloorDivS, argRanges, CmpMode::Signed));
}

void RemSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferRemS, argRanges, CmpMode::Signed));
}

void RemUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferRemU, argRanges, CmpMode::Unsigned));
}

void MaxSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferMaxS, argRanges, CmpMode::Signed));
}

void MaxUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferMaxU, argRanges, CmpMode::Unsigned));
}

void MinSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferMinS, argRanges, CmpMode::Signed));
}

void MinUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferMinU, argRanges, CmpMode::Unsigned));
}

void ShlOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferIndexOp(inferWithoutOverflowFlags(inferShl),
                                           argRanges, CmpMode::Both));
}

void ShrSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferShrS, argRanges, CmpMode::Signed));
}

void ShrUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferShrU, argRanges, CmpMode::Unsigned));
}

void AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferAnd, argRanges, CmpMode::Unsigned));
}

void OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                             SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferOr, argRanges, CmpMode::Unsigned));
}

void XOrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferIndexOp(inferXor, argRanges, CmpMode::Unsigned));
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

void CastSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                SetIntRangeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  setResultRange(getResult(), inferIndexCast(argRanges[0], sourceType, destType,
                                             /*isSigned=*/true));
}

void CastUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                SetIntRangeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  setResultRange(getResult(), inferIndexCast(argRanges[0], sourceType, destType,
                                             /*isSigned=*/false));
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

void CmpOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  index::IndexCmpPredicate indexPred = getPred();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(indexPred);
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  std::optional<bool> truthValue64 = intrange::evaluatePred(pred, lhs, rhs);

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
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}

//===----------------------------------------------------------------------===//
// SizeOf, which is bounded between the two supported bitwidth (32 and 64).
//===----------------------------------------------------------------------===//

void SizeOfOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                 SetIntRangeFn setResultRange) {
  unsigned storageWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  APInt min(/*numBits=*/storageWidth, indexMinWidth);
  APInt max(/*numBits=*/storageWidth, indexMaxWidth);
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}
