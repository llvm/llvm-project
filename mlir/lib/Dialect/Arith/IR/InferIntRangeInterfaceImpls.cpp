//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for arith -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::intrange;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRange) {
  auto constAttr = llvm::dyn_cast_or_null<IntegerAttr>(getValue());
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
  setResultRange(getResult(), inferAdd(argRanges));
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

void arith::SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferSub(argRanges));
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

void arith::MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMul(argRanges));
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

void arith::DivUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivU(argRanges));
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

void arith::DivSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivS(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivUIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferCeilDivU(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferCeilDivS(argRanges));
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

void arith::FloorDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  return setResultRange(getResult(), inferFloorDivS(argRanges));
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

void arith::RemUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemU(argRanges));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

void arith::RemSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemS(argRanges));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

void arith::AndIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAnd(argRanges));
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferOr(argRanges));
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

void arith::XOrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferXor(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

void arith::MaxSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxS(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

void arith::MaxUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxU(argRanges));
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

void arith::MinSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinS(argRanges));
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

void arith::MinUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinU(argRanges));
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

void arith::ExtUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extUIRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

void arith::ExtSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extSIRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

void arith::TruncIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRange) {
  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), truncRange(argRanges[0], destWidth));
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
    setResultRange(getResult(), extSIRange(argRanges[0], destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(argRanges[0], destWidth));
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
    setResultRange(getResult(), extUIRange(argRanges[0], destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(argRanges[0], destWidth));
  else
    setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

void arith::CmpIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  arith::CmpIPredicate arithPred = getPredicate();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(arithPred);
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  std::optional<bool> truthValue = intrange::evaluatePred(pred, lhs, rhs);
  if (truthValue.has_value() && *truthValue)
    min = max;
  else if (truthValue.has_value() && !(*truthValue))
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
  setResultRange(getResult(), inferShl(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

void arith::ShRUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrU(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

void arith::ShRSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrS(argRanges));
}
