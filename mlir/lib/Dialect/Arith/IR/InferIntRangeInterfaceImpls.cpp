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

#include <optional>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::intrange;

static intrange::OverflowFlags
convertArithOverflowFlags(arith::IntegerOverflowFlags flags) {
  intrange::OverflowFlags retFlags = intrange::OverflowFlags::None;
  if (bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nsw))
    retFlags |= intrange::OverflowFlags::Nsw;
  if (bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nuw))
    retFlags |= intrange::OverflowFlags::Nuw;
  return retFlags;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
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

void arith::AddIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer = inferFromOptionals([this](ArrayRef<ConstantIntRanges> ranges) {
    return inferAdd(ranges, convertArithOverflowFlags(getOverflowFlags()));
  });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

void arith::SubIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer = inferFromOptionals([this](ArrayRef<ConstantIntRanges> ranges) {
    return inferSub(ranges, convertArithOverflowFlags(getOverflowFlags()));
  });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

void arith::MulIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer = inferFromOptionals([this](ArrayRef<ConstantIntRanges> ranges) {
    return inferMul(ranges, convertArithOverflowFlags(getOverflowFlags()));
  });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

void arith::DivUIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferDivU)(argRanges));
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

void arith::DivSIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferDivS)(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivUIOp::inferResultRanges(
    ArrayRef<OptionalIntRanges> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferCeilDivU)(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivSIOp::inferResultRanges(
    ArrayRef<OptionalIntRanges> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferCeilDivS)(argRanges));
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

void arith::FloorDivSIOp::inferResultRanges(
    ArrayRef<OptionalIntRanges> argRanges, SetIntRangeFn setResultRange) {
  return setResultRange(getResult(),
                        inferFromOptionals(inferFloorDivS)(argRanges));
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

void arith::RemUIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferRemU)(argRanges));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

void arith::RemSIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferRemS)(argRanges));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

void arith::AndIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferAnd)(argRanges));
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferOr)(argRanges));
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

void arith::XOrIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferXor)(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

void arith::MaxSIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferMaxS)(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

void arith::MaxUIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferMaxU)(argRanges));
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

void arith::MinSIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferMinS)(argRanges));
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

void arith::MinUIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromOptionals(inferMinU)(argRanges));
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

void arith::ExtUIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  if (!argRanges[0])
    return;

  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extUIRange(*argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

void arith::ExtSIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  if (!argRanges[0])
    return;

  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extSIRange(*argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

void arith::TruncIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                        SetIntRangeFn setResultRange) {
  if (!argRanges[0])
    return;

  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), truncRange(*argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

void arith::IndexCastOp::inferResultRanges(
    ArrayRef<OptionalIntRanges> argRanges, SetIntRangeFn setResultRange) {
  if (!argRanges[0])
    return;

  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extSIRange(*argRanges[0], destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(*argRanges[0], destWidth));
  else
    setResultRange(getResult(), *argRanges[0]);
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

void arith::IndexCastUIOp::inferResultRanges(
    ArrayRef<OptionalIntRanges> argRanges, SetIntRangeFn setResultRange) {
  if (!argRanges[0])
    return;

  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extUIRange(*argRanges[0], destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(*argRanges[0], destWidth));
  else
    setResultRange(getResult(), *argRanges[0]);
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

void arith::CmpIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  arith::CmpIPredicate arithPred = getPredicate();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(arithPred);
  const OptionalIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  if (!lhs || !rhs)
    return;

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  std::optional<bool> truthValue = intrange::evaluatePred(pred, *lhs, *rhs);
  if (truthValue.has_value() && *truthValue)
    min = max;
  else if (truthValue.has_value() && !(*truthValue))
    max = min;

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void arith::SelectOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                        SetIntRangeFn setResultRange) {
  std::optional<APInt> mbCondVal =
      argRanges[0] ? argRanges[0]->getConstantValue() : std::nullopt;

  if (mbCondVal) {
    if (mbCondVal->isZero())
      setResultRange(getResult(), argRanges[2]);
    else
      setResultRange(getResult(), argRanges[1]);
    return;
  }

  if (argRanges[1] && argRanges[2])
    setResultRange(getResult(), argRanges[1]->rangeUnion(*argRanges[2]));
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

void arith::ShLIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer = inferFromOptionals([&](ArrayRef<ConstantIntRanges> ranges) {
    return inferShl(ranges, convertArithOverflowFlags(getOverflowFlags()));
  });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

void arith::ShRUIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  auto infer = inferFromOptionals(inferShrU);
  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

void arith::ShRSIOp::inferResultRanges(ArrayRef<OptionalIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  auto infer = inferFromOptionals(inferShrS);
  setResultRange(getResult(), infer(argRanges));
}
