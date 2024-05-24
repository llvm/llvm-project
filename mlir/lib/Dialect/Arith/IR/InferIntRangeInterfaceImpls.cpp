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

void arith::ConstantOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
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

void arith::AddIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([this](ArrayRef<ConstantIntRanges> ranges) {
        return inferAdd(ranges, convertArithOverflowFlags(getOverflowFlags()));
      });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

void arith::SubIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([this](ArrayRef<ConstantIntRanges> ranges) {
        return inferSub(ranges, convertArithOverflowFlags(getOverflowFlags()));
      });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

void arith::MulIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([this](ArrayRef<ConstantIntRanges> ranges) {
        return inferMul(ranges, convertArithOverflowFlags(getOverflowFlags()));
      });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

void arith::DivUIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferDivU)(argRanges));
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

void arith::DivSIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferDivS)(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivUIOp::inferResultRanges(
    ArrayRef<IntegerValueRange> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferFromIntegerValueRange(inferCeilDivU)(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivSIOp::inferResultRanges(
    ArrayRef<IntegerValueRange> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferFromIntegerValueRange(inferCeilDivS)(argRanges));
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

void arith::FloorDivSIOp::inferResultRanges(
    ArrayRef<IntegerValueRange> argRanges, SetIntRangeFn setResultRange) {
  return setResultRange(getResult(),
                        inferFromIntegerValueRange(inferFloorDivS)(argRanges));
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

void arith::RemUIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferRemU)(argRanges));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

void arith::RemSIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferRemS)(argRanges));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

void arith::AndIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferAnd)(argRanges));
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferOr)(argRanges));
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

void arith::XOrIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferXor)(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

void arith::MaxSIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferMaxS)(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

void arith::MaxUIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferMaxU)(argRanges));
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

void arith::MinSIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferMinS)(argRanges));
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

void arith::MinUIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferFromIntegerValueRange(inferMinU)(argRanges));
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

void arith::ExtUIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  if (argRanges[0].isUninitialized())
    return;

  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extUIRange(argRanges[0].getValue(), destWidth));
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

void arith::ExtSIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  if (argRanges[0].isUninitialized())
    return;

  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extSIRange(argRanges[0].getValue(), destWidth));
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

void arith::TruncIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                        SetIntRangeFn setResultRange) {
  if (argRanges[0].isUninitialized())
    return;

  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), truncRange(argRanges[0].getValue(), destWidth));
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

void arith::IndexCastOp::inferResultRanges(
    ArrayRef<IntegerValueRange> argRanges, SetIntRangeFn setResultRange) {
  if (argRanges[0].isUninitialized())
    return;

  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extSIRange(argRanges[0].getValue(), destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(argRanges[0].getValue(), destWidth));
  else
    setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

void arith::IndexCastUIOp::inferResultRanges(
    ArrayRef<IntegerValueRange> argRanges, SetIntRangeFn setResultRange) {
  if (argRanges[0].isUninitialized())
    return;

  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extUIRange(argRanges[0].getValue(), destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(argRanges[0].getValue(), destWidth));
  else
    setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

void arith::CmpIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                      SetIntRangeFn setResultRange) {
  arith::CmpIPredicate arithPred = getPredicate();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(arithPred);
  const IntegerValueRange &lhs = argRanges[0], &rhs = argRanges[1];

  if (lhs.isUninitialized() || rhs.isUninitialized())
    return;

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  std::optional<bool> truthValue =
      intrange::evaluatePred(pred, lhs.getValue(), rhs.getValue());
  if (truthValue.has_value() && *truthValue)
    min = max;
  else if (truthValue.has_value() && !(*truthValue))
    max = min;

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void arith::SelectOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                        SetIntRangeFn setResultRange) {
  std::optional<APInt> mbCondVal =
      !argRanges[0].isUninitialized()
          ? argRanges[0].getValue().getConstantValue()
          : std::nullopt;

  const IntegerValueRange &trueCase = argRanges[1];
  const IntegerValueRange &falseCase = argRanges[2];

  if (mbCondVal) {
    if (mbCondVal->isZero())
      setResultRange(getResult(), falseCase);
    else
      setResultRange(getResult(), trueCase);
    return;
  }

  setResultRange(getResult(), IntegerValueRange::join(trueCase, falseCase));
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

void arith::ShLIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                      SetIntRangeFn setResultRange) {
  auto infer =
      inferFromIntegerValueRange([&](ArrayRef<ConstantIntRanges> ranges) {
        return inferShl(ranges, convertArithOverflowFlags(getOverflowFlags()));
      });

  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

void arith::ShRUIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  auto infer = inferFromIntegerValueRange(inferShrU);
  setResultRange(getResult(), infer(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

void arith::ShRSIOp::inferResultRanges(ArrayRef<IntegerValueRange> argRanges,
                                       SetIntRangeFn setResultRange) {
  auto infer = inferFromIntegerValueRange(inferShrS);
  setResultRange(getResult(), infer(argRanges));
}
