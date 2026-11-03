//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for arith -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
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

static std::optional<unsigned>
getBitwidth(Type type, std::optional<unsigned> indexBitwidth) {
  return ConstantIntRanges::getBitwidth(type, indexBitwidth);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          std::optional<unsigned> indexBitwidth,
                                          SetIntRangeFn setResultRange) {
  std::optional<unsigned> width = getBitwidth(getType(), indexBitwidth);
  if (auto scalarCstAttr = llvm::dyn_cast_or_null<IntegerAttr>(getValue())) {
    const APInt &value = scalarCstAttr.getValue();
    setResultRange(getResult(), inferConstant(value, width));
    return;
  }
  if (auto arrayCstAttr =
          llvm::dyn_cast_or_null<DenseIntElementsAttr>(getValue())) {
    if (arrayCstAttr.empty())
      return;

    if (arrayCstAttr.isSplat()) {
      setResultRange(getResult(),
                     inferConstant(arrayCstAttr.getSplatValue<APInt>(), width));
      return;
    }

    std::optional<ConstantIntRanges> result;
    for (const APInt &val : arrayCstAttr) {
      auto range = inferConstant(val, width);
      result = (result ? result->rangeUnion(range) : range);
    }

    setResultRange(getResult(), *result);
    return;
  }

  auto range = IntegerValueRange::getMaxRange(getResult());
  setResultRange(getResult(), range.getValue());
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

void arith::AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      std::optional<unsigned> indexBitwidth,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferAdd(argRanges,
                          getBitwidth(getResult().getType(), indexBitwidth),
                          convertArithOverflowFlags(getOverflowFlags())));
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

void arith::SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      std::optional<unsigned> indexBitwidth,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferSub(argRanges,
                          getBitwidth(getResult().getType(), indexBitwidth),
                          convertArithOverflowFlags(getOverflowFlags())));
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

void arith::MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      std::optional<unsigned> indexBitwidth,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferMul(argRanges,
                          getBitwidth(getResult().getType(), indexBitwidth),
                          convertArithOverflowFlags(getOverflowFlags())));
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

void arith::DivUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      inferDivU(argRanges, getBitwidth(getResult().getType(), indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

void arith::DivSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      inferDivS(argRanges, getBitwidth(getResult().getType(), indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivUIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges,
    std::optional<unsigned> indexBitwidth, SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferCeilDivU(argRanges, getBitwidth(getResult().getType(),
                                                      indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges,
    std::optional<unsigned> indexBitwidth, SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferCeilDivS(argRanges, getBitwidth(getResult().getType(),
                                                      indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

void arith::FloorDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges,
    std::optional<unsigned> indexBitwidth, SetIntRangeFn setResultRange) {
  return setResultRange(
      getResult(), inferFloorDivS(argRanges, getBitwidth(getResult().getType(),
                                                         indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

void arith::RemUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      inferRemU(argRanges, getBitwidth(getResult().getType(), indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

void arith::RemSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      inferRemS(argRanges, getBitwidth(getResult().getType(), indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

void arith::AndIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      std::optional<unsigned> indexBitwidth,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAnd(argRanges));
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     std::optional<unsigned> indexBitwidth,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferOr(argRanges));
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

void arith::XOrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      std::optional<unsigned> indexBitwidth,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferXor(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

void arith::MaxSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxS(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

void arith::MaxUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxU(argRanges));
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

void arith::MinSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinS(argRanges));
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

void arith::MinUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinU(argRanges));
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

void arith::ExtUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  unsigned destWidth =
      cast<IntegerType>(getElementTypeOrSelf(getResult().getType())).getWidth();
  setResultRange(getResult(), extUIRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

void arith::ExtSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  unsigned destWidth =
      cast<IntegerType>(getElementTypeOrSelf(getResult().getType())).getWidth();
  setResultRange(getResult(), extSIRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

void arith::TruncIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        std::optional<unsigned> indexBitwidth,
                                        SetIntRangeFn setResultRange) {
  unsigned destWidth =
      cast<IntegerType>(getElementTypeOrSelf(getResult().getType())).getWidth();
  setResultRange(getResult(), truncRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

void arith::IndexCastOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> argRanges,
    std::optional<unsigned> indexBitwidth, SetIntLatticeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  setResultRange(getResult(),
                 inferIndexCast(argRanges[0],
                                getBitwidth(sourceType, indexBitwidth),
                                getBitwidth(destType, indexBitwidth),
                                /*isSigned=*/true));
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

void arith::IndexCastUIOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> argRanges,
    std::optional<unsigned> indexBitwidth, SetIntLatticeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  setResultRange(getResult(),
                 inferIndexCast(argRanges[0],
                                getBitwidth(sourceType, indexBitwidth),
                                getBitwidth(destType, indexBitwidth),
                                /*isSigned=*/false));
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

void arith::CmpIOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> argRanges,
    std::optional<unsigned> indexBitwidth, SetIntLatticeFn setResultRange) {
  arith::CmpIPredicate arithPred = getPredicate();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(arithPred);
  setResultRange(getResult(),
                 inferCmp(pred, argRanges,
                          getBitwidth(getLhs().getType(), indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void arith::SelectOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> argRanges,
    std::optional<unsigned> indexBitwidth, SetIntLatticeFn setResultRange) {
  std::optional<APInt> mbCondVal =
      argRanges[0].isUninitialized()
          ? std::nullopt
          : argRanges[0].getValue().getConstantValue();

  const IntegerValueRange &trueCase = argRanges[1];
  const IntegerValueRange &falseCase = argRanges[2];

  if (mbCondVal) {
    if (mbCondVal->isZero())
      setResultRange(getResult(), falseCase);
    else
      setResultRange(getResult(), trueCase);
    return;
  }

  // When one of the ranges is uninitialized, set the whole range to max
  // otherwise the result will ignore the uninitialized range.
  if (trueCase.isUninitialized() || falseCase.isUninitialized())
    setResultRange(getResult(),
                   IntegerValueRange::getMaxRange(
                       getResult(), indexBitwidth.value_or(
                                        IndexType::kInternalStorageBitWidth)));
  else
    setResultRange(getResult(), IntegerValueRange::join(trueCase, falseCase));
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

void arith::ShLIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      std::optional<unsigned> indexBitwidth,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 inferShl(argRanges,
                          getBitwidth(getResult().getType(), indexBitwidth),
                          convertArithOverflowFlags(getOverflowFlags())));
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

void arith::ShRUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      inferShrU(argRanges, getBitwidth(getResult().getType(), indexBitwidth)));
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

void arith::ShRSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      inferShrS(argRanges, getBitwidth(getResult().getType(), indexBitwidth)));
}
