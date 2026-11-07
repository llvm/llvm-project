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

static std::optional<unsigned>
getBitwidth(Type type, std::optional<unsigned> indexBitwidth) {
  return ConstantIntRanges::getBitwidth(type, indexBitwidth);
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

void ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   std::optional<unsigned> indexBitwidth,
                                   SetIntRangeFn setResultRange) {
  const APInt &value = getValue();
  setResultRange(getResult(), inferConstant(value, indexBitwidth));
}

void BoolConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       std::optional<unsigned> indexBitwidth,
                                       SetIntRangeFn setResultRange) {
  bool value = getValue();
  APInt asInt(/*numBits=*/1, value);
  setResultRange(getResult(), ConstantIntRanges::constant(asInt));
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

void AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              std::optional<unsigned> indexBitwidth,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAdd(argRanges, indexBitwidth));
}

void SubOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              std::optional<unsigned> indexBitwidth,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferSub(argRanges, indexBitwidth));
}

void MulOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              std::optional<unsigned> indexBitwidth,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMul(argRanges, indexBitwidth));
}

void DivUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivU(argRanges, indexBitwidth));
}

void DivSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivS(argRanges, indexBitwidth));
}

void CeilDivUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   std::optional<unsigned> indexBitwidth,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferCeilDivU(argRanges, indexBitwidth));
}

void CeilDivSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   std::optional<unsigned> indexBitwidth,
                                   SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferCeilDivS(argRanges, indexBitwidth));
}

void FloorDivSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    std::optional<unsigned> indexBitwidth,
                                    SetIntRangeFn setResultRange) {
  return setResultRange(getResult(), inferFloorDivS(argRanges, indexBitwidth));
}

void RemSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemS(argRanges, indexBitwidth));
}

void RemUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemU(argRanges, indexBitwidth));
}

void MaxSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxS(argRanges));
}

void MaxUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxU(argRanges));
}

void MinSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinS(argRanges));
}

void MinUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinU(argRanges));
}

void ShlOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              std::optional<unsigned> indexBitwidth,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShl(argRanges, indexBitwidth));
}

void ShrSOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrS(argRanges, indexBitwidth));
}

void ShrUOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               std::optional<unsigned> indexBitwidth,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrU(argRanges, indexBitwidth));
}

void AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              std::optional<unsigned> indexBitwidth,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAnd(argRanges));
}

void OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                             std::optional<unsigned> indexBitwidth,
                             SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferOr(argRanges));
}

void XOrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              std::optional<unsigned> indexBitwidth,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferXor(argRanges));
}

//===----------------------------------------------------------------------===//
// Casts
//===----------------------------------------------------------------------===//

void CastSOp::inferResultRangesFromOptional(
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

void CastUOp::inferResultRangesFromOptional(
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
// CmpOp
//===----------------------------------------------------------------------===//

void CmpOp::inferResultRangesFromOptional(ArrayRef<IntegerValueRange> argRanges,
                                          std::optional<unsigned> indexBitwidth,
                                          SetIntLatticeFn setResultRange) {
  index::IndexCmpPredicate indexPred = getPred();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(indexPred);
  setResultRange(getResult(), inferCmp(pred, argRanges, indexBitwidth));
}

//===----------------------------------------------------------------------===//
// SizeOf
//===----------------------------------------------------------------------===//

void SizeOfOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                 std::optional<unsigned> indexBitwidth,
                                 SetIntRangeFn setResultRange) {
  if (!indexBitwidth)
    return setResultRange(
        getResult(),
        ConstantIntRanges::maxRange(IndexType::kInternalStorageBitWidth));
  setResultRange(getResult(), ConstantIntRanges::constant(
                                  APInt(*indexBitwidth, *indexBitwidth)));
}
