//===- ConversionUtils.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions for TOSA lowering
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;
using namespace mlir::tosa;

SmallVector<utils::IteratorType>
mlir::tosa::getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<utils::IteratorType>(nParallelLoops,
                                          utils::IteratorType::parallel);
}

SmallVector<Value>
mlir::tosa::condenseValues(const SmallVector<Value> &values) {
  SmallVector<Value> condensedValues;
  for (auto value : values)
    if (value)
      condensedValues.push_back(value);
  return condensedValues;
}

Value mlir::tosa::clampFloatHelper(Location loc, Value arg, Value min,
                                   Value max, OpBuilder &rewriter) {
  Value minValue = rewriter.create<arith::MinFOp>(loc, arg, max);
  return rewriter.create<arith::MaxFOp>(loc, minValue, min);
}

Value mlir::tosa::clampIntHelper(Location loc, Value arg, Value min, Value max,
                                 OpBuilder &rewriter) {
  auto smallerThanMin =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, arg, min);
  auto minOrArg =
      rewriter.create<arith::SelectOp>(loc, smallerThanMin, min, arg);
  auto largerThanMax =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, max, arg);
  return rewriter.create<arith::SelectOp>(loc, largerThanMax, max, minOrArg);
}

bool mlir::tosa::validIntegerRange(IntegerType ty, int64_t value) {
  uint64_t bitwidth = ty.getIntOrFloatBitWidth();
  if (ty.getSignedness() == IntegerType::Unsigned) {
    uint64_t uvalue = value;
    APInt intMin = APInt::getMinValue(bitwidth);
    APInt intMax = APInt::getMaxValue(bitwidth);
    return uvalue >= intMin.getZExtValue() && uvalue <= intMax.getZExtValue();
  }

  APInt intMin = APInt::getSignedMinValue(bitwidth);
  APInt intMax = APInt::getSignedMaxValue(bitwidth);
  return value >= intMin.getSExtValue() && value <= intMax.getSExtValue();
}

namespace {
// Given two tensors of high and low ranks, derive the output shape
// to reshape the lower rank to.
// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
LogicalResult
computeReshapeOutput(ArrayRef<int64_t> higherRankShape,
                     ArrayRef<int64_t> lowerRankShape,
                     SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      return failure();
  }
  return success();
}
} // namespace

LogicalResult mlir::tosa::EqualizeRanks(PatternRewriter &rewriter, Location loc,
                                        Value &input1, Value &input2) {
  auto input1Ty = llvm::dyn_cast<RankedTensorType>(input1.getType());
  auto input2Ty = llvm::dyn_cast<RankedTensorType>(input2.getType());

  if (!input1Ty || !input2Ty) {
    return failure();
  }

  int64_t input1Rank = input1Ty.getRank();
  int64_t input2Rank = input2Ty.getRank();

  if (input1Rank == input2Rank)
    return success();

  Value higherTensorValue, lowerTensorValue;
  if (input1Rank > input2Rank) {
    higherTensorValue = input1;
    lowerTensorValue = input2;
  } else {
    higherTensorValue = input2;
    lowerTensorValue = input1;
  }

  ArrayRef<int64_t> higherRankShape =
      llvm::cast<RankedTensorType>(higherTensorValue.getType()).getShape();
  ArrayRef<int64_t> lowerRankShape =
      llvm::cast<RankedTensorType>(lowerTensorValue.getType()).getShape();

  SmallVector<int64_t, 4> reshapeOutputShape;

  if (computeReshapeOutput(higherRankShape, lowerRankShape, reshapeOutputShape)
          .failed())
    return failure();

  auto reshapeInputType =
      llvm::cast<RankedTensorType>(lowerTensorValue.getType());
  auto reshapeOutputType = RankedTensorType::get(
      ArrayRef<int64_t>(reshapeOutputShape), reshapeInputType.getElementType());

  auto reshapeLower = rewriter.create<tosa::ReshapeOp>(
      loc, reshapeOutputType, lowerTensorValue,
      rewriter.getDenseI64ArrayAttr(reshapeOutputShape));

  if (input1Rank > input2Rank) {
    input1 = higherTensorValue;
    input2 = reshapeLower.getResult();
  } else {
    input1 = reshapeLower.getResult();
    input2 = higherTensorValue;
  }

  return success();
}
