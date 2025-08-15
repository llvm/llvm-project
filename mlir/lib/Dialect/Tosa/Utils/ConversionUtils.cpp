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
  Value minValue = arith::MinimumFOp::create(rewriter, loc, arg, max);
  return arith::MaximumFOp::create(rewriter, loc, minValue, min);
}

Value mlir::tosa::clampIntHelper(Location loc, Value arg, Value min, Value max,
                                 OpBuilder &rewriter, bool isUnsigned) {
  if (isUnsigned) {
    auto minOrArg = arith::MaxUIOp::create(rewriter, loc, min, arg);
    return arith::MinUIOp::create(rewriter, loc, max, minOrArg);
  }
  auto minOrArg = arith::MaxSIOp::create(rewriter, loc, min, arg);
  return arith::MinSIOp::create(rewriter, loc, max, minOrArg);
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
  const int64_t rankDiff = higherRank - lowerRank;

  for (int64_t i = lowerRank - 1; i >= 0; i--) {
    higherRankDim = higherRankShape[i + rankDiff];
    lowerRankDim = lowerRankShape[i];

    if (lowerRankDim != 1 && higherRankDim != 1 &&
        lowerRankDim != higherRankDim)
      return failure();

    reshapeOutputShape[i + rankDiff] = lowerRankDim == 1 ? 1 : lowerRankDim;
  }
  return success();
}
} // namespace

LogicalResult mlir::tosa::EqualizeRanks(PatternRewriter &rewriter, Location loc,
                                        Value &input1, Value &input2) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  return EqualizeRanks(builder, input1, input2);
}

LogicalResult mlir::tosa::EqualizeRanks(ImplicitLocOpBuilder &builder,
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
  auto reshapeOutputShapeValue = getTosaConstShape(builder, reshapeOutputShape);

  auto reshapeLower = tosa::ReshapeOp::create(
      builder, reshapeOutputType, lowerTensorValue, reshapeOutputShapeValue);

  if (input1Rank > input2Rank) {
    input1 = higherTensorValue;
    input2 = reshapeLower.getResult();
  } else {
    input1 = reshapeLower.getResult();
    input2 = higherTensorValue;
  }

  return success();
}

Value mlir::tosa::getTosaConstShape(ImplicitLocOpBuilder &builder,
                                    llvm::ArrayRef<int64_t> shape) {
  auto attr = builder.getIndexTensorAttr(convertFromMlirShape(shape));
  auto type = mlir::tosa::shapeType::get(builder.getContext(), shape.size());
  mlir::Operation *mlir_op = tosa::ConstShapeOp::create(builder, type, attr);
  return mlir_op->getResult(0);
}

Value mlir::tosa::getTosaConstShape(PatternRewriter &rewriter, Location loc,
                                    llvm::ArrayRef<int64_t> shape) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  return getTosaConstShape(builder, shape);
}

SmallVector<int64_t> mlir::tosa::convertFromMlirShape(ArrayRef<int64_t> shape) {
  return to_vector(llvm::map_range(shape, [](int64_t dim) {
    return ShapedType::isDynamic(dim) ? -1 : dim;
  }));
}

bool mlir::tosa::getConstShapeValues(Operation *op,
                                     llvm::SmallVector<int64_t> &result_shape) {
  if (!op) {
    return false;
  }
  if (auto constOp = mlir::dyn_cast<tosa::ConstShapeOp>(op)) {
    Attribute constOpAttr = constOp->getAttr("values");
    DenseElementsAttr elementsAttr = cast<DenseElementsAttr>(constOpAttr);
    for (int i = 0; i < elementsAttr.size(); i++) {
      int64_t val = elementsAttr.getValues<int64_t>()[i];
      result_shape.push_back(val);
    }
    return true;
  }
  // for undefined op, return false.
  return false;
}

// returns a small vector of int64_t values that attr contains
SmallVector<int64_t>
mlir::tosa::convertFromIntAttr(const DenseElementsAttr &attr, const int rank) {
  if (attr.isSplat()) {
    int64_t v = attr.getSplatValue<APInt>().getSExtValue();
    return SmallVector<int64_t>(rank, v);
  }

  if (auto int_array_attr = llvm::dyn_cast<DenseIntElementsAttr>(attr)) {
    SmallVector<int64_t> vec;
    for (APInt val : int_array_attr.getValues<APInt>()) {
      vec.push_back(val.getSExtValue());
    }
    return vec;
  }
  return {};
}

bool mlir::tosa::hasUniqueConstantScatterIndices(
    ShapedType indicesType, DenseIntElementsAttr indicesAttr) {
  llvm::ArrayRef<int64_t> const indicesShape = indicesType.getShape();
  const unsigned int indicesRank = indicesShape.size();
  const unsigned int lastDimSize = indicesShape[indicesRank - 1];

  // check each batch of indices from the flat indicesAttr values
  // for duplicates
  auto const indicesValues = indicesAttr.getValues<int32_t>();
  assert(
      (indicesValues.size() % lastDimSize == 0) &&
      "Constant indices data length should be a multiple of indicesShape[-1]");

  std::vector<uint64_t> indices(lastDimSize);
  for (auto beg = indicesValues.begin(); beg < indicesValues.end();
       beg += lastDimSize) {
    std::copy(beg, beg + lastDimSize, indices.begin());
    std::sort(indices.begin(), indices.end());
    if (std::adjacent_find(indices.begin(), indices.end()) != indices.end()) {
      // found duplicate values in indices in batch
      return false;
    }
  }

  return true;
}
