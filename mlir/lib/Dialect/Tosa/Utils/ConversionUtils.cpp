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

#include "mlir/Dialect/Tosa/Utils/CoversionUtils.h"

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

Value mlir::tosa::clampFloatHelper(Location loc, Value arg,
                                   arith::ConstantOp min, arith::ConstantOp max,
                                   OpBuilder &rewriter) {
  Value minValue = rewriter.create<arith::MinFOp>(loc, arg, max);
  return rewriter.create<arith::MaxFOp>(loc, minValue, min);
}

Value mlir::tosa::clampIntHelper(Location loc, Value arg, arith::ConstantOp min,
                                 arith::ConstantOp max, OpBuilder &rewriter) {
  auto smallerThanMin =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, arg, min);
  auto minOrArg =
      rewriter.create<arith::SelectOp>(loc, smallerThanMin, min, arg);
  auto largerThanMax =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, max, arg);
  return rewriter.create<arith::SelectOp>(loc, largerThanMax, max, minOrArg);
}
