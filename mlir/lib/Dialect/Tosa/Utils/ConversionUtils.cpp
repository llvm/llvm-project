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

SmallVector<StringRef>
mlir::tosa::getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<StringRef>(nParallelLoops, getParallelIteratorTypeName());
}

SmallVector<Value>
mlir::tosa::condenseValues(const SmallVector<Value> &values) {
  SmallVector<Value> condensedValues;
  for (auto value : values)
    if (value)
      condensedValues.push_back(value);
  return condensedValues;
}
