//===- ValueRange.cpp - Indexed Value-Iterators Range Classes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/ValueRange.h"
#include "mlir/IR/TypeRange.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TypeRangeRange

TypeRangeRange OperandRangeRange::getTypes() const {
  return TypeRangeRange(*this);
}

TypeRangeRange OperandRangeRange::getType() const { return getTypes(); }

//===----------------------------------------------------------------------===//
// OperandRange

OperandRange::type_range OperandRange::getTypes() const {
  return {begin(), end()};
}

OperandRange::type_range OperandRange::getType() const { return getTypes(); }

//===----------------------------------------------------------------------===//
// ResultRange

ResultRange::type_range ResultRange::getTypes() const {
  return {begin(), end()};
}

ResultRange::type_range ResultRange::getType() const { return getTypes(); }

//===----------------------------------------------------------------------===//
// ValueRange

ValueRange::type_range ValueRange::getTypes() const { return {begin(), end()}; }

ValueRange::type_range ValueRange::getType() const { return getTypes(); }
