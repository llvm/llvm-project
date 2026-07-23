//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for affine --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::intrange;

//===----------------------------------------------------------------------===//
// AffineApplyOp
//===----------------------------------------------------------------------===//

void AffineApplyOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  AffineMap map = getAffineMap();

  // Split operand ranges into dimensions and symbols.
  unsigned numDims = map.getNumDims();
  ArrayRef<ConstantIntRanges> dimRanges = argRanges.take_front(numDims);
  ArrayRef<ConstantIntRanges> symbolRanges = argRanges.drop_front(numDims);

  // Affine maps should have exactly one result for affine.apply.
  assert(map.getNumResults() == 1 && "affine.apply must have single result");

  // Infer the range for the affine expression.
  ConstantIntRanges resultRange =
      inferAffineExpr(map.getResult(0), dimRanges, symbolRanges);

  setResultRange(getResult(), resultRange);
}
