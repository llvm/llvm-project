//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

mlir::edsc::VectorBoundsCapture::VectorBoundsCapture(VectorType t) {
  for (auto s : t.getShape()) {
    lbs.push_back(std_constant_index(0));
    ubs.push_back(std_constant_index(s));
    steps.push_back(1);
  }
}

mlir::edsc::VectorBoundsCapture::VectorBoundsCapture(Value v)
    : VectorBoundsCapture(v.getType().cast<VectorType>()) {}
