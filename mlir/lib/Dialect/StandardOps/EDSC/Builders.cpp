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

static SmallVector<Value, 8> getMemRefSizes(Value memRef) {
  MemRefType memRefType = memRef.getType().cast<MemRefType>();
  assert(isStrided(memRefType) && "Expected strided MemRef type");

  SmallVector<Value, 8> res;
  res.reserve(memRefType.getShape().size());
  const auto &shape = memRefType.getShape();
  for (unsigned idx = 0, n = shape.size(); idx < n; ++idx) {
    if (shape[idx] == -1)
      res.push_back(std_dim(memRef, idx));
    else
      res.push_back(std_constant_index(shape[idx]));
  }
  return res;
}

mlir::edsc::MemRefBoundsCapture::MemRefBoundsCapture(Value v) {
  auto memrefSizeValues = getMemRefSizes(v);
  for (auto s : memrefSizeValues) {
    lbs.push_back(std_constant_index(0));
    ubs.push_back(s);
    steps.push_back(1);
  }
}

mlir::edsc::VectorBoundsCapture::VectorBoundsCapture(VectorType t) {
  for (auto s : t.getShape()) {
    lbs.push_back(std_constant_index(0));
    ubs.push_back(std_constant_index(s));
    steps.push_back(1);
  }
}

mlir::edsc::VectorBoundsCapture::VectorBoundsCapture(Value v)
    : VectorBoundsCapture(v.getType().cast<VectorType>()) {}
