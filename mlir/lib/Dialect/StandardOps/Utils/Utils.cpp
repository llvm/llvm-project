//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Utils/Utils.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

SmallVector<Value, 4> mlir::getDynOperands(Location loc, Value val,
                                           OpBuilder &b) {
  SmallVector<Value, 4> dynOperands;
  auto shapedType = val.getType().cast<ShapedType>();
  for (auto dim : llvm::enumerate(shapedType.getShape())) {
    if (dim.value() == TensorType::kDynamicSize)
      dynOperands.push_back(b.create<DimOp>(loc, val, dim.index()));
  }
  return dynOperands;
}
