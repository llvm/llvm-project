//===- TensorTransformOps.cpp - Implementation of tensor transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace tensor;

//===----------------------------------------------------------------------===//
// TrackingListener
//===----------------------------------------------------------------------===//

Operation *
tensor::TrackingListener::findReplacementOp(Operation *op,
                                            ValueRange newValues) const {
  SmallVector<Value> values(newValues.begin(), newValues.end());
  do {
    if (Operation *replacement =
            transform::TrackingListener::findReplacementOp(op, values))
      return replacement;

    Operation *defOp = getCommonDefiningOp(values);
    if (!defOp)
      return nullptr;

    // Skip cast-like operations.
    // TODO: CastOpInterface could be used if CollapseShapeOp and ExpandShapeOp
    // implement that interface
    values.clear();
    llvm::TypeSwitch<Operation *>(defOp)
        .Case<CastOp>([&](CastOp op) { values.push_back(op.getSource()); })
        .Case<CollapseShapeOp>(
            [&](CollapseShapeOp op) { values.push_back(op.getSrc()); })
        .Case<ExpandShapeOp>(
            [&](ExpandShapeOp op) { values.push_back(op.getSrc()); })
        .Case<ReshapeOp>(
            [&](ReshapeOp op) { values.push_back(op.getSource()); })
        .Default([](Operation *op) {});
  } while (!values.empty());

  return nullptr;
}
