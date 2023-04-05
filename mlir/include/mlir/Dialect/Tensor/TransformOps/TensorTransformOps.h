//===- TensorTransformOps.h - Tensor transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
#define MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensor {

/// A specialized TrackingListener for transform ops that operate on tensor IR.
/// This listener skips cast-like tensor ops when looking for payload op
/// replacements.
class TrackingListener : public transform::TrackingListener {
public:
  explicit TrackingListener(transform::TransformState &state)
      : transform::TrackingListener(state) {}

protected:
  Operation *findReplacementOp(Operation *op,
                               ValueRange newValues) const override;
};

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
