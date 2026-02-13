//===- Passes.h - Shard Passes ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHARD_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_SHARD_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
}

namespace shard {

/// This enum controls the traversal order for the sharding propagation.
enum class TraversalOrder {
  /// Forward traversal.
  Forward,
  /// Backward traversal.
  Backward,
  /// Forward then backward traversal.
  ForwardBackward,
  /// Backward then forward traversal.
  BackwardForward
};

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/Shard/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Shard/Transforms/Passes.h.inc"

} // namespace shard
} // namespace mlir

#endif // MLIR_DIALECT_SHARD_TRANSFORMS_PASSES_H
