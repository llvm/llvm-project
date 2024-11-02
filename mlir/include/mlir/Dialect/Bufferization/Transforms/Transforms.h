//===- Transforms.h - Bufferization and related transforms ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace bufferization {
class AnalysisState;
struct OneShotBufferizationOptions;

/// A function that matches anchor OpOperands for tensor::EmptyOp elimination.
/// If an OpOperand is matched, the function should populate the SmallVector
/// with all values that are needed during `RewriteFn` to produce the
/// replacement value.
using AnchorMatchFn = std::function<bool(OpOperand &, SmallVector<Value> &)>;

/// A function that rewrites matched anchors.
using RewriteFn = std::function<Value(OpBuilder &, Location, OpOperand &)>;

/// Try to eliminate tensor::EmptyOps inside `op`.
///
/// * `rewriteFunc` generates the replacement for the tensor::EmptyOp.
/// * Only tensor::EmptyOps that are anchored on a matching OpOperand as per
///   `anchorMatchFunc` are considered. "Anchored" means that there is a path
///   on the reverse SSA use-def chain, starting from the OpOperand and always
///   following the aliasing  OpOperand, that eventually ends at a single
///   tensor::EmptyOp.
LogicalResult eliminateEmptyTensors(RewriterBase &rewriter, Operation *op,
                                    bufferization::AnalysisState &state,
                                    AnchorMatchFn anchorMatchFunc,
                                    RewriteFn rewriteFunc);

/// Try to eliminate tensor::EmptyOps inside `op` that are anchored on an
/// InsertSliceOp, i.e., if it is eventually inserted into another tensor
/// (and some other conditions are met).
LogicalResult insertSliceAnchoredEmptyTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, bufferization::AnalysisState &state);

/// Resolve RaW and other conflicts by inserting bufferization.alloc_tensor ops.
/// After applying this transform, the IR can be bufferized without inserting
/// additional buffer allocations.
LogicalResult insertTensorCopies(Operation *op,
                                 const OneShotBufferizationOptions &options);

/// Resolve RaW and other conflicts by inserting bufferization.alloc_tensor ops.
/// After applying this transform, the IR can be bufferized without inserting
/// additional buffer allocations.
LogicalResult insertTensorCopies(Operation *op, const AnalysisState &state);

/// Populate patterns to lower tensor.empty ops to bufferization.alloc_tensor
/// ops.
void populateEmptyTensorToAllocTensorPattern(RewritePatternSet &patterns);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H
