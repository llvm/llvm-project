//===- AllocTensorElimination.h - alloc_tensor op elimination -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ALLOCTENSORELIMINATION_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ALLOCTENSORELIMINATION_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

namespace mlir {
namespace bufferization {

/// A function that matches anchor OpOperands for AllocTensorOp elimination.
/// If an OpOperand is matched, the function should populate the SmallVector
/// with all values that are needed during `RewriteFn` to produce the
/// replacement value.
using AnchorMatchFn = std::function<bool(OpOperand &, SmallVector<Value> &)>;

/// A function that rewrites matched anchors.
using RewriteFn = std::function<Value(OpBuilder &, Location, OpOperand &)>;

/// Try to eliminate AllocTensorOps inside `op`.
///
/// * `rewriteFunc` generates the replacement for the AllocTensorOp.
/// * Only AllocTensorOps that are anchored on a matching OpOperand as per
///   `anchorMatchFunc` are considered. "Anchored" means that there is a path
///   on the reverse SSA use-def chain, starting from the OpOperand and always
///   following the aliasing  OpOperand, that eventually ends at a single
///   AllocTensorOp.
LogicalResult eliminateAllocTensors(RewriterBase &rewriter, Operation *op,
                                    bufferization::AnalysisState &state,
                                    AnchorMatchFn anchorMatchFunc,
                                    RewriteFn rewriteFunc);

/// Try to eliminate AllocTensorOps inside `op` that are anchored on an
/// InsertSliceOp, i.e., if it is eventually inserted into another tensor
/// (and some other conditions are met).
LogicalResult insertSliceAnchoredAllocTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, bufferization::AnalysisState &state);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ALLOCTENSORELIMINATION_H
