//===- LinalgInterfaceImpl.h - Linalg Impl. of BufferizableOpInterface ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_LINALGINTERFACEIMPL_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_LINALGINTERFACEIMPL_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

namespace mlir {
class DialectRegistry;

namespace linalg {
namespace comprehensive_bufferize {
namespace linalg_ext {

struct InitTensorEliminationStep : public bufferization::PostAnalysisStep {
  /// A function that matches anchor OpOperands for InitTensorOp elimination.
  /// If an OpOperand is matched, the function should populate the SmallVector
  /// with all values that are needed during `RewriteFn` to produce the
  /// replacement value.
  using AnchorMatchFn = std::function<bool(OpOperand &, SmallVector<Value> &)>;

  /// A function that rewrites matched anchors.
  using RewriteFn = std::function<Value(OpBuilder &, Location, OpOperand &)>;

  /// Try to eliminate InitTensorOps inside `op`.
  ///
  /// * `rewriteFunc` generates the replacement for the InitTensorOp.
  /// * Only InitTensorOps that are anchored on a matching OpOperand as per
  ///   `anchorMatchFunc` are considered. "Anchored" means that there is a path
  ///   on the reverse SSA use-def chain, starting from the OpOperand and always
  ///   following the aliasing  OpOperand, that eventually ends at a single
  ///   InitTensorOp.
  /// * The result of `rewriteFunc` must usually be analyzed for inplacability.
  ///   This analysis can be skipped with `skipAnalysis`.
  LogicalResult
  eliminateInitTensors(Operation *op, bufferization::BufferizationState &state,
                       bufferization::BufferizationAliasInfo &aliasInfo,
                       AnchorMatchFn anchorMatchFunc, RewriteFn rewriteFunc,
                       SmallVector<Operation *> &newOps);
};

/// Try to eliminate InitTensorOps inside `op` that are anchored on an
/// InsertSliceOp, i.e., if it is eventually inserted into another tensor
/// (and some other conditions are met).
struct InsertSliceAnchoredInitTensorEliminationStep
    : public InitTensorEliminationStep {
  LogicalResult run(Operation *op, bufferization::BufferizationState &state,
                    bufferization::BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override;
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace linalg_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_LINALGINTERFACEIMPL_H
