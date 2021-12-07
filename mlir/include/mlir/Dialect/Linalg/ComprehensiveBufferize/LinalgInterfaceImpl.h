//===- LinalgInterfaceImpl.h - Linalg Impl. of BufferizableOpInterface ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_LINALG_INTERFACE_IMPL_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_LINALG_INTERFACE_IMPL_H

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"

namespace mlir {

class DialectRegistry;

namespace linalg {
namespace comprehensive_bufferize {

class BufferizationAliasInfo;

namespace linalg_ext {

struct InitTensorEliminationStep : public PostAnalysisStep {
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
  LogicalResult eliminateInitTensors(
      Operation *op, BufferizationState &state,
      BufferizationAliasInfo &aliasInfo,
      std::function<bool(OpOperand &)> anchorMatchFunc,
      std::function<Value(OpBuilder &, Location, OpOperand &)> rewriteFunc,
      SmallVector<Operation *> &newOps);
};

/// Try to eliminate InitTensorOps inside `op` that are anchored on an
/// InsertSliceOp, i.e., if it is eventually inserted into another tensor
/// (and some other conditions are met).
struct InsertSliceAnchoredInitTensorEliminationStep
    : public InitTensorEliminationStep {
  LogicalResult run(Operation *op, BufferizationState &state,
                    BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override;
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace linalg_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_LINALG_INTERFACE_IMPL_H
