//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_BUFFERIZABLEOPINTERFACEIMPL_H
#define MLIR_DIALECT_SCF_BUFFERIZABLEOPINTERFACEIMPL_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

namespace mlir {
class DialectRegistry;

namespace scf {
/// Assert that yielded values of an scf.for op are aliasing their corresponding
/// bbArgs. This is required because the i-th OpResult of an scf.for op is
/// currently assumed to alias with the i-th iter_arg (in the absence of
/// conflicts).
struct AssertScfForAliasingProperties : public bufferization::PostAnalysisStep {
  LogicalResult run(Operation *op, bufferization::BufferizationState &state,
                    bufferization::BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override;
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_BUFFERIZABLEOPINTERFACEIMPL_H
