//===- SCFInterfaceImpl.h - SCF Impl. of BufferizableOpInterface ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_SCF_INTERFACE_IMPL_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_SCF_INTERFACE_IMPL_H

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"

namespace mlir {

class DialectRegistry;

namespace linalg {
namespace comprehensive_bufferize {
namespace scf_ext {

/// Equivalence analysis for scf.for. Raise an error if iter_args are not
/// equivalent to their corresponding loop yield values.
struct AssertDestinationPassingStyle : public PostAnalysisStep {
  LogicalResult run(Operation *op, BufferizationState &state,
                    BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override;
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace scf_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_SCF_INTERFACE_IMPL_H
