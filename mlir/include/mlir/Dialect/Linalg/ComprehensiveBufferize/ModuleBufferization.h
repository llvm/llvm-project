//===- ModuleBufferization.h - Bufferization across Func. Boundaries ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_MODULE_BUFFERIZATION_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_MODULE_BUFFERIZATION_H

namespace mlir {

class DialectRegistry;
struct LogicalResult;
class ModuleOp;

namespace linalg {
namespace comprehensive_bufferize {

struct BufferizationOptions;

/// Bufferize the given module. This bufferizations performs a simple function
/// call analysis to determine which function arguments are inplaceable.
LogicalResult runComprehensiveBufferize(ModuleOp moduleOp,
                                        const BufferizationOptions &options);

namespace std_ext {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace std_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_MODULE_BUFFERIZATION_H
