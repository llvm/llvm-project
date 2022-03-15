//===- ModuleBufferization.h - Bufferization across Func. Boundaries ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_MODULEBUFFERIZATION_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_MODULEBUFFERIZATION_H

#include <memory>

namespace mlir {

class DialectRegistry;
struct LogicalResult;
class ModuleOp;

namespace bufferization {
struct OneShotBufferizationOptions;
} // namespace bufferization

namespace linalg {
namespace comprehensive_bufferize {

/// Run Module Bufferization on the given module. Performs a simple function
/// call analysis to determine which function arguments are inplaceable. Then
/// analyzes and bufferizes FuncOps one-by-one with One-Shot Bufferize.
LogicalResult
runModuleBufferize(ModuleOp moduleOp,
                   bufferization::OneShotBufferizationOptions options);

namespace std_ext {

void registerModuleBufferizationExternalModels(DialectRegistry &registry);

} // namespace std_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_MODULEBUFFERIZATION_H
