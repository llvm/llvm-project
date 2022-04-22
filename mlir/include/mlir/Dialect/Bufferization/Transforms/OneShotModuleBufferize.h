//===- OneShotModuleBufferize.h - Bufferization across Func. Boundaries ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTMODULEBUFFERIZE_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTMODULEBUFFERIZE_H

namespace mlir {

struct LogicalResult;
class ModuleOp;

namespace bufferization {
struct OneShotBufferizationOptions;

/// Run One-Shot Module Bufferization on the given module. Performs a simple
/// function call analysis to determine which function arguments are
/// inplaceable. Then analyzes and bufferizes FuncOps one-by-one with One-Shot
/// Bufferize.
LogicalResult
runOneShotModuleBufferize(ModuleOp moduleOp,
                          bufferization::OneShotBufferizationOptions options);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTMODULEBUFFERIZE_H
