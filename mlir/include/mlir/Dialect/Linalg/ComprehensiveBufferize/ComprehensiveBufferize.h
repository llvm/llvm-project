//===- ComprehensiveBufferize.h - Linalg bufferization pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {

namespace linalg {
namespace comprehensive_bufferize {

struct BufferizationOptions;
struct BufferizationState;
struct PostAnalysisStep;

/// Bufferize the given function. Does not bufferize the function boundary.
/// Reuses an existing BufferizationState object.
// TODO: This function is meant to be called from ModuleBufferize and not can
// not yet be called standalone.
LogicalResult runComprehensiveBufferize(
    FuncOp funcOp, const BufferizationOptions &options,
    BufferizationState &state,
    const std::vector<std::unique_ptr<PostAnalysisStep>> &extraSteps);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
