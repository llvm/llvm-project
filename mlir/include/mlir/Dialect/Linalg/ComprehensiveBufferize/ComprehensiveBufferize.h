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
class BufferizationState;
struct PostAnalysisStep;

/// Bufferize the given operation. Reuses an existing BufferizationState object.
LogicalResult runComprehensiveBufferize(
    Operation *op, const BufferizationOptions &options,
    BufferizationState &state,
    const std::vector<std::unique_ptr<PostAnalysisStep>> &extraSteps);

/// Bufferize the given operation.
LogicalResult runComprehensiveBufferize(Operation *op,
                                        const BufferizationOptions &options);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
