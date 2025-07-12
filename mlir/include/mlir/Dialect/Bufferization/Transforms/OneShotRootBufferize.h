//===- OneShotRootBufferize.h - Bufferization across Func. Boundaries ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTROOTBUFFERIZE_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTROOTBUFFERIZE_H

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace mlir {
class Operation;

namespace bufferization {
struct BufferizationStatistics;
class OneShotAnalysisState;
struct OneShotBufferizationOptions;
class BufferizationState;

/// Analyze `rootOp` and its nested ops. Bufferization decisions are stored in
/// `state`.
llvm::LogicalResult
analyzeRootOp(Operation *rootOp, OneShotAnalysisState &state,
              BufferizationStatistics *statistics = nullptr);

/// Bufferize `op` and its nested ops that implement `BufferizableOpInterface`.
///
/// Note: This function does not run One-Shot Analysis. No buffer copies are
/// inserted except two cases:
/// - `options.copyBeforeWrite` is set, in which case buffers are copied before
///   every write.
/// - `options.copyBeforeWrite` is not set and `options.noAnalysisFuncFilter`
///   is not empty. The FuncOps it contains were not analyzed. Buffer copies
///   will be inserted only to these FuncOps.
llvm::LogicalResult
bufferizeRootOp(Operation *rootOp, const OneShotBufferizationOptions &options,
                BufferizationState &state,
                BufferizationStatistics *statistics = nullptr);

/// Remove bufferization attributes on every FuncOp arguments in the RootOp.
void removeBufferizationAttributesInRoot(Operation *rootOp);

/// Run One-Shot Root Bufferization on the given root op. Performs a simple
/// function call analysis to determine which function arguments are
/// inplaceable. Then analyzes and bufferizes FuncOps one-by-one with One-Shot
/// Bufferize.
llvm::LogicalResult runOneShotRootBufferize(
    Operation *rootOp,
    const bufferization::OneShotBufferizationOptions &options,
    BufferizationState &state, BufferizationStatistics *statistics = nullptr);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTROOTBUFFERIZE_H
