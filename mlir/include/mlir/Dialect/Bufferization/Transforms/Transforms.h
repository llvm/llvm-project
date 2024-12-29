//===- Transforms.h - Bufferization and related transforms ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SubsetOpInterface.h"

namespace mlir {
namespace bufferization {
class AnalysisState;
struct BufferizationStatistics;
class OneShotAnalysisState;
struct OneShotBufferizationOptions;

/// Try to eliminate "tensor.empty" ops inside `op`. This transformation looks
/// for subset ops that insert a tensor that originates from a "tensor.empty"
/// (as per the reverse use-def chain). Such "tensor.empty" ops are replaced
/// with the destination subset.
///
/// E.g.:
/// %0 = tensor.empty() : tensor<10xf32>
/// %1 = linalg.fill ... outs(%0 : tensor<10xf32>)
/// %2 = tensor.insert_slice %0 into %t ...
///
/// In the above example, the subset op is "tensor.insert_slice". When tracing
/// back the reverse use-def chain of a the source, we end up at a
/// "tensor.empty" op.
LogicalResult eliminateEmptyTensors(RewriterBase &rewriter, Operation *op);

/// A function type that defines a callback to control the construction
/// of the subset extraction of the `SubsetInsertionOpInterface`.
/// The subset extraction value can be used as a replacement for the
/// `emptyTensorOp` value which is being consumed by `user`, failing
/// of building such a value should be indicated with an empty value.
/// This function should guarantee the legality of the replacement,
/// i.e. the replacement should dominate the user of the `emptyTensorOp`
/// being eliminated.
using ControlBuildSubsetExtractionFn =
    std::function<Value(RewriterBase &, SubsetInsertionOpInterface,
                        tensor::EmptyOp emptyTensorOp, Operation *user)>;

/// This method builds and returns a subset extraction value for the
/// destination tensor that the given `op` inserts into.
/// It returns a value which should replace the `emptyTensorOp` use
/// that is being consumed by `user`.
/// If no such a value found it will return an empty Value.
Value buildSubsetExtraction(RewriterBase &rewriter,
                            SubsetInsertionOpInterface op,
                            tensor::EmptyOp emptyTensorOp, Operation *user);

/// Try to eliminate "tensor.empty" ops inside `op`.
///
/// This function overload accepts an existing `OneShotAnalysisState`, which
/// contains in-place bufferization decisions. This overload is useful if an
/// existing analysis should be reused for empty tensor elimination.
LogicalResult eliminateEmptyTensors(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state,
    ControlBuildSubsetExtractionFn subsetsExtractionFn = buildSubsetExtraction);

/// Within the given operation, hoist buffers from loops where possible. See
/// "BufferLoopHoistingPass" for more information.
void hoistBuffersFromLoops(Operation *op);

/// Resolve RaW and other conflicts by inserting bufferization.alloc_tensor ops.
/// After applying this transform, the IR can be bufferized without inserting
/// additional buffer allocations.
LogicalResult insertTensorCopies(Operation *op,
                                 const OneShotBufferizationOptions &options,
                                 BufferizationStatistics *statistics = nullptr);

/// Resolve RaW and other conflicts by inserting bufferization.alloc_tensor ops.
/// After applying this transform, the IR can be bufferized without inserting
/// additional buffer allocations.
LogicalResult insertTensorCopies(Operation *op, const AnalysisState &state);

/// Populate patterns to lower tensor.empty ops to bufferization.alloc_tensor
/// ops.
void populateEmptyTensorToAllocTensorPattern(RewritePatternSet &patterns);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H
