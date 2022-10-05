//===- GPUTransformOps.h - GPU transform ops --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMOPS_GPUTRANSFORMOPS_H
#define MLIR_DIALECT_GPU_TRANSFORMOPS_GPUTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gpu {
class GpuOp;
} // namespace gpu
} // namespace mlir

//===----------------------------------------------------------------------===//
// GPU Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h.inc"

namespace mlir {
class DialectRegistry;
namespace transform {
namespace gpu {

/// Searches `scf.foreach_thread` ops nested under `target` and maps each such
/// op to GPU threads. Mapping is one-to-one and the induction variables of
/// `scf.foreach_thread` are rewritten to gpu.thread_id according to the
/// thread_dim_apping attribute. Sibling `scf.foreach_thread` are supported in
/// which case, the union of the number of threads is computed and may result in
/// predication. Dynamic, `scf.foreach_thread` trip counts are currently not
/// supported. Dynamic block dim sizes are currently not supported.
DiagnosedSilenceableFailure
mapNestedForeachToThreadsImp(RewriterBase &rewriter, Operation *target,
                             const SmallVectorImpl<int64_t> &blockDim,
                             bool syncAfterDistribute,
                             llvm::Optional<TransformOpInterface> transformOp);

/// Maps the top level `scf.foreach_thread` op to GPU Thread Blocks. Mapping is
/// one-to-one and the induction variables of `scf.foreach_thread` are rewritten
/// to gpu.block_id according to the thread_dim_apping attribute. Dynamic,
/// `scf.foreach_thread` trip counts are currently not supported. Dynamic block
/// dim sizes are currently not supported.
DiagnosedSilenceableFailure mapForeachToBlocksImp(
    RewriterBase &rewriter, scf::ForeachThreadOp foreachThreadOp,
    function_ref<void(RewriterBase &, scf::ForeachThreadOp,
                      SmallVectorImpl<Value> &)>
        blockIdGenerator,
    SmallVectorImpl<int64_t> &gridDims, TransformOpInterface transformOp);

/// Finds the top level scf::ForeachThreadOp of given target.
DiagnosedSilenceableFailure
findTopLevelForeachThreadOp(Operation *target,
                            scf::ForeachThreadOp &topLevelForeachThreadOp,
                            TransformOpInterface transformOp);

} // namespace gpu
} // namespace transform

namespace gpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMOPS_GPUTRANSFORMOPS_H
