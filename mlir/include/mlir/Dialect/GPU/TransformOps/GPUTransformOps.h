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

/// Map the top level `scf.forall` op to GPU Thread Blocks.
/// Mapping is one-to-one and the induction variables of `scf.forall` are
/// rewritten to gpu.block_id according to the thread_dim_apping attribute.
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic block dim sizes are currently not supported.
DiagnosedSilenceableFailure mapForallToBlocksImpl(
    RewriterBase &rewriter, TransformOpInterface transformOp,
    scf::ForallOp forallOp, SmallVectorImpl<int64_t> &gridDims,
    const ArrayRef<DeviceMappingAttrInterface> &mappingAttributes,
    function_ref<void(RewriterBase &, scf::ForallOp, SmallVectorImpl<Value> &)>
        blockIdGenerator);

/// Search `scf.forall` ops nested under `target` and map each such op to GPU
/// threads. Mapping is one-to-one and the induction variables of `scf.forall`
/// are rewritten to gpu.thread_id according to the thread_dim_mapping
/// attribute.
/// Sibling `scf.forall` are supported in which case, the union of the number of
/// threads is computed and may result in predication.
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic block dim sizes are currently not supported.
DiagnosedSilenceableFailure mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    Operation *target, const SmallVectorImpl<int64_t> &kernelBlockDims,
    bool syncAfterDistribute,
    const ArrayRef<DeviceMappingAttrInterface> &threadMappingAttributes,
    function_ref<void(RewriterBase &, scf::ForallOp, SmallVectorImpl<Value> &)>
        threadIdGenerator);

/// Find the unique top level scf::ForallOp within a given target op.
DiagnosedSilenceableFailure
findTopLevelForallOp(Operation *target, scf::ForallOp &topLevelForallOp,
                     TransformOpInterface transformOp);

} // namespace gpu
} // namespace transform

namespace gpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMOPS_GPUTRANSFORMOPS_H
