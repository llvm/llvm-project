//===- GPUTransformOps.h - GPU transform ops --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMOPS_GPUTRANSFORMOPS_H
#define MLIR_DIALECT_GPU_TRANSFORMOPS_GPUTRANSFORMOPS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
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
struct GpuIdBuilder;

/// Map the top level `scf.forall` op to GPU blocks.
/// Mapping is one-to-one and the induction variables of `scf.forall` are
/// rewritten to gpu.block_id according to the thread_dim_mapping attribute.
///
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic `gridDims` are currently not supported.
DiagnosedSilenceableFailure
mapForallToBlocksImpl(RewriterBase &rewriter, TransformOpInterface transformOp,
                      scf::ForallOp forallOp,
                      SmallVectorImpl<int64_t> &gridDims,
                      const GpuIdBuilder &gpuIdBuilder);

/// Search `scf.forall` ops nested under `target` and map each such op to an
/// explicit GPU implementation along `blockDims`.
/// The mapping is one-to-one and the induction variables of `scf.forall` are
/// rewritten to gpuIdBuilder.idBuilder according to the
/// gpuIdBuilder.mappingAttributes attribute.
///
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic `blockDims` sizes are currently not supported.
/// `blockDims` is expected to be of size 3.
DiagnosedSilenceableFailure
mapOneForallToThreadsImpl(RewriterBase &rewriter,
                          std::optional<TransformOpInterface> transformOp,
                          scf::ForallOp forallOp, ArrayRef<int64_t> blockDims,
                          int64_t warpSize, bool syncAfterDistribute);

/// Search `scf.forall` ops nested under `target` and map each such op to an
/// explicit GPU implementation along `blockDims`.
/// The mapping is one-to-one and the induction variables of `scf.forall` are
/// rewritten to appropriate ids according to the mapping attribute.
///
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic `blockDims` or `newBasis` entries are currently not
/// supported. `blockDims` is expected to be of size 3.
///
/// The insertion point of the `rewriter` is expected to be set at the
/// beginning of the `target` body block and dominate all other blocks.
DiagnosedSilenceableFailure
mapNestedForallToThreadsImpl(RewriterBase &rewriter,
                             std::optional<TransformOpInterface> transformOp,
                             Operation *target, ArrayRef<int64_t> blockDims,
                             int64_t warpSize, bool syncAfterDistribute);

} // namespace gpu
} // namespace transform

namespace gpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMOPS_GPUTRANSFORMOPS_H
