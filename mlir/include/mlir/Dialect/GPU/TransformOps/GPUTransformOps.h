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

/// Helper type for functions that generate ids for the mapping of a
/// scf.forall.
struct IdBuilderResult {
  // Ops used to replace the forall induction variables.
  SmallVector<Value> mappingIdOps;
  // Actual mapping sizes used to predicate the forall body when they are
  // smaller than the available mapping sizes.
  SmallVector<int64_t> predicateMappingSizes;
  // Ops used to predicate the forall body when predicateMappingSizes is smaller
  // than the available mapping sizes.
  SmallVector<Value> predicateIdOps;
};

/// Common gpu id builder type, allows the configuration of lowering for various
/// mapping schemes. Takes:
///   - A rewriter with insertion point set before the forall op to rewrite.
///   - The loc of the forall op to rewrite.
///   - A list of positive integers carrying the mapping sizes for the current
///     forall op to rewrite.
using GpuIdBuilderFnType =
    std::function<IdBuilderResult(RewriterBase &, Location, ArrayRef<int64_t>)>;

/// Helper struct for configuring the rewrite of mapped scf.forall ops to
/// various gpu id configurations.
struct GpuIdBuilder {
  GpuIdBuilder(ArrayRef<OpFoldResult> blockDims, ArrayRef<int64_t> mappingSizes)
      : blockDimsOfr(blockDims), availableMappingSizes(mappingSizes),
        mappingAttributes(), idBuilder() {}

  /// List of OpFoldResult carrying the  multi-dimensional number of
  /// threads available in the current kernel (i.e. the current blockDims in
  /// CUDA parlance).
  ArrayRef<OpFoldResult> blockDimsOfr;

  /// A list of positive integers carrying the number of available mapping
  /// resources that can trigger predication,
  ArrayRef<int64_t> availableMappingSizes;

  /// The mapping attributes targeted by this generator.
  SmallVector<DeviceMappingAttrInterface> mappingAttributes;

  /// The constructor that builds the concrete IR for mapping ids.
  GpuIdBuilderFnType idBuilder;
};

/// Map the top level `scf.forall` op to GPU Thread Blocks.
/// Mapping is one-to-one and the induction variables of `scf.forall` are
/// rewritten to gpu.block_id according to the thread_dim_mapping attribute.
///
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic block dim sizes are currently not supported.
DiagnosedSilenceableFailure
mapForallToBlocksImpl(RewriterBase &rewriter, TransformOpInterface transformOp,
                      scf::ForallOp forallOp,
                      SmallVectorImpl<int64_t> &gridDims,
                      const GpuIdBuilder &gpuIdBuilder);

/// Search `scf.forall` ops nested under `target` and map each such op to an
/// explicit GPU implementation along `availableMappingSizes`.
/// The mapping is one-to-one and the induction variables of `scf.forall` are
/// rewritten to gpuIdBuilder.idBuilder according to the
/// gpuIdBuilder.mappingAttributes attribute.
///
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic `availableMappingSizes` sizes are currently not supported.
/// `availableMappingSizes` is expected to be of size 3.
DiagnosedSilenceableFailure mapOneForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> availableMappingSizes,
    bool syncAfterDistribute, const GpuIdBuilder &gpuIdBuilder);

/// Search `scf.forall` ops nested under `target` and map each such op to an
/// explicit GPU implementation along blockDims and warpDims.
/// The mapping is one-to-one and the induction variables of `scf.forall` are
/// rewritten to threads and warps ids according to the mapping attribute.
///
/// Dynamic, `scf.forall` trip counts are currently not supported.
/// Dynamic `blockDims` or `warpDims` or `linearDims` sizes are currently not
/// supported.
/// `blockDims` is expected to be of size 3.
/// `warpDims` is expected to be empty or of size 3.
///
/// The insertion point of the `rewriter` is expected to be set at the
/// beginning of the `target` body block and dominate all other blocks.
DiagnosedSilenceableFailure mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    Operation *target, ArrayRef<int64_t> blockDimsOfr,
    ArrayRef<int64_t> warpDims, bool syncAfterDistribute);

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
