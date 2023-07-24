//===- Utils.h - Utils for GPU transform ops --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMOPS_UTILS_H
#define MLIR_DIALECT_GPU_TRANSFORMOPS_UTILS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gpu {
class GPUOp;
class LaunchOp;
enum class MappingId : uint64_t;
} // namespace gpu
namespace scf {
class ForallOp;
} // namespace scf
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

/// Builder for gpu::BlockIdOps used in mapping scf.forall to blocks.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuBlockIdBuilder : public GpuIdBuilder {
  GpuBlockIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                    ArrayRef<int64_t> mappingSizes);
};

/// Builder for gpu::ThreadIdOp used in mapping scf.forall to thread ids without
/// any reindexing.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuThreadIdBuilder : public GpuIdBuilder {
  GpuThreadIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                     ArrayRef<int64_t> mappingSizes);
};

/// Builder for warp ids used in mapping scf.forall to warps.
/// This builder requires a specification of the number of warps along each
/// dimension to more finely control mapping to warps as well a predication than
/// by solely analyzing the IR.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuWarpIdBuilder : public GpuIdBuilder {
  GpuWarpIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                   ArrayRef<int64_t> mappingSizes);
  /// Static specification of the warp size.
  /// In the future this may be configured by the transformation.
  static constexpr int64_t kWarpSize = 32;
};

/// Builder for linear ids used in mapping scf.forall to reindexed threads.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 1-D sizes for predicate generation.
struct GpuLinearIdBuilder : public GpuIdBuilder {
  GpuLinearIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                     ArrayRef<int64_t> mappingSizes);
};

/// Determine if the size of the kernel configuration is supported by the
/// GPU architecture being used.
/// TODO this is currently hardwired to CUDA, parameterize and generalize.
DiagnosedSilenceableFailure checkGpuLimits(TransformOpInterface transformOp,
                                           std::optional<int64_t> gridDimX,
                                           std::optional<int64_t> gridDimY,
                                           std::optional<int64_t> gridDimZ,
                                           std::optional<int64_t> blockDimX,
                                           std::optional<int64_t> blockDimY,
                                           std::optional<int64_t> blockDimZ);

/// Create an empty-body gpu::LaunchOp using the provided kernel settings
/// and put a terminator within.
DiagnosedSilenceableFailure
createGpuLaunch(RewriterBase &rewriter, Location loc,
                TransformOpInterface transformOp, mlir::gpu::LaunchOp &launchOp,
                std::optional<int64_t> gridDimX = std::nullopt,
                std::optional<int64_t> gridDimY = std::nullopt,
                std::optional<int64_t> gridDimZ = std::nullopt,
                std::optional<int64_t> blockDimX = std::nullopt,
                std::optional<int64_t> blockDimY = std::nullopt,
                std::optional<int64_t> blockDimZ = std::nullopt);

/// Alter kernel configuration of the given kernel.
DiagnosedSilenceableFailure
alterGpuLaunch(RewriterBase &rewriter, mlir::gpu::LaunchOp gpuLaunch,
               TransformOpInterface transformOp,
               std::optional<int64_t> gridDimX = std::nullopt,
               std::optional<int64_t> gridDimY = std::nullopt,
               std::optional<int64_t> gridDimZ = std::nullopt,
               std::optional<int64_t> blockDimX = std::nullopt,
               std::optional<int64_t> blockDimY = std::nullopt,
               std::optional<int64_t> blockDimZ = std::nullopt);

/// Find the unique top level scf::ForallOp within a given target op.
DiagnosedSilenceableFailure
findTopLevelForallOp(Operation *target, scf::ForallOp &topLevelForallOp,
                     TransformOpInterface transformOp);

} // namespace gpu
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMOPS_UTILS_H
