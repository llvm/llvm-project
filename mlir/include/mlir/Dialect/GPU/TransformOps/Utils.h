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
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
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

/// Helper type for functions that generate ids for the mapping of a scf.forall.
struct IdBuilderResult {
  /// Error message, if not empty then building the ids failed.
  std::string errorMsg;
  /// Values used to replace the forall induction variables.
  SmallVector<Value> mappingIdOps;
  /// Values used to predicate the forall body when activeMappingSizes is
  /// smaller than the available mapping sizes.
  SmallVector<Value> predicateOps;
};

inline raw_ostream &operator<<(raw_ostream &os, const IdBuilderResult &res) {
  llvm::interleaveComma(res.mappingIdOps, os << "----mappingIdOps: ");
  os << "\n";
  llvm::interleaveComma(res.predicateOps, os << "----predicateOps: ");
  os << "\n";
  return os;
}

/// Common gpu id builder type, allows the configuration of lowering for various
/// mapping schemes. Takes:
///   - A rewriter with insertion point set before the forall op to rewrite.
///   - The loc of the forall op to rewrite.
///   - A list of positive integers carrying the mapping sizes for the current
///     forall op to rewrite.
using GpuIdBuilderFnType = std::function<IdBuilderResult(
    RewriterBase &, Location, ArrayRef<int64_t>, ArrayRef<int64_t>)>;

/// Helper struct for configuring the rewrite of mapped scf.forall ops to
/// various gpu id configurations.
struct GpuIdBuilder {
  using MappingIdBuilderFnType = std::function<DeviceMappingAttrInterface(
      MLIRContext *, mlir::gpu::MappingId)>;

  GpuIdBuilder() = default;
  GpuIdBuilder(MLIRContext *ctx, bool useLinearMapping,
               const MappingIdBuilderFnType &builder);

  /// The mapping attributes targeted by this generator.
  SmallVector<DeviceMappingAttrInterface> mappingAttributes;

  /// The constructor that builds the concrete IR for mapping ids.
  GpuIdBuilderFnType idBuilder;
};

/// Builder for gpu::BlockIdOps used to map scf.forall to blocks.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
/// If `mask` is provided, it will be used to filter the active blocks.
struct GpuBlockIdBuilder : public GpuIdBuilder {
  GpuBlockIdBuilder(MLIRContext *ctx, bool useLinearMapping = false,
                    DeviceMaskingAttrInterface mask = nullptr);
};

/// Builder for warpgroup ids used to map scf.forall to reindexed warpgroups.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
/// If `mask` is provided, it will be used to filter the active warpgroups.
struct GpuWarpgroupIdBuilder : public GpuIdBuilder {
  GpuWarpgroupIdBuilder(MLIRContext *ctx, int64_t warpSize,
                        bool useLinearMapping = false,
                        DeviceMaskingAttrInterface mask = nullptr);
  int64_t warpSize = 32;
  /// In the future this may be configured by the transformation.
  static constexpr int64_t kNumWarpsPerGroup = 4;
};

/// Builder for warp ids used to map scf.forall to reindexed warps.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
/// If `mask` is provided, it will be used to filter the active warps.
struct GpuWarpIdBuilder : public GpuIdBuilder {
  GpuWarpIdBuilder(MLIRContext *ctx, int64_t warpSize,
                   bool useLinearMapping = false,
                   DeviceMaskingAttrInterface mask = nullptr);
  int64_t warpSize = 32;
};

/// Builder for warp ids used to map scf.forall to reindexed threads.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
/// If `mask` is provided, it will be used to filter the active threads.
struct GpuThreadIdBuilder : public GpuIdBuilder {
  GpuThreadIdBuilder(MLIRContext *ctx, bool useLinearMapping = false,
                     DeviceMaskingAttrInterface mask = nullptr);
};

/// Builder for lane id.
/// The `idBuilder` method returns nD values used for indexing rewrites as well
/// as 1D sizes for predicate generation.
/// This `useLinearMapping` case is the only supported case.
/// If `mask` is provided, it will be used to filter the active lanes.
struct GpuLaneIdBuilder : public GpuIdBuilder {
  GpuLaneIdBuilder(MLIRContext *ctx, int64_t warpSize, bool unused,
                   DeviceMaskingAttrInterface mask = nullptr);
  int64_t warpSize = 32;
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
