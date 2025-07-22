//===- Utils.cpp - Utils for GPU transform ops ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/TransformOps/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu;

#define DEBUG_TYPE "gpu-transforms"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGS_ALIAS() (llvm::dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")

/// Build predicates to filter execution by only the activeIds. Along each
/// dimension, 3 cases appear:
///   1. activeMappingSize > availableMappingSize: this is an unsupported case
///      as this requires additional looping. An error message is produced to
///      advise the user to tile more or to use more threads.
///   2. activeMappingSize == availableMappingSize: no predication is needed.
///   3. activeMappingSize < availableMappingSize: only a subset of threads
///      should be active and we produce the boolean `id < activeMappingSize`
///      for further use in building predicated execution.
static FailureOr<SmallVector<Value>>
buildPredicates(RewriterBase &rewriter, Location loc, ArrayRef<Value> activeIds,
                ArrayRef<int64_t> activeMappingSizes,
                ArrayRef<int64_t> availableMappingSizes,
                std::string &errorMsg) {
  // clang-format off
  LLVM_DEBUG(
    llvm::interleaveComma(
      activeMappingSizes, DBGS() << "----activeMappingSizes: ");
    DBGS() << "\n";
    llvm::interleaveComma(
      availableMappingSizes, DBGS() << "----availableMappingSizes: ");
    DBGS() << "\n";);
  // clang-format on

  SmallVector<Value> predicateOps;
  for (auto [activeId, activeMappingSize, availableMappingSize] :
       llvm::zip_equal(activeIds, activeMappingSizes, availableMappingSizes)) {
    if (activeMappingSize > availableMappingSize) {
      errorMsg = "Trying to map to fewer GPU threads than loop iterations but "
                 "overprovisioning is not yet supported. Try additional tiling "
                 "before mapping or map to more threads.";
      return failure();
    }
    if (activeMappingSize == availableMappingSize)
      continue;
    Value idx =
        arith::ConstantIndexOp::create(rewriter, loc, activeMappingSize);
    Value pred = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                       activeId, idx);
    predicateOps.push_back(pred);
  }
  return predicateOps;
}

/// Return a flattened thread id for the workgroup with given sizes.
template <typename ThreadOrBlockIdOp>
static Value buildLinearId(RewriterBase &rewriter, Location loc,
                           ArrayRef<OpFoldResult> originalBasisOfr) {
  LLVM_DEBUG(llvm::interleaveComma(
                 originalBasisOfr,
                 DBGS() << "----buildLinearId with originalBasisOfr:  ");
             llvm::dbgs() << "\n");
  assert(originalBasisOfr.size() == 3 && "expected 3 sizes");
  IndexType indexType = rewriter.getIndexType();
  AffineExpr tx, ty, tz, bdx, bdy;
  bindDims(rewriter.getContext(), tx, ty, tz);
  bindSymbols(rewriter.getContext(), bdx, bdy);
  SmallVector<OpFoldResult> vals{
      ThreadOrBlockIdOp::create(rewriter, loc, indexType, Dimension::x)
          .getResult(),
      ThreadOrBlockIdOp::create(rewriter, loc, indexType, Dimension::y)
          .getResult(),
      ThreadOrBlockIdOp::create(rewriter, loc, indexType, Dimension::z)
          .getResult(),
      originalBasisOfr[0], originalBasisOfr[1]};
  OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
      rewriter, loc, tx + ty * bdx + tz * bdx * bdy, vals);
  return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
}

/// Create a linear id builder that takes the `originalBasisOfr` and decompose
/// it in the basis of `forallMappingSizes`. The linear id builder returns an
/// n-D vector of ids for indexing and 1-D size + id for predicate generation.
template <typename ThreadOrBlockIdOp>
static GpuIdBuilderFnType
commonLinearIdBuilderFn(int64_t multiplicity = 1,
                        DeviceMaskingAttrInterface mask = nullptr) {
  auto res = [multiplicity, mask](RewriterBase &rewriter, Location loc,
                                  ArrayRef<int64_t> forallMappingSizes,
                                  ArrayRef<int64_t> originalBasis) {
    // 0. Early-exit mask case.
    if (mask) {
      if (computeProduct(originalBasis) >
          mask.getMaxNumPhysicalIds() * multiplicity) {
        return IdBuilderResult{
            /*errorMsg=*/std::string(
                "mask representation too short to capture all physical ids: ") +
                std::to_string(mask.getMaxNumPhysicalIds()),
            /*mappingIdOps=*/{},
            /*predicateOps=*/{}};
      }
    }

    // 1. Compute linearId.
    SmallVector<OpFoldResult> originalBasisOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), originalBasis);
    Value physicalLinearId =
        buildLinearId<ThreadOrBlockIdOp>(rewriter, loc, originalBasisOfr);

    // 2. Compute scaledLinearId.
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    OpFoldResult scaledLinearIdOfr = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0.floorDiv(multiplicity), {physicalLinearId});

    // 2.b. Adjust with mask if needed.
    Value scaledLinearIdI64;
    Value scaledLinearId =
        getValueOrCreateConstantIndexOp(rewriter, loc, scaledLinearIdOfr);
    if (mask) {
      scaledLinearId =
          getValueOrCreateConstantIndexOp(rewriter, loc, scaledLinearIdOfr);
      scaledLinearIdI64 = arith::IndexCastUIOp::create(
          rewriter, loc, rewriter.getI64Type(), scaledLinearId);
      Value logicalLinearIdI64 =
          mask.createLogicalLinearMappingId(rewriter, scaledLinearIdI64);
      scaledLinearId = arith::IndexCastUIOp::create(
          rewriter, loc, rewriter.getIndexType(), logicalLinearIdI64);
      LDBG("------adjusting linearId with mask: " << scaledLinearId);
    }

    // 3. Compute remapped indices.
    SmallVector<Value> ids;
    // Sizes in [0 .. n] -> [n .. 0] order to properly compute strides in
    // "row-major" order.
    SmallVector<int64_t> reverseBasisSizes(llvm::reverse(forallMappingSizes));
    SmallVector<int64_t> strides = computeStrides(reverseBasisSizes);
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
    // Reverse back to be in [0 .. n] order.
    for (AffineExpr e : llvm::reverse(delinearizingExprs)) {
      ids.push_back(
          affine::makeComposedAffineApply(rewriter, loc, e, {scaledLinearId}));
    }

    std::string errorMsg;
    SmallVector<Value> predicateOps;
    // 4. If mask present, it takes precedence to determine predication.
    if (mask) {
      Value isActiveIdPredicate =
          mask.createIsActiveIdPredicate(rewriter, scaledLinearIdI64);
      LDBG("------adjusting predicate with mask: " << isActiveIdPredicate);
      predicateOps.push_back(isActiveIdPredicate);
    } else {
      // 4.b. Otherwise, handle predicates using physicalLinearId.
      FailureOr<SmallVector<Value>> maybePredicateOps =
          buildPredicates(rewriter, loc, physicalLinearId,
                          computeProduct(forallMappingSizes) * multiplicity,
                          computeProduct(originalBasis), errorMsg);
      if (succeeded(maybePredicateOps))
        predicateOps = *maybePredicateOps;
    }

    return IdBuilderResult{/*errorMsg=*/errorMsg,
                           /*mappingIdOps=*/ids,
                           /*predicateOps=*/predicateOps};
  };

  return res;
}

/// Create a simple 3-D id builder that takes the `originalBasisOfr`
/// The 3-D id builder returns a 3-D vector of ids for indexing and 3-D sizes
/// + ids for predicate generation.
template <typename ThreadOrBlockIdOp>
static GpuIdBuilderFnType common3DIdBuilderFn(int64_t multiplicity = 1) {
  auto res = [multiplicity](RewriterBase &rewriter, Location loc,
                            ArrayRef<int64_t> forallMappingSizes,
                            ArrayRef<int64_t> originalBasis) {
    IndexType indexType = rewriter.getIndexType();
    SmallVector<Value> ids{
        ThreadOrBlockIdOp::create(rewriter, loc, indexType, Dimension::x),
        ThreadOrBlockIdOp::create(rewriter, loc, indexType, Dimension::y),
        ThreadOrBlockIdOp::create(rewriter, loc, indexType, Dimension::z)};
    // In the 3-D mapping case, scale the first dimension by the multiplicity.
    SmallVector<Value> scaledIds = ids;
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    scaledIds[0] = cast<Value>(affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0.floorDiv(multiplicity), {scaledIds[0]}));
    // In the 3-D mapping case, unscale the first dimension by the multiplicity.
    SmallVector<int64_t> forallMappingSizeInOriginalBasis(forallMappingSizes);
    forallMappingSizeInOriginalBasis[0] *= multiplicity;

    std::string errorMsg;
    SmallVector<Value> predicateOps;
    FailureOr<SmallVector<Value>> maybePredicateOps =
        buildPredicates(rewriter, loc, ids, forallMappingSizeInOriginalBasis,
                        originalBasis, errorMsg);
    if (succeeded(maybePredicateOps))
      predicateOps = *maybePredicateOps;

    return IdBuilderResult{/*errorMsg=*/errorMsg,
                           /*mappingIdOps=*/scaledIds,
                           /*predicateOps=*/predicateOps};
  };
  return res;
}

/// Create a lane id builder that takes the `originalBasis` and decompose
/// it in the basis of `forallMappingSizes`. The linear id builder returns an
/// n-D vector of ids for indexing and 1-D size + id for predicate generation.
static GpuIdBuilderFnType laneIdBuilderFn(int64_t warpSize) {
  auto res = [warpSize](RewriterBase &rewriter, Location loc,
                        ArrayRef<int64_t> forallMappingSizes,
                        ArrayRef<int64_t> originalBasis) {
    // 1. Compute linearId.
    SmallVector<OpFoldResult> originalBasisOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), originalBasis);
    Value physicalLinearId =
        buildLinearId<ThreadIdOp>(rewriter, loc, originalBasisOfr);

    // 2. Compute laneId.
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    OpFoldResult laneId = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0 % warpSize, {physicalLinearId});

    // 3. Compute remapped indices.
    SmallVector<Value> ids;
    // Sizes in [0 .. n] -> [n .. 0] order to properly compute strides in
    // "row-major" order.
    SmallVector<int64_t> reverseBasisSizes(llvm::reverse(forallMappingSizes));
    SmallVector<int64_t> strides = computeStrides(reverseBasisSizes);
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
    // Reverse back to be in [0 .. n] order.
    for (AffineExpr e : llvm::reverse(delinearizingExprs)) {
      ids.push_back(
          affine::makeComposedAffineApply(rewriter, loc, e, {laneId}));
    }

    // 4. Handle predicates using laneId.
    std::string errorMsg;
    SmallVector<Value> predicateOps;
    FailureOr<SmallVector<Value>> maybePredicateOps = buildPredicates(
        rewriter, loc, cast<Value>(laneId), computeProduct(forallMappingSizes),
        computeProduct(originalBasis), errorMsg);
    if (succeeded(maybePredicateOps))
      predicateOps = *maybePredicateOps;

    return IdBuilderResult{/*errorMsg=*/errorMsg,
                           /*mappingIdOps=*/ids,
                           /*predicateOps=*/predicateOps};
  };

  return res;
}

namespace mlir {
namespace transform {
namespace gpu {

GpuIdBuilder::GpuIdBuilder(MLIRContext *ctx, bool useLinearMapping,
                           const MappingIdBuilderFnType &fn)
    : mappingAttributes(), idBuilder() {
  if (useLinearMapping) {
    for (uint64_t d = static_cast<uint64_t>(MappingId::LinearDim0),
                  e = getMaxEnumValForMappingId();
         d <= e; ++d)
      mappingAttributes.push_back(fn(ctx, symbolizeMappingId(d).value()));
  } else {
    for (uint64_t d = static_cast<uint64_t>(MappingId::DimX),
                  e = static_cast<uint64_t>(MappingId::DimZ);
         d <= e; ++d)
      mappingAttributes.push_back(fn(ctx, symbolizeMappingId(d).value()));
  }
}

GpuBlockIdBuilder::GpuBlockIdBuilder(MLIRContext *ctx, bool useLinearMapping,
                                     DeviceMaskingAttrInterface mask)
    : GpuIdBuilder(ctx, useLinearMapping, [](MLIRContext *ctx, MappingId id) {
        return GPUBlockMappingAttr::get(ctx, id);
      }) {
  assert((!mask || useLinearMapping) && "mask requires linear mapping");
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<BlockIdOp>(/*multiplicity=*/1, mask)
                  : common3DIdBuilderFn<BlockIdOp>(/*multiplicity=*/1);
}

GpuWarpgroupIdBuilder::GpuWarpgroupIdBuilder(MLIRContext *ctx, int64_t warpSize,
                                             bool useLinearMapping,
                                             DeviceMaskingAttrInterface mask)
    : GpuIdBuilder(ctx, useLinearMapping,
                   [](MLIRContext *ctx, MappingId id) {
                     return GPUWarpgroupMappingAttr::get(ctx, id);
                   }),
      warpSize(warpSize) {
  assert((!mask || useLinearMapping) && "mask requires linear mapping");
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<ThreadIdOp>(
                        /*multiplicity=*/kNumWarpsPerGroup * warpSize, mask)
                  : common3DIdBuilderFn<ThreadIdOp>(
                        /*multiplicity=*/kNumWarpsPerGroup * warpSize);
}

GpuWarpIdBuilder::GpuWarpIdBuilder(MLIRContext *ctx, int64_t warpSize,
                                   bool useLinearMapping,
                                   DeviceMaskingAttrInterface mask)
    : GpuIdBuilder(ctx, useLinearMapping,
                   [](MLIRContext *ctx, MappingId id) {
                     return GPUWarpMappingAttr::get(ctx, id);
                   }),
      warpSize(warpSize) {
  assert((!mask || useLinearMapping) && "mask requires linear mapping");
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<ThreadIdOp>(
                        /*multiplicity=*/warpSize, mask)
                  : common3DIdBuilderFn<ThreadIdOp>(/*multiplicity=*/warpSize);
}

GpuThreadIdBuilder::GpuThreadIdBuilder(MLIRContext *ctx, bool useLinearMapping,
                                       DeviceMaskingAttrInterface mask)
    : GpuIdBuilder(ctx, useLinearMapping, [](MLIRContext *ctx, MappingId id) {
        return GPUThreadMappingAttr::get(ctx, id);
      }) {
  idBuilder =
      useLinearMapping
          ? commonLinearIdBuilderFn<ThreadIdOp>(/*multiplicity=*/1, mask)
          : common3DIdBuilderFn<ThreadIdOp>(/*multiplicity=*/1);
}

GpuLaneIdBuilder::GpuLaneIdBuilder(MLIRContext *ctx, int64_t warpSize,
                                   bool unused, DeviceMaskingAttrInterface mask)
    : GpuIdBuilder(ctx, /*useLinearMapping=*/true,
                   [](MLIRContext *ctx, MappingId id) {
                     return GPULaneMappingAttr::get(ctx, id);
                   }),
      warpSize(warpSize) {
  assert(!mask && "mask NYI for lanes, unclear it should be at all");
  idBuilder = laneIdBuilderFn(/*periodicity=*/warpSize);
}

DiagnosedSilenceableFailure checkGpuLimits(TransformOpInterface transformOp,
                                           std::optional<int64_t> gridDimX,
                                           std::optional<int64_t> gridDimY,
                                           std::optional<int64_t> gridDimZ,
                                           std::optional<int64_t> blockDimX,
                                           std::optional<int64_t> blockDimY,
                                           std::optional<int64_t> blockDimZ) {

  // TODO: pass a configuration object to set the limits properly.

  if ((blockDimX.value_or(1) * blockDimY.value_or(1) * blockDimZ.value_or(1)) >
          kMaxTotalBlockdim ||
      (gridDimX.value_or(1) * gridDimY.value_or(1) * gridDimZ.value_or(1)) >
          kMaxTotalGriddim ||
      blockDimX.value_or(1) > kMaxBlockdimx ||
      blockDimY.value_or(1) > kMaxBlockdimy ||
      blockDimZ.value_or(1) > kMaxBlockdimz ||
      gridDimY.value_or(1) > kMaxGriddimy ||
      gridDimZ.value_or(1) > kMaxGriddimz ||
      gridDimX.value_or(1) > kMaxGriddimx) {
    return transformOp.emitSilenceableError()
           << "Trying to launch a GPU kernel with grid_dims = ("
           << gridDimX.value_or(1) << ", " << gridDimY.value_or(1) << ", "
           << gridDimZ.value_or(1) << ") block_dims = ("
           << blockDimX.value_or(1) << ", " << blockDimY.value_or(1) << ", "
           << blockDimZ.value_or(1) << "). It is larger than the limits.";
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure createGpuLaunch(
    RewriterBase &rewriter, Location loc, TransformOpInterface transformOp,
    LaunchOp &launchOp, std::optional<int64_t> gridDimX,
    std::optional<int64_t> gridDimY, std::optional<int64_t> gridDimZ,
    std::optional<int64_t> blockDimX, std::optional<int64_t> blockDimY,
    std::optional<int64_t> blockDimZ) {
  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, gridDimX, gridDimY, gridDimZ, blockDimX,
                     blockDimY, blockDimZ);
  if (!diag.succeeded())
    return diag;

  auto createConst = [&](int dim) {
    return arith::ConstantIndexOp::create(rewriter, loc, dim);
  };
  OpBuilder::InsertionGuard guard(rewriter);
  Value one = createConst(1);
  Value gridSizeX = gridDimX.has_value() ? createConst(gridDimX.value()) : one;
  Value gridSizeY = gridDimY.has_value() ? createConst(gridDimY.value()) : one;
  Value gridSizeZ = gridDimZ.has_value() ? createConst(gridDimZ.value()) : one;
  Value blkSizeX = blockDimX.has_value() ? createConst(blockDimX.value()) : one;
  Value blkSizeY = blockDimY.has_value() ? createConst(blockDimY.value()) : one;
  Value blkSizeZ = blockDimZ.has_value() ? createConst(blockDimZ.value()) : one;
  launchOp = LaunchOp::create(rewriter, loc, gridSizeX, gridSizeY, gridSizeZ,
                              blkSizeX, blkSizeY, blkSizeZ);
  rewriter.setInsertionPointToEnd(&launchOp.getBody().front());
  TerminatorOp::create(rewriter, loc);
  return DiagnosedSilenceableFailure::success();
}

/// Alter kernel configuration of the given kernel.
DiagnosedSilenceableFailure alterGpuLaunch(
    RewriterBase &rewriter, LaunchOp gpuLaunch,
    TransformOpInterface transformOp, std::optional<int64_t> gridDimX,
    std::optional<int64_t> gridDimY, std::optional<int64_t> gridDimZ,
    std::optional<int64_t> blockDimX, std::optional<int64_t> blockDimY,
    std::optional<int64_t> blockDimZ) {
  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, gridDimX, gridDimY, gridDimZ, blockDimX,
                     blockDimY, blockDimZ);
  if (!diag.succeeded())
    return diag;

  KernelDim3 currentBlockdim = gpuLaunch.getBlockSizeOperandValues();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(currentBlockdim.x);
  auto createConstValue = [&](int dim) {
    return arith::ConstantIndexOp::create(rewriter, currentBlockdim.x.getLoc(),
                                          dim);
  };

  if (gridDimX.has_value())
    gpuLaunch.getGridSizeXMutable().assign(createConstValue(gridDimX.value()));
  if (gridDimY.has_value())
    gpuLaunch.getGridSizeYMutable().assign(createConstValue(gridDimY.value()));
  if (gridDimZ.has_value())
    gpuLaunch.getGridSizeZMutable().assign(createConstValue(gridDimZ.value()));
  if (blockDimX.has_value())
    gpuLaunch.getBlockSizeXMutable().assign(
        createConstValue(blockDimX.value()));
  if (blockDimY.has_value())
    gpuLaunch.getBlockSizeYMutable().assign(
        createConstValue(blockDimY.value()));
  if (blockDimZ.has_value())
    gpuLaunch.getBlockSizeZMutable().assign(
        createConstValue(blockDimZ.value()));
  return DiagnosedSilenceableFailure::success();
}

} // namespace gpu
} // namespace transform
} // namespace mlir
