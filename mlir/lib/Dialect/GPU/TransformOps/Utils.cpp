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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu;

#define DEBUG_TYPE "gpu-transforms"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGS_ALIAS() (llvm::dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")

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
  AffineExpr tx, ty, tz, BDX, BDY;
  bindDims(rewriter.getContext(), tx, ty, tz);
  bindSymbols(rewriter.getContext(), BDX, BDY);
  SmallVector<OpFoldResult> vals{
      rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::x)
          .getResult(),
      rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::y)
          .getResult(),
      rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::z)
          .getResult(),
      originalBasisOfr[0], originalBasisOfr[1]};
  OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
      rewriter, loc, tx + ty * BDX + tz * BDX * BDY, vals);
  return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
}

/// Create a linear id builder that takes the `originalBasisOfr` and decompose
/// it in the basis of `forallMappingSizes`. The linear id builder returns an
/// n-D vector of ids for indexing and 1-D size + id for predicate generation.
template <typename ThreadOrBlockIdOp>
static GpuIdBuilderFnType commonLinearIdBuilderFn(int64_t multiplicity = 1) {
  auto res = [multiplicity](RewriterBase &rewriter, Location loc,
                            ArrayRef<int64_t> forallMappingSizes,
                            ArrayRef<int64_t> originalBasis) {
    SmallVector<OpFoldResult> originalBasisOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), originalBasis);
    OpFoldResult linearId =
        buildLinearId<ThreadOrBlockIdOp>(rewriter, loc, originalBasisOfr);
    // Sizes in [0 .. n] -> [n .. 0] order to properly compute strides in
    // "row-major" order.
    SmallVector<int64_t> reverseBasisSizes(llvm::reverse(forallMappingSizes));
    SmallVector<int64_t> strides = computeStrides(reverseBasisSizes);
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    OpFoldResult scaledLinearId = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0.floorDiv(multiplicity), {linearId});
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
    SmallVector<Value> ids;
    // Reverse back to be in [0 .. n] order.
    for (AffineExpr e : llvm::reverse(delinearizingExprs)) {
      ids.push_back(
          affine::makeComposedAffineApply(rewriter, loc, e, {scaledLinearId}));
    }

    // clang-format off
      LLVM_DEBUG(llvm::interleaveComma(reverseBasisSizes,
                                       DBGS() << "--delinearization basis: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(strides,
                                       DBGS() << "--delinearization strides: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(delinearizingExprs,
                                       DBGS() << "--delinearization exprs: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(ids, DBGS() << "--ids: ");
                 llvm::dbgs() << "\n";);
    // clang-format on

    // Return n-D ids for indexing and 1-D size + id for predicate generation.
    return IdBuilderResult{
        /*mappingIdOps=*/ids,
        /*availableMappingSizes=*/
        SmallVector<int64_t>{computeProduct(originalBasis)},
        // `forallMappingSizes` iterate in the scaled basis, they need to be
        // scaled back into the original basis to provide tight
        // activeMappingSizes quantities for predication.
        /*activeMappingSizes=*/
        SmallVector<int64_t>{computeProduct(forallMappingSizes) * multiplicity},
        /*activeIdOps=*/SmallVector<Value>{linearId.get<Value>()}};
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
        rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::x),
        rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::y),
        rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::z)};
    // In the 3-D mapping case, scale the first dimension by the multiplicity.
    SmallVector<Value> scaledIds = ids;
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    scaledIds[0] = affine::makeComposedFoldedAffineApply(
                       rewriter, loc, d0.floorDiv(multiplicity), {scaledIds[0]})
                       .get<Value>();
    // In the 3-D mapping case, unscale the first dimension by the multiplicity.
    SmallVector<int64_t> forallMappingSizeInOriginalBasis(
        forallMappingSizes.begin(), forallMappingSizes.end());
    forallMappingSizeInOriginalBasis[0] *= multiplicity;
    return IdBuilderResult{
        /*mappingIdOps=*/scaledIds,
        /*availableMappingSizes=*/SmallVector<int64_t>{originalBasis},
        // `forallMappingSizes` iterate in the scaled basis, they need to be
        // scaled back into the original basis to provide tight
        // activeMappingSizes quantities for predication.
        /*activeMappingSizes=*/
        SmallVector<int64_t>{forallMappingSizeInOriginalBasis},
        /*activeIdOps=*/ids};
  };
  return res;
}

namespace mlir {
namespace transform {
namespace gpu {

GpuIdBuilder::GpuIdBuilder(MLIRContext *ctx, bool useLinearMapping,
                           MappingIdBuilderFnType fn)
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

GpuBlockIdBuilder::GpuBlockIdBuilder(MLIRContext *ctx, bool useLinearMapping)
    : GpuIdBuilder(ctx, useLinearMapping, [](MLIRContext *ctx, MappingId id) {
        return GPUBlockMappingAttr::get(ctx, id);
      }) {
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<BlockIdOp>(/*multiplicity=*/1)
                  : common3DIdBuilderFn<BlockIdOp>(/*multiplicity=*/1);
}

GpuWarpgroupIdBuilder::GpuWarpgroupIdBuilder(MLIRContext *ctx, int64_t warpSize,
                                             bool useLinearMapping)
    : GpuIdBuilder(ctx, useLinearMapping,
                   [](MLIRContext *ctx, MappingId id) {
                     return GPUWarpgroupMappingAttr::get(ctx, id);
                   }),
      warpSize(warpSize) {
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<ThreadIdOp>(
                        /*multiplicity=*/kNumWarpsPerGroup * warpSize)
                  : common3DIdBuilderFn<ThreadIdOp>(
                        /*multiplicity=*/kNumWarpsPerGroup * warpSize);
}

GpuWarpIdBuilder::GpuWarpIdBuilder(MLIRContext *ctx, int64_t warpSize,
                                   bool useLinearMapping)
    : GpuIdBuilder(ctx, useLinearMapping,
                   [](MLIRContext *ctx, MappingId id) {
                     return GPUWarpMappingAttr::get(ctx, id);
                   }),
      warpSize(warpSize) {
  idBuilder =
      useLinearMapping
          ? commonLinearIdBuilderFn<ThreadIdOp>(/*multiplicity=*/warpSize)
          : common3DIdBuilderFn<ThreadIdOp>(/*multiplicity=*/warpSize);
}

GpuThreadIdBuilder::GpuThreadIdBuilder(MLIRContext *ctx, bool useLinearMapping)
    : GpuIdBuilder(ctx, useLinearMapping, [](MLIRContext *ctx, MappingId id) {
        return GPUThreadMappingAttr::get(ctx, id);
      }) {
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<ThreadIdOp>(/*multiplicity=*/1)
                  : common3DIdBuilderFn<ThreadIdOp>(/*multiplicity=*/1);
}

DiagnosedSilenceableFailure checkGpuLimits(TransformOpInterface transformOp,
                                           std::optional<int64_t> gridDimX,
                                           std::optional<int64_t> gridDimY,
                                           std::optional<int64_t> gridDimZ,
                                           std::optional<int64_t> blockDimX,
                                           std::optional<int64_t> blockDimY,
                                           std::optional<int64_t> blockDimZ) {

  // TODO: pass a configuration object to set the limits properly.
  static constexpr int maxTotalBlockdim = 1024;
  static constexpr int maxBlockdimx = 1024;
  static constexpr int maxBlockdimy = 1024;
  static constexpr int maxBlockdimz = 64;
  static constexpr int maxTotalGriddim = 2147483647;
  static constexpr int maxGriddimx = 2147483647;
  static constexpr int maxGriddimy = 65535;
  static constexpr int maxGriddimz = 65535;

  if ((blockDimX.value_or(1) * blockDimY.value_or(1) * blockDimZ.value_or(1)) >
          maxTotalBlockdim ||
      (gridDimX.value_or(1) * gridDimY.value_or(1) * gridDimZ.value_or(1)) >
          maxTotalGriddim ||
      blockDimX.value_or(1) > maxBlockdimx ||
      blockDimY.value_or(1) > maxBlockdimy ||
      blockDimZ.value_or(1) > maxBlockdimz ||
      gridDimY.value_or(1) > maxGriddimy ||
      gridDimZ.value_or(1) > maxGriddimz ||
      gridDimX.value_or(1) > maxGriddimx) {
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
    return rewriter.create<arith::ConstantIndexOp>(loc, dim);
  };
  OpBuilder::InsertionGuard guard(rewriter);
  Value one = createConst(1);
  Value gridSizeX = gridDimX.has_value() ? createConst(gridDimX.value()) : one;
  Value gridSizeY = gridDimY.has_value() ? createConst(gridDimY.value()) : one;
  Value gridSizeZ = gridDimZ.has_value() ? createConst(gridDimZ.value()) : one;
  Value blkSizeX = blockDimX.has_value() ? createConst(blockDimX.value()) : one;
  Value blkSizeY = blockDimY.has_value() ? createConst(blockDimY.value()) : one;
  Value blkSizeZ = blockDimZ.has_value() ? createConst(blockDimZ.value()) : one;
  launchOp = rewriter.create<LaunchOp>(loc, gridSizeX, gridSizeY, gridSizeZ,
                                       blkSizeX, blkSizeY, blkSizeZ);
  rewriter.setInsertionPointToEnd(&launchOp.getBody().front());
  rewriter.create<TerminatorOp>(loc);
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
    return rewriter.create<arith::ConstantIndexOp>(currentBlockdim.x.getLoc(),
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
