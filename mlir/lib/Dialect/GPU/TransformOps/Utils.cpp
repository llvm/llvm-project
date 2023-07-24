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
static Value buildLinearThreadId(RewriterBase &rewriter, Location loc,
                                 ArrayRef<OpFoldResult> blockDimsOfr) {
  LLVM_DEBUG(llvm::interleaveComma(
                 blockDimsOfr,
                 DBGS() << "----buildLinearThreadId with blockDimsOfr:  ");
             llvm::dbgs() << "\n");
  assert(blockDimsOfr.size() == 3 && "expected 3 workgroup sizes");
  AffineExpr tx, ty, tz, BDX, BDY;
  bindDims(rewriter.getContext(), tx, ty, tz);
  bindSymbols(rewriter.getContext(), BDX, BDY);
  IndexType indexType = rewriter.getIndexType();
  SmallVector<OpFoldResult> threadsAndWorkGroups{
      rewriter.create<ThreadIdOp>(loc, indexType, Dimension::x).getResult(),
      rewriter.create<ThreadIdOp>(loc, indexType, Dimension::y).getResult(),
      rewriter.create<ThreadIdOp>(loc, indexType, Dimension::z).getResult()};
  threadsAndWorkGroups.push_back(blockDimsOfr[0]);
  threadsAndWorkGroups.push_back(blockDimsOfr[1]);
  OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
      rewriter, loc, tx + ty * BDX + tz * BDX * BDY, threadsAndWorkGroups);
  return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
}

namespace mlir {
namespace transform {
namespace gpu {

GpuBlockIdBuilder::GpuBlockIdBuilder(MLIRContext *ctx,
                                     ArrayRef<OpFoldResult> blockDims,
                                     ArrayRef<int64_t> mappingSizes)
    : GpuIdBuilder(blockDims, mappingSizes) {
  mappingAttributes = {GPUBlockMappingAttr::get(ctx, Blocks::DimX),
                       GPUBlockMappingAttr::get(ctx, Blocks::DimY),
                       GPUBlockMappingAttr::get(ctx, Blocks::DimZ)},
  idBuilder = [](RewriterBase &rewriter, Location loc,
                 ArrayRef<int64_t> forallMappingSizes) {
    IndexType indexType = rewriter.getIndexType();
    SmallVector<Value> ids{
        rewriter.create<BlockIdOp>(loc, indexType, Dimension::x),
        rewriter.create<BlockIdOp>(loc, indexType, Dimension::y),
        rewriter.create<BlockIdOp>(loc, indexType, Dimension::z)};
    // Return 3-D ids for indexing rewrite and 3-D sizes and ids for
    // predicate generation.
    return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes}, ids};
  };
}

GpuThreadIdBuilder::GpuThreadIdBuilder(MLIRContext *ctx,
                                       ArrayRef<OpFoldResult> blockDims,
                                       ArrayRef<int64_t> mappingSizes)
    : GpuIdBuilder(blockDims, mappingSizes) {
  mappingAttributes = {GPUThreadMappingAttr::get(ctx, Threads::DimX),
                       GPUThreadMappingAttr::get(ctx, Threads::DimY),
                       GPUThreadMappingAttr::get(ctx, Threads::DimZ)};
  idBuilder = [](RewriterBase &rewriter, Location loc,
                 ArrayRef<int64_t> forallMappingSizes) {
    IndexType indexType = rewriter.getIndexType();
    SmallVector<Value> ids{
        rewriter.create<ThreadIdOp>(loc, indexType, Dimension::x),
        rewriter.create<ThreadIdOp>(loc, indexType, Dimension::y),
        rewriter.create<ThreadIdOp>(loc, indexType, Dimension::z)};
    // Return 3-D ids for indexing rewrite and 3-D sizes and ids for
    // predicate generation.
    return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes}, ids};
  };
}

GpuWarpIdBuilder::GpuWarpIdBuilder(MLIRContext *ctx,
                                   ArrayRef<OpFoldResult> blockDims,
                                   ArrayRef<int64_t> mappingSizes)
    : GpuIdBuilder(blockDims, mappingSizes) {
  mappingAttributes = {GPUWarpMappingAttr::get(ctx, Warps::DimX),
                       GPUWarpMappingAttr::get(ctx, Warps::DimY),
                       GPUWarpMappingAttr::get(ctx, Warps::DimZ)};
  idBuilder = [this](RewriterBase &rewriter, Location loc,
                     ArrayRef<int64_t> forallMappingSizes) {
    // Build the linear warp id and decompose it in the basis of
    // `forallMappingSizes`.
    Value linearId = buildLinearThreadId(rewriter, loc, this->blockDimsOfr);
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    OpFoldResult warpIdOfr = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0.floorDiv(kWarpSize), {linearId});
    Value warpId = getValueOrCreateConstantIndexOp(rewriter, loc, warpIdOfr);
    // Sizes in [x, y, z] -> [z, y x] order to properly compute strides in
    // "row-major" order.
    SmallVector<int64_t> reverseBasisSizes(
        llvm::reverse(this->availableMappingSizes));
    SmallVector<int64_t> strides = computeStrides(reverseBasisSizes);
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
    SmallVector<Value> ids;
    // Reverse back to be in [x, y, z] order.
    for (AffineExpr e : llvm::reverse(delinearizingExprs))
      ids.push_back(
          affine::makeComposedAffineApply(rewriter, loc, e, {warpId}));

    // clang-format off
      LDBG("----linearId: " << linearId);
          LDBG("----warpId: " << warpId);
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

    // Return 3-D ids for indexing rewrite and 3-D sizes and ids for
    // predicate generation.
    return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes}, ids};
  };
}

GpuLinearIdBuilder::GpuLinearIdBuilder(MLIRContext *ctx,
                                       ArrayRef<OpFoldResult> blockDims,
                                       ArrayRef<int64_t> mappingSizes)
    : GpuIdBuilder(blockDims, mappingSizes) {
  mappingAttributes = {GPULinearIdMappingAttr::get(ctx, LinearId::DimX),
                       GPULinearIdMappingAttr::get(ctx, LinearId::DimY),
                       GPULinearIdMappingAttr::get(ctx, LinearId::DimZ)};
  idBuilder = [this](RewriterBase &rewriter, Location loc,
                     ArrayRef<int64_t> forallMappingSizes) {
    // Build the linear thread id and decompose it in the basis of
    // `forallMappingSizes`.
    Value linearId = buildLinearThreadId(rewriter, loc, this->blockDimsOfr);
    // Sizes in [x, y, z] -> [z, y x] order to properly compute strides in
    // "row-major" order.
    SmallVector<int64_t> reverseBasisSizes(llvm::reverse(forallMappingSizes));
    SmallVector<int64_t> strides = computeStrides(reverseBasisSizes);
    AffineExpr d0;
    bindDims(rewriter.getContext(), d0);
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
    SmallVector<Value> ids;
    // Reverse back to be in [x, y, z] order.
    for (AffineExpr e : llvm::reverse(delinearizingExprs))
      ids.push_back(
          affine::makeComposedAffineApply(rewriter, loc, e, {linearId}));

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

    // Compute and return the 1-D actual mapping size spanned by the linearId,
    // it will be used to predicate against the linearized total number of
    // threads.
    int64_t actualMappingSize = 1;
    for (int64_t s : forallMappingSizes)
      actualMappingSize *= s;

    // Return 3-D ids for indexing rewrite and 1-D size and id for
    // predicate generation.
    return IdBuilderResult{ids, SmallVector<int64_t>{actualMappingSize},
                           SmallVector<Value>{linearId}};
  };
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

DiagnosedSilenceableFailure
findTopLevelForallOp(Operation *target, scf::ForallOp &topLevelForallOp,
                     TransformOpInterface transformOp) {
  auto walkResult = target->walk([&](scf::ForallOp forallOp) {
    if (forallOp->getParentOfType<scf::ForallOp>())
      return WalkResult::advance();
    if (topLevelForallOp)
      // TODO: Handle multiple forall if they are independent.
      return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return transformOp.emitSilenceableError()
           << "could not find a unique topLevel scf.forall";
  return DiagnosedSilenceableFailure::success();
}

} // namespace gpu
} // namespace transform
} // namespace mlir
