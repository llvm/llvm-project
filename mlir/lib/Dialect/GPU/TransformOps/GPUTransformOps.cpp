//===- GPUTransformOps.cpp - Implementation of GPU transform ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu;

#define DEBUG_TYPE "gpu-transforms"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

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

/// Builder for gpu::BlockIdOps used in mapping scf.forall to blocks.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuBlockIdBuilder : public GpuIdBuilder {

  GpuBlockIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
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
      return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes},
                             ids};
    };
  }
};

/// Builder for gpu::ThreadIdOp used in mapping scf.forall to thread ids without
/// any reindexing.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuThreadIdBuilder : public GpuIdBuilder {
  GpuThreadIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
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
      return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes},
                             ids};
    };
  }
};

/// Builder for warp ids used in mapping scf.forall to warps.
/// This builder requires a specification of the number of warps along each
/// dimension to more finely control mapping to warps as well a predication than
/// by solely analyzing the IR.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuWarpIdBuilder : public GpuIdBuilder {
  GpuWarpIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
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
            affine::makeComposedAffineApply(rewriter, loc, e, warpId));

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
      return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes},
                             ids};
    };
  }

  /// Static specification of the warp size.
  /// In the future this may be configured by the transformation.
  static constexpr int64_t kWarpSize = 32;
};

/// Builder for linear ids used in mapping scf.forall to reindexed threads.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 1-D sizes for predicate generation.
struct GpuLinearIdBuilder : public GpuIdBuilder {
  GpuLinearIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
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
            affine::makeComposedAffineApply(rewriter, loc, e, linearId));

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
};

} // namespace

static DiagnosedSilenceableFailure
definiteFailureHelper(std::optional<TransformOpInterface> transformOp,
                      Operation *target, const Twine &message) {
  if (transformOp.has_value())
    return transformOp->emitDefiniteFailure() << message;
  return emitDefiniteFailure(target, message);
}

/// Check if given mapping attributes are one of the desired attributes
static DiagnosedSilenceableFailure
checkMappingAttributeTypes(std::optional<TransformOpInterface> transformOp,
                           scf::ForallOp forallOp) {
  if (!forallOp.getMapping().has_value())
    return definiteFailureHelper(transformOp, forallOp,
                                 "mapping must be present");

  bool hasBlockMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUBlockMappingAttr>(attr);
      });
  bool hasThreadMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUThreadMappingAttr>(attr);
      });
  bool hasWarpMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUWarpMappingAttr>(attr);
      });
  bool hasLinearMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPULinearIdMappingAttr>(attr);
      });
  int64_t countMappingTypes = 0;
  countMappingTypes += hasBlockMapping ? 1 : 0;
  countMappingTypes += hasThreadMapping ? 1 : 0;
  countMappingTypes += hasWarpMapping ? 1 : 0;
  countMappingTypes += hasLinearMapping ? 1 : 0;
  if (countMappingTypes > 1) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix different mapping types, use nesting");
  }

  DenseSet<Attribute> seen;
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (seen.contains(map)) {
      return definiteFailureHelper(
          transformOp, forallOp,
          "duplicated attribute, cannot map different loops "
          "to the same processor");
    }
    seen.insert(map);
  }

  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
verifyGpuMapping(std::optional<TransformOpInterface> transformOp,
                 scf::ForallOp forallOp) {
  // Check the types of the mapping attributes match.
  DiagnosedSilenceableFailure typeRes =
      checkMappingAttributeTypes(transformOp, forallOp);
  if (!typeRes.succeeded())
    return typeRes;

  // Perform other non-types verifications.
  if (!forallOp.isNormalized())
    return definiteFailureHelper(transformOp, forallOp,
                                 "unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return definiteFailureHelper(transformOp, forallOp,
                                 "only bufferized scf.forall can be mapped");
  if (forallOp.getRank() > 3)
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall with rank > 3 does not lower");
  if (llvm::any_of(forallOp.getMixedUpperBound(), [&](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "unsupported dynamic sizes in forall op");
  }
  return DiagnosedSilenceableFailure::success();
}

/// Determines if the size of the kernel configuration is supported by the
/// GPU architecture being used. It presently makes use of CUDA limitations,
/// however that aspect may be enhanced for other GPUs.
static DiagnosedSilenceableFailure checkGpuLimits(
    TransformOpInterface transformOp, std::optional<int64_t> gridDimX,
    std::optional<int64_t> gridDimY, std::optional<int64_t> gridDimZ,
    std::optional<int64_t> blockDimX, std::optional<int64_t> blockDimY,
    std::optional<int64_t> blockDimZ) {

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

/// Creates an empty-body gpu::LaunchOp using the provided kernel settings
/// and put a terminator within.
static DiagnosedSilenceableFailure
createGpuLaunch(RewriterBase &rewriter, Location loc,
                TransformOpInterface transformOp, LaunchOp &launchOp,
                std::optional<int64_t> gridDimX = std::nullopt,
                std::optional<int64_t> gridDimY = std::nullopt,
                std::optional<int64_t> gridDimZ = std::nullopt,
                std::optional<int64_t> blockDimX = std::nullopt,
                std::optional<int64_t> blockDimY = std::nullopt,
                std::optional<int64_t> blockDimZ = std::nullopt) {
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
static DiagnosedSilenceableFailure
alterGpuLaunch(IRRewriter &rewriter, LaunchOp gpuLaunch,
               TransformOpInterface transformOp,
               std::optional<int64_t> gridDimX = std::nullopt,
               std::optional<int64_t> gridDimY = std::nullopt,
               std::optional<int64_t> gridDimZ = std::nullopt,
               std::optional<int64_t> blockDimX = std::nullopt,
               std::optional<int64_t> blockDimY = std::nullopt,
               std::optional<int64_t> blockDimZ = std::nullopt) {
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

/// Struct to return the result of the rewrite of a forall operation.
struct ForallRewriteResult {
  SmallVector<int64_t> mappingSizes;
  SmallVector<Value> mappingIds;
};

/// Helper to replace ids of dimensions known to be 1 by 0 to simplify the IR.
template <typename OpTy, typename OperationOrBlock>
static void
replaceUnitMappingIdsHelper(RewriterBase &rewriter, Location loc,
                            OperationOrBlock *parent, Value replacement,
                            ArrayRef<int64_t> availableMappingSizes) {
  parent->walk([&](OpTy idOp) {
    if (availableMappingSizes[static_cast<int64_t>(idOp.getDimension())] == 1)
      rewriter.replaceAllUsesWith(idOp.getResult(), replacement);
  });
}

static DiagnosedSilenceableFailure rewriteOneForallCommonImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ForallRewriteResult &result,
    ArrayRef<int64_t> availableMappingSizes, const GpuIdBuilder &gpuIdBuilder) {
  LDBG("--start rewriteOneForallCommonImpl");

  // Step 0. GPU-specific verifications. There is no better place to anchor
  // those right now: the ForallOp is target-independent and the transform
  // op does not apply to individual ForallOp.
  DiagnosedSilenceableFailure diag = verifyGpuMapping(transformOp, forallOp);
  if (!diag.succeeded())
    return diag;

  // Step 1. Complete the mapping to a full mapping (with 1s) if necessary.
  SmallVector<int64_t> tmpMappingSizes = llvm::to_vector(
      llvm::map_range(forallOp.getMixedUpperBound(), [](OpFoldResult ofr) {
        auto maybeStaticValue = getConstantIntValue(ofr);
        assert(maybeStaticValue && "expected static value");
        return maybeStaticValue.value();
      }));
  SmallVector<Attribute> forallMappingAttrs =
      llvm::to_vector(forallOp.getMapping()->getValue());
  for (auto attr : gpuIdBuilder.mappingAttributes) {
    if (llvm::is_contained(forallMappingAttrs, attr))
      continue;
    forallMappingAttrs.push_back(attr);
    tmpMappingSizes.push_back(1);
  }
  LLVM_DEBUG(
      llvm::interleaveComma(
          tmpMappingSizes,
          DBGS() << "----tmpMappingSizes extracted from scf.forall op: ");
      llvm::dbgs() << "\n");

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  auto comparator = [&](Attribute a, Attribute b) -> bool {
    return cast<DeviceMappingAttrInterface>(a).getMappingId() <
           cast<DeviceMappingAttrInterface>(b).getMappingId();
  };
  SmallVector<int64_t> forallMappingSizes =
      getValuesSortedByKey(forallMappingAttrs, tmpMappingSizes, comparator);
  LLVM_DEBUG(llvm::interleaveComma(forallMappingSizes,
                                   DBGS() << "----forallMappingSizes: ");
             llvm::dbgs() << "\n"; llvm::interleaveComma(
                 forallMappingAttrs, DBGS() << "----mappingAttrs: ");
             llvm::dbgs() << "\n");

  // Step 3. Generate the mappingIdOps using the provided generator.
  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  IdBuilderResult builderResult =
      gpuIdBuilder.idBuilder(rewriter, loc, forallMappingSizes);

  // Step 4. Map the induction variables to the mappingIdOps, this may involve a
  // permutation.
  SmallVector<Value> mappingIdOps = builderResult.mappingIdOps;
  IRMapping bvm;
  for (auto [iv, dim] :
       llvm::zip_equal(forallOp.getInductionVars(),
                       ArrayRef<Attribute>{forallMappingAttrs}.take_front(
                           forallOp.getInductionVars().size()))) {
    Value peIdOp = mappingIdOps[static_cast<int64_t>(
        cast<DeviceMappingAttrInterface>(dim).getMappingId())];
    bvm.map(iv, peIdOp);
  }

  // Step 5. If the availableMappingSizes are already known, create conditionals
  // to predicate the region. Otherwise, the current forall determines the
  // availableMappingSizes and no predication occurs.
  Value predicate;
  if (!availableMappingSizes.empty()) {
    SmallVector<int64_t> predicateMappingSizes =
        builderResult.predicateMappingSizes;
    SmallVector<Value> predicateIdOps = builderResult.predicateIdOps;
    // clang-format off
    LLVM_DEBUG(
        llvm::interleaveComma(
          predicateMappingSizes, DBGS() << "----predicateMappingSizes: ");
        llvm::dbgs() << "\n"; 
        llvm::interleaveComma(
          availableMappingSizes, DBGS() << "----availableMappingSizes: ");
        llvm::dbgs() << "\n";
        llvm::interleaveComma(predicateIdOps, DBGS() << "----predicateIdOps: ");
        llvm::dbgs() << "\n");
    // clang-format on
    for (auto [id, mappingSize, availableMappingSize] : llvm::zip_equal(
             predicateIdOps, predicateMappingSizes, availableMappingSizes)) {
      if (mappingSize > availableMappingSize) {
        return definiteFailureHelper(
            transformOp, forallOp,
            "Trying to map to fewer GPU threads than loop iterations but "
            "overprovisioning is not yet supported. "
            "Try additional tiling of the before mapping or map to more "
            "threads.");
      }
      if (mappingSize == availableMappingSize)
        continue;
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, mappingSize);
      Value tmpPredicate = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, id, idx);
      LDBG("----predicate: " << tmpPredicate);
      predicate = predicate ? rewriter.create<arith::AndIOp>(loc, predicate,
                                                             tmpPredicate)
                            : tmpPredicate;
    }
  }

  // Step 6. Move the body of forallOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 6.a. If predicated, move at the beginning.
    auto ifOp = rewriter.create<scf::IfOp>(loc, predicate,
                                           /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 6.b. Otherwise, move inline just at the rewriter insertion
    // point.
    targetBlock = forallOp->getBlock();
    insertionPoint = rewriter.getInsertionPoint();
  }
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 7. RAUW indices.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  // Step 8. Erase old op.
  rewriter.eraseOp(forallOp);

  result = ForallRewriteResult{forallMappingSizes, mappingIdOps};
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MapForallToBlocks
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapForallToBlocksImpl(
    RewriterBase &rewriter, TransformOpInterface transformOp,
    scf::ForallOp forallOp, SmallVectorImpl<int64_t> &gridDims,
    const GpuIdBuilder &gpuIdBuilder) {
  LDBG("Start mapForallToBlocksImpl");

  Location loc = forallOp.getLoc();
  Block *parentBlock = forallOp->getBlock();
  Value zero;
  {
    // Create an early zero index value for replacements and immediately reset
    // the insertion point.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentBlock);
    zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  }

  SmallVector<int64_t> anyAvailableMappingSizes;
  ForallRewriteResult rewriteResult;
  // Pass an empty anyAvailableMappingSizes.
  DiagnosedSilenceableFailure diag =
      rewriteOneForallCommonImpl(rewriter, transformOp, forallOp, rewriteResult,
                                 anyAvailableMappingSizes, gpuIdBuilder);

  // Return if anything goes wrong, use silenceable failure as a match failure.
  if (!diag.succeeded())
    return diag;

  // Set the gridDims that act as a return.
  gridDims = rewriteResult.mappingSizes;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<BlockDimOp>(rewriter, loc, parentBlock, zero,
                                          gridDims);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::transform::gpu::findTopLevelForallOp(Operation *target,
                                           scf::ForallOp &topLevelForallOp,
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

DiagnosedSilenceableFailure
transform::MapForallToBlocks::applyToOne(Operation *target,
                                         ApplyToEachResultList &results,
                                         transform::TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  IRRewriter rewriter(getContext());
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!getGenerateGpuLaunch() && !gpuLaunch) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "Given target is not gpu.launch, set `generate_gpu_launch` "
           "attribute";
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  scf::ForallOp topLevelForallOp;
  DiagnosedSilenceableFailure diag = mlir::transform::gpu::findTopLevelForallOp(
      target, topLevelForallOp, transformOp);
  if (!diag.succeeded()) {
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  SmallVector<int64_t> gridDims{getGridDims()};
  if (!getGenerateGpuLaunch() && gridDims.size() != 3)
    return transformOp.emitDefiniteFailure("transform require size-3 mapping");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(topLevelForallOp);

  // Generate gpu launch here and move the forall inside
  if (getGenerateGpuLaunch()) {
    DiagnosedSilenceableFailure diag =
        createGpuLaunch(rewriter, target->getLoc(), transformOp, gpuLaunch);
    if (!diag.succeeded()) {
      return diag;
    }
    rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
    Operation *newForallOp = rewriter.clone(*topLevelForallOp);
    rewriter.eraseOp(topLevelForallOp);
    topLevelForallOp = cast<scf::ForallOp>(newForallOp);
  }

  GpuBlockIdBuilder gpuBlockIdBuilder(getContext(), {}, {});
  diag = mlir::transform::gpu::mapForallToBlocksImpl(
      rewriter, transformOp, topLevelForallOp, gridDims, gpuBlockIdBuilder);
  if (!diag.succeeded())
    return diag;

  // Set the GPU launch configuration for the grid dims late, this is subject to
  // IR inspection.
  diag = alterGpuLaunch(rewriter, gpuLaunch,
                        cast<TransformOpInterface>(getOperation()), gridDims[0],
                        gridDims[1], gridDims[2]);

  results.push_back(gpuLaunch);
  return diag;
}

//===----------------------------------------------------------------------===//
// MapNestedForallToThreads
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapOneForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> availableMappingSizes,
    bool syncAfterDistribute, const GpuIdBuilder &gpuIdBuilder) {
  // Ignore cases with different attributes than this builder supports.
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (!llvm::is_contained(gpuIdBuilder.mappingAttributes, map)) {
      LDBG("--skip " << map);
      LLVM_DEBUG(llvm::interleaveComma(gpuIdBuilder.mappingAttributes,
                                       DBGS() << "----not in: ");
                 llvm::dbgs() << "\n";);
      return emitSilenceableFailure(forallOp);
    }
  }

  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  // Insert after to allow for syncthreads after `forall` is erased.
  rewriter.setInsertionPointAfter(forallOp);
  ForallRewriteResult rewriteResult;
  DiagnosedSilenceableFailure diag =
      rewriteOneForallCommonImpl(rewriter, transformOp, forallOp, rewriteResult,
                                 availableMappingSizes, gpuIdBuilder);

  // Return if anything goes wrong, use silenceable failure as a match failure.
  if (!diag.succeeded())
    return diag;

  // Add a syncthreads if needed. TODO: warpsync
  if (syncAfterDistribute)
    rewriter.create<BarrierOp>(loc);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    Operation *target, ArrayRef<int64_t> blockDims, ArrayRef<int64_t> warpDims,
    bool syncAfterDistribute) {
  LDBG("Start mapNestedForallToThreadsImpl");
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<OpFoldResult> blockDimsOfr =
      getAsIndexOpFoldResult(ctx, blockDims);

  if (blockDims.size() != 3)
    return definiteFailureHelper(transformOp, target,
                                 "requires size-3 thread mapping");
  if (!warpDims.empty()) {
    if (warpDims.size() != 3)
      return definiteFailureHelper(transformOp, target,
                                   "requires empty or size-3 warp mapping");
  }

  // Create an early zero index value for replacements.
  Location loc = target->getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  WalkResult walkResult = target->walk([&](scf::ForallOp forallOp) {
    //===--------------------------------------------------------------------===//
    // Mapping to warp ids.
    //===--------------------------------------------------------------------===//
    if (!warpDims.empty()) {
      LLVM_DEBUG(
          llvm::interleaveComma(
              warpDims, DBGS() << "+mapNestedForallToThreadsImpl warpDims: ");
          llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::interleaveComma(
                     blockDimsOfr, DBGS() << "--warpDims with blockDimsOfr:  ");
                 llvm::dbgs() << "\n");
      GpuWarpIdBuilder gpuWarpIdBuilder(ctx, blockDimsOfr, warpDims);
      diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
          rewriter, transformOp, forallOp, warpDims, syncAfterDistribute,
          gpuWarpIdBuilder);
      // Use silenceable failure to encode "failure to match" and pass
      // through.
      if (diag.isDefiniteFailure())
        return WalkResult::interrupt();
      if (diag.succeeded())
        return WalkResult::skip();
    }

    //===--------------------------------------------------------------------===//
    // Mapping to linear ids.
    //===--------------------------------------------------------------------===//
    LDBG("+mapNestedForallToThreadsImpl linearDims");
    LLVM_DEBUG(llvm::interleaveComma(
                   blockDimsOfr, DBGS() << "--linearDims with blockDimsOfr:  ");
               llvm::dbgs() << "\n");
    int64_t numThreads = 1;
    for (int64_t b : blockDims)
      numThreads *= b;
    GpuLinearIdBuilder gpuLinearIdBuilder(ctx, blockDimsOfr, numThreads);
    diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
        rewriter, transformOp, forallOp, numThreads, syncAfterDistribute,
        gpuLinearIdBuilder);
    // Use silenceable failure to encode "failure to match" and pass through.
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();
    if (diag.succeeded())
      return WalkResult::skip();

    //===--------------------------------------------------------------------===//
    // Mapping to block ids (happens last so we can replay ThreadIdOp).
    //===--------------------------------------------------------------------===//
    LLVM_DEBUG(
        llvm::interleaveComma(
            blockDimsOfr, DBGS() << "mapNestedForallToThreadsImpl blockDims: ");
        llvm::dbgs() << "\n");
    GpuThreadIdBuilder gpuThreadIdBuilder(ctx, blockDimsOfr, blockDims);
    diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
        rewriter, transformOp, forallOp, blockDims, syncAfterDistribute,
        gpuThreadIdBuilder);
    // Use silenceable failure to encode "failure to match" and pass through.
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return diag;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<ThreadIdOp>(rewriter, loc, target, zero,
                                          blockDims);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::MapNestedForallToThreads::applyToOne(
    Operation *target, ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Basic high-level verifications.
  if (!gpuLaunch)
    return emitSilenceableError() << "Given target is not a gpu.launch";

  // Mapping to block ids.
  SmallVector<int64_t> blockDims{getBlockDims()};

  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, std::nullopt, std::nullopt, std::nullopt,
                     blockDims[0], blockDims[1], blockDims[2]);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(getLoc()) << getBlockDimsAttrName() << " is too large";
    return diag;
  }

  // Set the GPU launch configuration for the block dims early, this is not
  // subject to IR inspection.
  IRRewriter rewriter(getContext());
  diag = alterGpuLaunch(rewriter, gpuLaunch, transformOp, std::nullopt,
                        std::nullopt, std::nullopt, blockDims[0], blockDims[1],
                        blockDims[2]);

  rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
  diag =
      mapNestedForallToThreadsImpl(rewriter, transformOp, gpuLaunch, blockDims,
                                   getWarpDims(), getSyncAfterDistribute());

  results.push_back(gpuLaunch.getOperation());
  return diag;
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class GPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          GPUTransformDialectExtension> {
public:
  GPUTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<GPUDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"

void mlir::gpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<GPUTransformDialectExtension>();
}
