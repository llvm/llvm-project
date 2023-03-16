//===- GPUTransformOps.cpp - Implementation of GPU transform ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;

#define DEBUG_TYPE "gpu-transforms"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Helper type for functions that generate ids for the mapping of a scf.forall.
using IdGeneratorFnType = llvm::function_ref<void(RewriterBase &, scf::ForallOp,
                                                  SmallVectorImpl<Value> &)>;

struct MappingToGpuHelper {
  MappingToGpuHelper(SmallVector<DeviceMappingAttrInterface> mappingAttributes,
                     IdGeneratorFnType idGenerator)
      : mappingAttributes(mappingAttributes), idGenerator(idGenerator) {}

  SmallVector<DeviceMappingAttrInterface> mappingAttributes;
  IdGeneratorFnType idGenerator;
};

struct MappingToGpuBlocksHelper : public MappingToGpuHelper {

  MappingToGpuBlocksHelper(MLIRContext *ctx)
      : MappingToGpuHelper(
            SmallVector<DeviceMappingAttrInterface>{
                GPUBlockMappingAttr::get(ctx, Blocks::DimX),
                GPUBlockMappingAttr::get(ctx, Blocks::DimY),
                GPUBlockMappingAttr::get(ctx, Blocks::DimZ)},
            IdGeneratorFnType{[](RewriterBase &rewriter, scf::ForallOp forallOp,
                                 SmallVectorImpl<Value> &ids) {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(forallOp);
              IndexType indexType = rewriter.getIndexType();
              auto loc = forallOp->getLoc();
              ids.assign(
                  {rewriter.create<BlockIdOp>(loc, indexType, Dimension::x),
                   rewriter.create<BlockIdOp>(loc, indexType, Dimension::y),
                   rewriter.create<BlockIdOp>(loc, indexType, Dimension::z)});
            }}) {}
};

struct MappingToGpuThreadsHelper : public MappingToGpuHelper {
  MappingToGpuThreadsHelper(MLIRContext *ctx)
      : MappingToGpuHelper(
            SmallVector<DeviceMappingAttrInterface>{
                GPUThreadMappingAttr::get(ctx, Threads::DimX),
                GPUThreadMappingAttr::get(ctx, Threads::DimY),
                GPUThreadMappingAttr::get(ctx, Threads::DimZ)},
            IdGeneratorFnType{[](RewriterBase &rewriter, scf::ForallOp forallOp,
                                 SmallVectorImpl<Value> &ids) {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(forallOp);
              IndexType indexType = rewriter.getIndexType();
              auto loc = forallOp->getLoc();
              ids.assign(
                  {rewriter.create<ThreadIdOp>(loc, indexType, Dimension::x),
                   rewriter.create<ThreadIdOp>(loc, indexType, Dimension::y),
                   rewriter.create<ThreadIdOp>(loc, indexType, Dimension::z)});
            }}) {}
};

} // namespace

static DiagnosedSilenceableFailure
failureHelper(std::optional<TransformOpInterface> transformOp,
              scf::ForallOp forallOp, const Twine &message) {
  if (transformOp.has_value())
    return emitDefiniteFailure(*transformOp, message);
  return emitDefiniteFailure(forallOp, message);
}

/// Check if given mapping attributes are one of the desired attributes
static DiagnosedSilenceableFailure
checkMappingAttributeTypes(std::optional<TransformOpInterface> transformOp,
                           scf::ForallOp forallOp) {
  if (!forallOp.getMapping().has_value())
    return failureHelper(transformOp, forallOp, "mapping must be present");

  bool hasBlockMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return attr.isa<GPUBlockMappingAttr>();
      });
  bool hasThreadMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return attr.isa<GPUThreadMappingAttr>();
      });
  int64_t countMappingTypes = 0;
  countMappingTypes += hasBlockMapping ? 1 : 0;
  countMappingTypes += hasThreadMapping ? 1 : 0;
  if (countMappingTypes > 1) {
    return failureHelper(transformOp, forallOp,
                         "cannot mix different mapping types, use nesting");
  }

  DenseSet<Attribute> seen;
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (seen.contains(map)) {
      return failureHelper(transformOp, forallOp,
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
    return failureHelper(transformOp, forallOp,
                         "unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return failureHelper(transformOp, forallOp,
                         "only bufferized scf.forall can be mapped");
  if (forallOp.getRank() > 3)
    return failureHelper(transformOp, forallOp,
                         "scf.forall with rank > 3 does not lower");
  if (llvm::any_of(forallOp.getMixedUpperBound(), [&](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return failureHelper(transformOp, forallOp,
                         "unsupported dynamic sizes in forall op");
  }
  return DiagnosedSilenceableFailure::success();
}

/// Determines if the size of the kernel configuration is supported by the GPU
/// architecture being used. It presently makes use of CUDA limitations, however
/// that aspect may be enhanced for other GPUs.
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
           << "Trying to launch a GPU kernel with gridDim = ("
           << gridDimX.value_or(1) << ", " << gridDimY.value_or(1) << ", "
           << gridDimZ.value_or(1) << ") blockDim = (" << blockDimX.value_or(1)
           << ", " << blockDimY.value_or(1) << ", " << blockDimZ.value_or(1)
           << "). It is larger than the limits.";
  }
  return DiagnosedSilenceableFailure::success();
}

/// Creates an empty-body gpu::LaunchOp using the provided kernel settings and
/// put a terminator within.
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

//===----------------------------------------------------------------------===//
// MapForallToBlocks
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<int64_t>> rewriteOneForallCommonImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp,
    const SmallVectorImpl<int64_t> &availableMappingSizes,
    const ArrayRef<DeviceMappingAttrInterface> &allMappingAttributes,
    IdGeneratorFnType idGenerator) {
  LDBG("Start rewriteOneForallCommonImpl");

  // Step 0. GPU-specific verifications. There is no better place to anchor
  // those right now: the ForallOp is target-independent and the transform op
  // does not apply to individual ForallOp.
  DiagnosedSilenceableFailure diag = verifyGpuMapping(transformOp, forallOp);
  if (!diag.succeeded())
    return failure();

  // Step 1. Complete the mapping to a full mapping (with 1s) if necessary.
  SmallVector<int64_t> tmpMappingSizes = llvm::to_vector(
      llvm::map_range(forallOp.getMixedUpperBound(), [](OpFoldResult ofr) {
        auto maybeStaticValue = getConstantIntValue(ofr);
        assert(maybeStaticValue && "expected static value");
        return maybeStaticValue.value();
      }));
  SmallVector<Attribute> forallMappings =
      llvm::to_vector(forallOp.getMapping()->getValue());
  for (auto attr : allMappingAttributes) {
    if (llvm::is_contained(forallMappings, attr))
      continue;
    forallMappings.push_back(attr);
    tmpMappingSizes.push_back(1);
  }

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  auto comparator = [&](DeviceMappingAttrInterface a,
                        DeviceMappingAttrInterface b) -> bool {
    return a.getMappingId() < b.getMappingId();
  };
  SmallVector<int64_t> mappingSizes =
      getValuesSortedByKey(forallMappings, tmpMappingSizes, comparator);
  LLVM_DEBUG(llvm::interleaveComma(mappingSizes, DBGS() << "mappingSizes: ");
             llvm::dbgs() << "\n";
             llvm::interleaveComma(forallMappings, DBGS() << "mappingAttrs: ");
             llvm::dbgs() << "\n");

  // Step 3. Generate the mappingIdOps using the provided generator and map the
  // induction variables to the newly created ops. Replace ids of dimension
  // known to be of size 1 by zero to simplify the IR.
  SmallVector<Value> mappingIdOps;
  Location loc = forallOp.getLoc();
  idGenerator(rewriter, forallOp, mappingIdOps);
  LLVM_DEBUG(llvm::interleaveComma(mappingIdOps, DBGS() << "mappingIdOps: ");
             llvm::dbgs() << "\n");
  assert(mappingIdOps.size() == mappingSizes.size() && "expect equal sizes");
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  if (!availableMappingSizes.empty()) {
    for (size_t i : llvm::seq(size_t(0), availableMappingSizes.size())) {
      if (availableMappingSizes[i] == 1)
        mappingIdOps[i] = zero;
    }
  }

  IRMapping bvm;
  for (auto [iv, dim] :
       llvm::zip_equal(forallOp.getInductionVars(),
                       ArrayRef<Attribute>{forallMappings}.take_front(
                           forallOp.getInductionVars().size()))) {
    Value peIdOp = mappingIdOps[static_cast<int64_t>(
        dim.cast<DeviceMappingAttrInterface>().getMappingId())];
    bvm.map(iv, peIdOp);
  }

  // Step 4. Maybe create conditionals to predicate the region.
  // Skip this step when availableMappingSizes is empty.
  Value predicate;
  if (!availableMappingSizes.empty()) {
    LLVM_DEBUG(llvm::interleaveComma(availableMappingSizes,
                                     DBGS() << "availableMappingSizes: ");
               llvm::dbgs() << "\n");
    for (auto [id, mappingSize, availableMappingSize] :
         llvm::zip_equal(mappingIdOps, mappingSizes, availableMappingSizes)) {
      if (mappingSize > availableMappingSize) {
        (void)failureHelper(
            transformOp, forallOp,
            "Trying to map to fewer GPU threads than loop iterations but "
            "overprovisioning is not yet supported. "
            "Try additional tiling of the before mapping or map to more "
            "threads.");
        return failure();
      }
      if (mappingSize == availableMappingSize)
        continue;
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, mappingSize);
      Value tmpPredicate = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, id, idx);
      LDBG("predicate: " << tmpPredicate);
      predicate = predicate ? rewriter.create<arith::AndIOp>(loc, predicate,
                                                             tmpPredicate)
                            : tmpPredicate;
    }
  }

  // Step 5. Move the body of forallOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 5.a. If predicated, move at the beginning.
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 5.b. Otherwise, move inline just at the rewriter insertion point.
    targetBlock = forallOp->getBlock();
    insertionPoint = rewriter.getInsertionPoint();
  }
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 6. RAUW thread indices to thread ops.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  // Step 7. Erase old op.
  rewriter.eraseOp(forallOp);

  return mappingSizes;
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapForallToBlocksImpl(
    RewriterBase &rewriter, TransformOpInterface transformOp,
    scf::ForallOp forallOp, SmallVectorImpl<int64_t> &gridDims,
    const ArrayRef<DeviceMappingAttrInterface> &allMappingAttributes,
    IdGeneratorFnType idGenerator) {
  // Pass an empty anyAvailableMappingSizes.
  SmallVector<int64_t> anyAvailableMappingSizes;
  FailureOr<SmallVector<int64_t>> maybeMappingSizes =
      rewriteOneForallCommonImpl(rewriter, transformOp, forallOp,
                                 anyAvailableMappingSizes, allMappingAttributes,
                                 idGenerator);
  if (failed(maybeMappingSizes))
    return DiagnosedSilenceableFailure::definiteFailure();
  gridDims = *maybeMappingSizes;
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

  SmallVector<int64_t> gridDims = extractFromI64ArrayAttr(getGridDim());
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

  diag = verifyGpuMapping(transformOp, topLevelForallOp);
  if (!diag.succeeded())
    return diag;

  MappingToGpuBlocksHelper helper(getContext());
  diag = mlir::transform::gpu::mapForallToBlocksImpl(
      rewriter, transformOp, topLevelForallOp, gridDims,
      helper.mappingAttributes, helper.idGenerator);
  if (!diag.succeeded())
    return diag;

  diag = alterGpuLaunch(rewriter, gpuLaunch,
                        cast<TransformOpInterface>(getOperation()), gridDims[0],
                        gridDims[1], gridDims[2]);

  results.push_back(gpuLaunch);
  return diag;
}

//===----------------------------------------------------------------------===//
// MapNestedForallToThreads
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    Operation *target, const SmallVectorImpl<int64_t> &kernelBlockDims,
    bool syncAfterDistribute,
    const ArrayRef<DeviceMappingAttrInterface> &allMappingAttributes,
    IdGeneratorFnType idGenerator) {
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  target->walk([&](scf::ForallOp forallOp) {
    // Ignore cases with different attributes.
    for (Attribute map : forallOp.getMapping()->getValue()) {
      if (!llvm::is_contained(allMappingAttributes, map)) {
        return WalkResult::skip();
      }
    }
    diag = verifyGpuMapping(transformOp, forallOp);
    if (diag.succeeded()) {
      // Take the loc ahead of time
      Location loc = forallOp.getLoc();
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(forallOp);
      if (failed(rewriteOneForallCommonImpl(rewriter, transformOp, forallOp,
                                            kernelBlockDims,
                                            allMappingAttributes, idGenerator)))
        diag = DiagnosedSilenceableFailure::definiteFailure();
      // Add a syncthreads if needed. TODO: warpsync
      if (syncAfterDistribute)
        rewriter.create<BarrierOp>(loc);
    }
    return diag.succeeded() ? WalkResult::advance() : WalkResult::interrupt();
  });
  return diag;
}

DiagnosedSilenceableFailure transform::MapNestedForallToThreads::applyToOne(
    Operation *target, ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Basic high-level verifications.
  if (!gpuLaunch)
    return emitSilenceableError() << "Given target is not a gpu.launch";

  SmallVector<int64_t> blockDims = extractFromI64ArrayAttr(getBlockDim());
  if (blockDims.size() != 3)
    return transformOp.emitDefiniteFailure("transform require size-3 mapping");

  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, std::nullopt, std::nullopt, std::nullopt,
                     blockDims[0], blockDims[1], blockDims[2]);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(getLoc()) << getBlockDimAttrName() << " is too large";
    return diag;
  }

  MLIRContext *ctx = getContext();
  IRRewriter rewriter(ctx);
  MappingToGpuThreadsHelper helper(ctx);
  diag = mlir::transform::gpu::mapNestedForallToThreadsImpl(
      rewriter, transformOp, target, blockDims, getSyncAfterDistribute(),
      helper.mappingAttributes, helper.idGenerator);

  if (!diag.succeeded())
    return diag;

  diag = alterGpuLaunch(rewriter, gpuLaunch, transformOp, std::nullopt,
                        std::nullopt, std::nullopt, blockDims[0], blockDims[1],
                        blockDims[2]);

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
