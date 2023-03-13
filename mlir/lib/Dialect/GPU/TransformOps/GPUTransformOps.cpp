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
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;

/// Check if given mapping attributes are one of the desired attributes
static DiagnosedSilenceableFailure
checkAttributeType(ArrayRef<DeviceMappingAttrInterface> threadMappingAttributes,
                   const std::optional<ArrayAttr> &foreachMapping,
                   std::optional<TransformOpInterface> transformOp) {
  if (!foreachMapping.has_value())
    return transformOp->emitSilenceableError() << "mapping must be present";

  DenseSet<Attribute> seen;
  for (Attribute map : foreachMapping->getValue()) {
    if (!llvm::is_contained(threadMappingAttributes, map)) {
      return transformOp->emitDefiniteFailure()
             << "mapping must be one of " << threadMappingAttributes;
    }
    if (llvm::is_contained(seen, map)) {
      return transformOp->emitDefiniteFailure()
             << map
             << " is duplicated, cannot map different "
                "loops to the same processor";
    }
    seen.insert(map);
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
alterGpuLaunch(TrivialPatternRewriter &rewriter, LaunchOp gpuLaunch,
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
// MapForeachToBlocks
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapForeachToBlocksImpl(
    RewriterBase &rewriter, scf::ForallOp forallOp,
    function_ref<void(RewriterBase &, scf::ForallOp, SmallVectorImpl<Value> &)>
        blockIdGenerator,
    SmallVectorImpl<int64_t> &gridDims, TransformOpInterface transformOp,
    const ArrayRef<DeviceMappingAttrInterface> &mappingAttributes) {
  // Step 0. Target-specific verifications. There is no good place to anchor
  // those right now: the ForallOp is target-independent and the
  // transform op does not apply to individual ForallOp.
  Location loc = forallOp->getLoc();

  if (!forallOp.isNormalized())
    return transformOp.emitSilenceableError()
           << "unsupported non-normalized loops";
  if (forallOp.getNumResults() > 0)
    return transformOp.emitSilenceableError()
           << "only bufferized scf.forall lowers to "
              "gpu.block_id";
  if (forallOp.getRank() > 3)
    return transformOp.emitSilenceableError()
           << "scf.forall with rank > 3 does not lower to "
              "gpu.block_id";
  if (llvm::any_of(forallOp.getMixedUpperBound(), [](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return transformOp.emitSilenceableError()
           << "unsupported dynamic griddim size";
  }
  SmallVector<Attribute> blockMapping =
      llvm::to_vector(forallOp.getMapping()->getValue());

  // Step 1. Complete the blockMapping to a full mapping (with 1s) if necessary.
  SmallVector<Value> numBlocks = forallOp.getUpperBound(rewriter);
  // Ensure we have 3 block sizes, one for each id.
  Value one;
  for (auto attr : mappingAttributes) {
    if (!llvm::is_contained(blockMapping, attr)) {
      blockMapping.push_back(attr);
      one = one ? one : rewriter.create<arith::ConstantIndexOp>(loc, 1);
      numBlocks.push_back(one);
    }
  }

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  auto comparator = [&](DeviceMappingAttrInterface a,
                        DeviceMappingAttrInterface b) -> bool {
    return a.getMappingId() < b.getMappingId();
  };
  SmallVector<Value> gridDimValues =
      scf::ForallOp::getValuesSortedByKey(blockMapping, numBlocks, comparator);
  for (Value v : gridDimValues)
    gridDims.push_back(v.getDefiningOp<arith::ConstantIndexOp>().value());

  // Step 3. Generate the blockIds using the provided generator and map the
  // induction variables to the newly created ops.
  SmallVector<Value> blockOps;
  blockIdGenerator(rewriter, forallOp, blockOps);
  IRMapping bvm;
  for (auto [blockIdx, blockDim] :
       llvm::zip(forallOp.getInductionVars(), blockMapping)) {
    bvm.map(blockIdx,
            blockOps[static_cast<int64_t>(
                blockDim.cast<DeviceMappingAttrInterface>().getMappingId())]);
  }

  // Step 4. Move the body of forallOp.
  // Erase the terminator first, it will not be used since we are on buffers.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock = forallOp->getBlock();
  Block::iterator insertionPoint = Block::iterator(forallOp);
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 5. RAUW thread indices to thread ops.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value blockIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, blockIdx);
  }

  // Step 6. Erase old op.
  rewriter.eraseOp(forallOp);

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
      // TODO: Handle multiple foreach if there is no dependences between them
      return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return transformOp.emitSilenceableError()
           << "could not find a unique topLevel scf.forall";
  return DiagnosedSilenceableFailure::success();
}

/// This is a helper that is only used in
/// rewriteTopLevelForallToGpuBlocks. It generates GPU dialects
/// block_id.
static void generateGpuBlockIds(RewriterBase &rewriter, scf::ForallOp foreachOp,
                                SmallVectorImpl<Value> &blockOps) {
  Location loc = foreachOp->getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(foreachOp);
  IndexType indexType = rewriter.getIndexType();
  blockOps = SmallVector<Value>{
      rewriter.create<BlockIdOp>(loc, indexType, Dimension::x),
      rewriter.create<BlockIdOp>(loc, indexType, Dimension::y),
      rewriter.create<BlockIdOp>(loc, indexType, Dimension::z)};
}

DiagnosedSilenceableFailure
transform::MapForeachToBlocks::applyToOne(Operation *target,
                                          ApplyToEachResultList &results,
                                          transform::TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  TrivialPatternRewriter rewriter(getContext());
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

  SmallVector<int64_t> gridDim = extractFromI64ArrayAttr(getGridDim());
  SmallVector<DeviceMappingAttrInterface> blockMappingAttributes = {
      GPUBlockMappingAttr::get(getContext(), Blocks::DimX),
      GPUBlockMappingAttr::get(getContext(), Blocks::DimY),
      GPUBlockMappingAttr::get(getContext(), Blocks::DimZ)};

  diag = checkAttributeType(blockMappingAttributes,
                            topLevelForallOp.getMapping(), transformOp);
  if (diag.succeeded())
    diag = mlir::transform::gpu::mapForeachToBlocksImpl(
        rewriter, topLevelForallOp, generateGpuBlockIds, gridDim, transformOp,
        blockMappingAttributes);
  if (diag.succeeded()) {
    diag = alterGpuLaunch(rewriter, gpuLaunch,
                          cast<TransformOpInterface>(getOperation()),
                          gridDim[0], gridDim[1], gridDim[2]);
  }

  results.push_back(gpuLaunch);
  return diag;
}

//===----------------------------------------------------------------------===//
// MapNestedForeachToThreads
//===----------------------------------------------------------------------===//

/// Searches `scf.forall` ops nested under `target` and maps each such
/// op to GPU threads. Mapping is one-to-one and the induction variables of
/// `scf.forall` are rewritten to gpu.thread_id according to the
/// thread_dim_mapping attribute. Sibling `scf.forall` are supported in
/// which case, the union of the number of threads is computed and may result
/// in predication. Dynamic, `scf.forall` trip counts are currently
/// not supported. Dynamic block dim sizes are currently not supported.
static DiagnosedSilenceableFailure rewriteOneForallToGpuThreads(
    RewriterBase &rewriter, scf::ForallOp forallOp,
    const SmallVectorImpl<int64_t> &globalBlockDims,
    const SmallVectorImpl<Value> &threadOps, bool syncAfterDistribute,
    std::optional<TransformOpInterface> transformOp,
    const ArrayRef<DeviceMappingAttrInterface> &threadMappingAttributes) {
  // Step 0. Target-specific verifications. There is no good place to anchor
  // those right now: the ForallOp is target-independent and the
  // transform op does not apply to individual ForallOp.
  auto failureHelper =
      [&](const Twine &message) -> DiagnosedSilenceableFailure {
    if (transformOp.has_value()) {
      return transformOp->emitSilenceableError() << message;
    }
    return emitDefiniteFailure(forallOp, message);
  };
  Location loc = forallOp->getLoc();
  if (!forallOp.isNormalized())
    return failureHelper("unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return failureHelper("only bufferized scf.forall lowers to gpu.thread_id");
  if (forallOp.getRank() > 3)
    return failureHelper(
        "scf.forall with rank > 3 does not lower to gpu.thread_id");
  if (llvm::any_of(forallOp.getMixedUpperBound(), [](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return failureHelper("unsupported dynamic blockdim size");
  }
  if (!forallOp.getMapping().has_value())
    return failureHelper("mapping must be present");
  SmallVector<Attribute> threadMapping =
      llvm::to_vector(forallOp.getMapping()->getValue());

  // Step 1. Complete the threadMapping to a full mapping (with 1s) if
  // necessary.
  SmallVector<Value> numThreads = forallOp.getUpperBound(rewriter);
  // Ensure we have 3 block sizes, one for each id.
  Value one;
  for (auto attr : threadMappingAttributes) {
    if (!llvm::is_contained(threadMapping, attr)) {
      threadMapping.push_back(attr);
      one = one ? one : rewriter.create<arith::ConstantIndexOp>(loc, 1);
      numThreads.push_back(one);
    }
  }

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  auto comparator = [&](DeviceMappingAttrInterface a,
                        DeviceMappingAttrInterface b) -> bool {
    return a.getMappingId() < b.getMappingId();
  };
  SmallVector<Value> blockDimValues = scf::ForallOp::getValuesSortedByKey(
      threadMapping, numThreads, comparator);
  SmallVector<int64_t> blockDims =
      llvm::to_vector(llvm::map_range(blockDimValues, [](Value v) {
        return v.getDefiningOp<arith::ConstantIndexOp>().value();
      }));

  // Step 3. Create the gpu.thread ops and map the induction variables to the
  // newly created ops.
  // Replace ids of dimension size 1 by zero to simplify the IR.
  SmallVector<Value> threadOpsUpdated(threadOps.begin(), threadOps.end());
  assert(threadOps.size() == globalBlockDims.size());
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (size_t i : llvm::seq(size_t(0), globalBlockDims.size())) {
    if (globalBlockDims[i] == 1)
      threadOpsUpdated[i] = zero;
  }
  IRMapping bvm;
  for (auto [blockIdx, blockDim] :
       llvm::zip(forallOp.getInductionVars(), threadMapping)) {
    bvm.map(blockIdx,
            threadOpsUpdated[blockDim.cast<DeviceMappingAttrInterface>()
                                 .getMappingId()]);
  }

  // Step 4. Maybe create conditionals to predicate the region.
  Value predicate;
  for (auto [threadId, blockDim, globalBlockDim] :
       llvm::zip(threadOpsUpdated, blockDims, globalBlockDims)) {
    if (blockDim > globalBlockDim) {
      return failureHelper(
          "The requested GPU threads are fewer than the number of loop trip "
          "counts. Try to tile scf.forall before mapping or set "
          "small blockDim.");
    }
    if (blockDim == globalBlockDim)
      continue;
    Value blockIdx = rewriter.create<arith::ConstantIndexOp>(loc, blockDim);
    Value tmpPredicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadId, blockIdx);
    predicate =
        predicate ? rewriter.create<arith::AndIOp>(loc, predicate, tmpPredicate)
                  : tmpPredicate;
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
    // Step 5.b. Otherwise, move inline just before forallOp.
    targetBlock = forallOp->getBlock();
    insertionPoint = Block::iterator(forallOp);
  }
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 6. RAUW thread indices to thread ops.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  // Step 7. syncthreads.
  // TODO: Need warpsync
  if (syncAfterDistribute)
    rewriter.create<BarrierOp>(loc);

  // Step 8. Erase old op.
  rewriter.eraseOp(forallOp);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapNestedForeachToThreadsImpl(
    RewriterBase &rewriter, Operation *target,
    const SmallVectorImpl<int64_t> &blockDim,
    function_ref<void(RewriterBase &, scf::ForallOp, SmallVectorImpl<Value> &)>
        threadIdGenerator,
    bool syncAfterDistribute, std::optional<TransformOpInterface> transformOp,
    const ArrayRef<DeviceMappingAttrInterface> &threadMappingAttributes) {
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  target->walk([&](scf::ForallOp forallOp) {
    // Ignore cases with different attributes.
    for (Attribute map : forallOp.getMapping()->getValue()) {
      if (!llvm::is_contained(threadMappingAttributes, map)) {
        return WalkResult::skip();
      }
    }
    diag = checkAttributeType(threadMappingAttributes, forallOp.getMapping(),
                              transformOp);
    if (diag.succeeded()) {
      rewriter.setInsertionPoint(forallOp);
      SmallVector<Value> threadOps;
      threadIdGenerator(rewriter, forallOp, threadOps);
      diag = rewriteOneForallToGpuThreads(rewriter, forallOp, blockDim,
                                          threadOps, syncAfterDistribute,
                                          transformOp, threadMappingAttributes);
    }
    return diag.succeeded() ? WalkResult::advance() : WalkResult::interrupt();
  });
  return diag;
}

DiagnosedSilenceableFailure transform::MapNestedForeachToThreads::applyToOne(
    Operation *target, ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!gpuLaunch) {
    return emitSilenceableError() << "Given target is not gpu.launch";
  }

  SmallVector<int64_t> blockDim = extractFromI64ArrayAttr(getBlockDim());
  blockDim.resize(/*size=*/3, /*value=*/1);

  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, std::nullopt, std::nullopt, std::nullopt,
                     blockDim[0], blockDim[1], blockDim[2]);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(getLoc()) << getBlockDimAttrName() << " is very large";
    return diag;
  }

  MLIRContext *ctx = getContext();
  TrivialPatternRewriter rewriter(ctx);
  rewriter.setInsertionPoint(target);

  SmallVector<DeviceMappingAttrInterface> threadMappingAttributes = {
      GPUThreadMappingAttr::get(ctx, Threads::DimX),
      GPUThreadMappingAttr::get(ctx, Threads::DimY),
      GPUThreadMappingAttr::get(ctx, Threads::DimZ)};
  auto threadIdGenerator = [](RewriterBase &rewriter, scf::ForallOp forallOp,
                              SmallVectorImpl<Value> &threadIds) {
    IndexType indexType = rewriter.getIndexType();
    threadIds.assign({rewriter.create<ThreadIdOp>(forallOp->getLoc(), indexType,
                                                  Dimension::x),
                      rewriter.create<ThreadIdOp>(forallOp->getLoc(), indexType,
                                                  Dimension::y),
                      rewriter.create<ThreadIdOp>(forallOp->getLoc(), indexType,
                                                  Dimension::z)});
  };
  diag = mlir::transform::gpu::mapNestedForeachToThreadsImpl(
      rewriter, target, blockDim, threadIdGenerator, getSyncAfterDistribute(),
      transformOp, threadMappingAttributes);

  if (diag.succeeded()) {
    diag = alterGpuLaunch(rewriter, gpuLaunch, transformOp, std::nullopt,
                          std::nullopt, std::nullopt, blockDim[0], blockDim[1],
                          blockDim[2]);
  }

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
