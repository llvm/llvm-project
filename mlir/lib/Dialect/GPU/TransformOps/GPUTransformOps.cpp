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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

/// Determines if the size of the kernel configuration is supported by the GPU
/// architecture being used. It presently makes use of CUDA limitations, however
/// that aspect may be enhanced for other GPUs.
static DiagnosedSilenceableFailure
checkGpuLimits(TransformOpInterface transformOp, Optional<int64_t> gridDimX,
               Optional<int64_t> gridDimY, Optional<int64_t> gridDimZ,
               Optional<int64_t> blockDimX, Optional<int64_t> blockDimY,
               Optional<int64_t> blockDimZ) {

  static constexpr int max_total_blockdim = 1024;
  static constexpr int max_blockdimx = 1024;
  static constexpr int max_blockdimy = 1024;
  static constexpr int max_blockdimz = 64;
  static constexpr int max_total_griddim = 2147483647;
  static constexpr int max_griddimx = 2147483647;
  static constexpr int max_griddimy = 65535;
  static constexpr int max_griddimz = 65535;

  if ((blockDimX.value_or(1) * blockDimY.value_or(1) * blockDimZ.value_or(1)) >
          max_total_blockdim ||
      (gridDimX.value_or(1) * gridDimY.value_or(1) * gridDimZ.value_or(1)) >
          max_total_griddim ||
      blockDimX.value_or(1) > max_blockdimx ||
      blockDimY.value_or(1) > max_blockdimy ||
      blockDimZ.value_or(1) > max_blockdimz ||
      gridDimY.value_or(1) > max_griddimy ||
      gridDimZ.value_or(1) > max_griddimz ||
      gridDimX.value_or(1) > max_griddimx) {
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
                Optional<int64_t> gridDimX = llvm::None,
                Optional<int64_t> gridDimY = llvm::None,
                Optional<int64_t> gridDimZ = llvm::None,
                Optional<int64_t> blockDimX = llvm::None,
                Optional<int64_t> blockDimY = llvm::None,
                Optional<int64_t> blockDimZ = llvm::None) {
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
  return DiagnosedSilenceableFailure(success());
}

/// Alter kernel configuration of the given kernel.
static DiagnosedSilenceableFailure
alterGpuLaunch(SimpleRewriter &rewriter, LaunchOp gpuLaunch,
               TransformOpInterface transformOp,
               Optional<int64_t> gridDimX = llvm::None,
               Optional<int64_t> gridDimY = llvm::None,
               Optional<int64_t> gridDimZ = llvm::None,
               Optional<int64_t> blockDimX = llvm::None,
               Optional<int64_t> blockDimY = llvm::None,
               Optional<int64_t> blockDimZ = llvm::None) {
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
    RewriterBase &rewriter, scf::ForeachThreadOp foreachThreadOp,
    function_ref<void(RewriterBase &, scf::ForeachThreadOp,
                      SmallVectorImpl<Value> &)>
        blockIdGenerator,
    SmallVectorImpl<int64_t> &gridDims, TransformOpInterface transformOp) {
  if (foreachThreadOp.getNumResults() > 0)
    return transformOp.emitSilenceableError()
           << "only bufferized scf.foreach_thread lowers to gpu.block_id";
  if (foreachThreadOp.getNumThreads().size() > 3)
    return transformOp.emitSilenceableError()
           << "scf.foreach_thread with rank > 3 does not lower to gpu.block_id";

  // Step 0. Outline the compute workload region and set up the workload
  // operands.
  FailureOr<SmallVector<OpFoldResult>> potentialGridDim =
      foreachThreadOp.getPermutedNumThreads(rewriter);

  if (failed(potentialGridDim) ||
      llvm::any_of(*potentialGridDim, [](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return transformOp.emitSilenceableError() << "unsupported dynamic gridDim";
  }

  for (OpFoldResult ofr : *potentialGridDim)
    gridDims.push_back(getConstantIntValue(ofr).value());

  SmallVector<Value> blockOps;
  blockIdGenerator(rewriter, foreachThreadOp, blockOps);

  // Step 1. Move the body of foreachThreadOp.
  // Erase the terminator first, it will not be used since we are on buffers.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *targetBlock = foreachThreadOp->getBlock();
  Block::iterator insertionPoint = Block::iterator(foreachThreadOp);
  Block &sourceBlock = foreachThreadOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 2. RAUW thread indices to thread ops.
  SmallVector<Value> threadIndices =
      *foreachThreadOp.getPermutedThreadIndices();
  assert(blockOps.size() == 3 && "3 block id ops are required");
  for (auto [blockIdx, blockOp] : llvm::zip(threadIndices, blockOps)) {
    Value val = blockIdx;
    Value blkOp = blockOp;
    if (!val)
      continue;
    for (Operation *user : llvm::make_early_inc_range(val.getUsers()))
      user->replaceUsesOfWith(val, blkOp);
  }

  // Step 3. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::findTopLevelForeachThreadOp(
    Operation *target, scf::ForeachThreadOp &topLevelForeachThreadOp,
    TransformOpInterface transformOp) {
  auto walkResult = target->walk([&](scf::ForeachThreadOp foreachThreadOp) {
    if (foreachThreadOp->getParentOfType<scf::ForeachThreadOp>())
      return WalkResult::advance();
    if (topLevelForeachThreadOp)
      // TODO: Handle multiple foreach if there is no dependences between them
      return WalkResult::interrupt();
    topLevelForeachThreadOp = foreachThreadOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return transformOp.emitSilenceableError()
           << "could not find a unique topLevel scf.foreach_thread";
  return DiagnosedSilenceableFailure::success();
}

/// This is a helper that is only used in
/// rewriteTopLevelForeachThreadToGpuBlocks. It generates GPU dialects block_id.
static void generateGpuBlockIds(RewriterBase &rewriter,
                                scf::ForeachThreadOp foreachOp,
                                SmallVectorImpl<Value> &blockOps) {
  Location loc = foreachOp->getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(foreachOp);
  IndexType indexType = rewriter.getIndexType();
  SmallVector<Dimension> gpuDims{Dimension::x, Dimension::y, Dimension::z};
  for (int64_t idx : llvm::seq<int64_t>(0, gpuDims.size())) {
    blockOps.push_back(
        rewriter.create<BlockIdOp>(loc, indexType, gpuDims[idx]));
  }
}

DiagnosedSilenceableFailure
transform::MapForeachToBlocks::applyToOne(Operation *target,
                                          SmallVectorImpl<Operation *> &results,
                                          transform::TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  SimpleRewriter rewriter(getContext());
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!getGenerateGpuLaunch() && !gpuLaunch) {
    results.assign({target});
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "Given target is not gpu.launch, set `generate_gpu_launch` "
           "attribute";
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  scf::ForeachThreadOp topLevelForeachThreadOp;
  DiagnosedSilenceableFailure diag =
      mlir::transform::gpu::findTopLevelForeachThreadOp(
          target, topLevelForeachThreadOp, transformOp);
  if (!diag.succeeded()) {
    results.assign({target});
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(topLevelForeachThreadOp);

  // Generate gpu launch here and move the foreach_thread inside
  if (getGenerateGpuLaunch()) {
    DiagnosedSilenceableFailure diag =
        createGpuLaunch(rewriter, target->getLoc(), transformOp, gpuLaunch);
    if (!diag.succeeded()) {
      results.assign({target});
      return diag;
    }
    rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
    Operation *newForeachThreadOp = rewriter.clone(*topLevelForeachThreadOp);
    rewriter.eraseOp(topLevelForeachThreadOp);
    topLevelForeachThreadOp = cast<scf::ForeachThreadOp>(newForeachThreadOp);
  }

  SmallVector<int64_t> gridDim = extractFromI64ArrayAttr(getGridDim());
  diag = mlir::transform::gpu::mapForeachToBlocksImpl(
      rewriter, topLevelForeachThreadOp, generateGpuBlockIds, gridDim,
      transformOp);
  if (diag.succeeded()) {
    diag = alterGpuLaunch(rewriter, gpuLaunch,
                          cast<TransformOpInterface>(getOperation()),
                          gridDim[0], gridDim[1], gridDim[2]);
  }

  results.assign({gpuLaunch});
  return diag;
}

//===----------------------------------------------------------------------===//
// MapNestedForeachToThreads
//===----------------------------------------------------------------------===//

/// Searches `scf.foreach_thread` ops nested under `target` and maps each such
/// op to GPU threads. Mapping is one-to-one and the induction variables of
/// `scf.foreach_thread` are rewritten to gpu.thread_id according to the
/// thread_dim_apping attribute. Sibling `scf.foreach_thread` are supported in
/// which case, the union of the number of threads is computed and may result
/// in predication. Dynamic, `scf.foreach_thread` trip counts are currently
/// not supported. Dynamic block dim sizes are currently not supported.
static DiagnosedSilenceableFailure rewriteOneForeachThreadToGpuThreads(
    RewriterBase &rewriter, scf::ForeachThreadOp foreachThreadOp,
    const SmallVectorImpl<int64_t> &globalBlockDims, bool syncAfterDistribute,
    llvm::Optional<TransformOpInterface> transformOp) {
  auto failureHelper =
      [&](const Twine &message) -> DiagnosedSilenceableFailure {
    if (transformOp.has_value()) {
      return transformOp->emitSilenceableError() << message;
    }
    foreachThreadOp->emitError() << message;
    return DiagnosedSilenceableFailure::definiteFailure();
  };

  if (foreachThreadOp.getNumResults() > 0)
    return failureHelper(
        "only bufferized scf.foreach_thread lowers to gpu.thread_id");

  if (foreachThreadOp.getNumThreads().size() > 3)
    return failureHelper(
        "scf.foreach_thread with rank > 3 does not lower to gpu.thread_id");

  auto potentialBlockDim = foreachThreadOp.getPermutedNumThreads(rewriter);
  if (failed(potentialBlockDim) ||
      llvm::any_of(*potentialBlockDim, [](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return failureHelper("unsupported dynamic blockdim size");
  }

  SmallVector<int64_t> blockDim =
      llvm::to_vector(llvm::map_range(*potentialBlockDim, [](OpFoldResult ofr) {
        return getConstantIntValue(ofr).value();
      }));

  // Step 1. Create the gpu.thread ops
  Location loc = foreachThreadOp.getLoc();
  IndexType indexType = rewriter.getIndexType();

  SmallVector<Dimension> gpuDims{Dimension::x, Dimension::y, Dimension::z};
  SmallVector<Value> threadOps;
  for (int64_t idx : llvm::seq<int64_t>(0, blockDim.size())) {
    threadOps.push_back(
        rewriter.create<ThreadIdOp>(loc, indexType, gpuDims[idx]));
  }
  // Step 2. Maybe create conditionals to predicate the region.
  Value predicate;
  for (auto [threadId, blockDim, globalBlockDim] :
       llvm::zip(threadOps, blockDim, globalBlockDims)) {
    if (blockDim > globalBlockDim) {
      return failureHelper(
          "The requested GPU threads are fewer than the number of loop trip "
          "counts. Try to tile scf.foreach_thread before mapping or set small "
          "blockDim.");
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

  // Step 3. Move the body of foreachThreadOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 3.a. If predicated, move at the beginning.
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 3.a. Otherwise, move inline just before foreachThreadOp.
    targetBlock = foreachThreadOp->getBlock();
    insertionPoint = Block::iterator(foreachThreadOp);
  }
  Block &sourceBlock = foreachThreadOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 4. RAUW thread indices to thread ops.
  SmallVector<Value> threadIndices =
      *foreachThreadOp.getPermutedThreadIndices();
  for (auto [threadIdx, threadOp] : llvm::zip(threadIndices, threadOps)) {
    Value val = threadIdx;
    Value op = threadOp;
    if (!val)
      continue;
    for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
      user->replaceUsesOfWith(val, op);
    }
  }

  // Step 5. syncthreads.
  // TODO: Need warpsync
  if (syncAfterDistribute)
    rewriter.create<BarrierOp>(loc);

  // Step 6. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapNestedForeachToThreadsImpl(
    RewriterBase &rewriter, Operation *target,
    const SmallVectorImpl<int64_t> &blockDim, bool syncAfterDistribute,
    llvm::Optional<TransformOpInterface> transformOp) {
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  target->walk([&](scf::ForeachThreadOp foreachThreadOp) {
    rewriter.setInsertionPoint(foreachThreadOp);
    diag = rewriteOneForeachThreadToGpuThreads(
        rewriter, foreachThreadOp, blockDim, syncAfterDistribute, transformOp);
    return diag.succeeded() ? WalkResult::advance() : WalkResult::interrupt();
  });
  return diag;
}

DiagnosedSilenceableFailure transform::MapNestedForeachToThreads::applyToOne(
    ::mlir::Operation *target,
    ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
    ::mlir::transform::TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!gpuLaunch) {
    results.assign({target});
    return emitSilenceableError() << "Given target is not gpu.launch";
  }

  SmallVector<int64_t> blockDim = extractFromI64ArrayAttr(getBlockDim());
  blockDim.resize(/*size=*/3, /*value=*/1);

  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, llvm::None, llvm::None, llvm::None,
                     blockDim[0], blockDim[1], blockDim[2]);
  if (diag.isSilenceableFailure()) {
    results.assign({target});
    diag.attachNote(getLoc()) << getBlockDimAttrName() << " is very large";
    return diag;
  }

  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);

  diag = mlir::transform::gpu::mapNestedForeachToThreadsImpl(
      rewriter, target, blockDim, getSyncAfterDistribute(), transformOp);
  if (diag.succeeded()) {
    diag =
        alterGpuLaunch(rewriter, gpuLaunch, transformOp, llvm::None, llvm::None,
                       llvm::None, blockDim[0], blockDim[1], blockDim[2]);
  }

  results.assign({gpuLaunch});
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
