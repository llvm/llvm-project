//===- TestLinalgTransforms.cpp - Test Linalg transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/HoistPadding.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgTransforms
    : public PassWrapper<TestLinalgTransforms, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgTransforms)

  TestLinalgTransforms() = default;
  TestLinalgTransforms(const TestLinalgTransforms &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    bufferization::BufferizationDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    linalg::LinalgDialect,
                    vector::VectorDialect,
                    gpu::GPUDialect>();
    // clang-format on
  }
  StringRef getArgument() const final {
    return "test-linalg-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Linalg transformation patterns by applying them greedily.";
  }

  void runOnOperation() override;

  Option<bool> testPatterns{*this, "test-patterns",
                            llvm::cl::desc("Test a mixed set of patterns"),
                            llvm::cl::init(false)};
  Option<bool> testTileAndDistributionOptions{
      *this, "test-tile-and-distribute-options",
      llvm::cl::desc("Test tile and distribute options"),
      llvm::cl::init(false)};
  Option<bool> testTileFuseAndDistributionOptions{
      *this, "test-tile-fuse-and-distribute-options",
      llvm::cl::desc("Test tile, fuse and distribute options"),
      llvm::cl::init(false)};
  Option<bool> testVectorTransferForwardingPatterns{
      *this, "test-vector-transfer-forwarding-patterns",
      llvm::cl::desc(
          "Test a fused pass that forwards memref.copy to vector.transfer"),
      llvm::cl::init(false)};
  Option<bool> testGenericToVectorPattern{
      *this, "test-linalg-to-vector-patterns",
      llvm::cl::desc("Test a set of patterns that rewrite a linalg contraction "
                     "in vector.contract form"),
      llvm::cl::init(false)};
  Option<bool> testTilePattern{*this, "test-tile-pattern",
                               llvm::cl::desc("Test tile pattern"),
                               llvm::cl::init(false)};
  Option<bool> testTileScalarizeDynamicDims{
      *this, "test-tile-scalarize-dynamic-dims",
      llvm::cl::desc("Test tiling of dynamic dims by 1"),
      llvm::cl::init(false)};
  Option<bool> testTransformPadTensor{
      *this, "test-transform-pad-tensor",
      llvm::cl::desc("Test transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> testGeneralizePadTensor{
      *this, "test-generalize-pad-tensor",
      llvm::cl::desc("Test transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> testSwapSubTensorPadTensor{
      *this, "test-swap-subtensor-padtensor",
      llvm::cl::desc("Test rewrite of subtensor(tensor.pad) into "
                     "tensor.pad(subtensor)"),
      llvm::cl::init(false)};
  Option<bool> testSplitReduction{
      *this, "test-split-reduction",
      llvm::cl::desc("Test split reduction transformation"),
      llvm::cl::init(false)};
  ListOption<int64_t> peeledLoops{
      *this, "peeled-loops",
      llvm::cl::desc("Loops to be peeled when test-tile-pattern")};
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Linalg tile sizes for test-tile-pattern")};
  Option<bool> skipPartial{
      *this, "skip-partial",
      llvm::cl::desc("Skip loops inside partial iterations during peeling"),
      llvm::cl::init(false)};
  Option<std::string> loopType{
      *this, "loop-type",
      llvm::cl::desc("Specify the type of loops to generate: for, parallel or "
                     "tiled_loop"),
      llvm::cl::init("for")};
  Option<bool> testBubbleUpExtractSliceOpPattern{
      *this, "test-bubble-up-extract-slice-op-pattern",
      llvm::cl::desc("Test rewrite of linalgOp + extract_slice into "
                     "extract_slice + linalgOp"),
      llvm::cl::init(false)};
};
} // namespace

static void applyPatterns(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

  //===--------------------------------------------------------------------===//
  // Linalg tiling patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({2000, 3000, 4000}),
      LinalgTransformationFilter(StringAttr::get(ctx, "MEM"),
                                 StringAttr::get(ctx, "L3")));
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({200, 300, 400}),
      LinalgTransformationFilter(StringAttr::get(ctx, "L3"),
                                 StringAttr::get(ctx, "L2")));
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgTransformationFilter(StringAttr::get(ctx, "L2"),
                                 StringAttr::get(ctx, "L1")));
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({2, 3, 4}),
      LinalgTransformationFilter(StringAttr::get(ctx, "L1"),
                                 StringAttr::get(ctx, "REG")));

  patterns.add<LinalgTilingPattern>(
      MatvecOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({5, 6}).setLoopType(
          LinalgTilingLoopType::ParallelLoops),
      LinalgTransformationFilter(ArrayRef<StringAttr>{},
                                 StringAttr::get(ctx, "L1")));

  patterns.add<LinalgTilingPattern>(
      DotOp::getOperationName(), ctx, LinalgTilingOptions().setTileSizes(8000),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>{StringAttr::get(ctx, "MEM"),
                               StringAttr::get(ctx, "L3"),
                               StringAttr::get(ctx, "L2")},
          StringAttr::get(ctx, "REG")));

  //===--------------------------------------------------------------------===//
  // Linalg tiling and permutation patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions()
          .setTileSizes({2000, 3000, 4000})
          .setInterchange({1, 2, 0}),
      LinalgTransformationFilter(StringAttr::get(ctx, "__with_perm__"),
                                 StringAttr::get(ctx, "L2__with_perm__")));
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions()
          .setTileSizes({200, 300, 400})
          .setInterchange({1, 0, 2}),
      LinalgTransformationFilter(StringAttr::get(ctx, "L2__with_perm__"),
                                 StringAttr::get(ctx, "L1__with_perm__")));
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgTransformationFilter(StringAttr::get(ctx, "L1__with_perm__"),
                                 StringAttr::get(ctx, "REG__with_perm__")));

  patterns.add<LinalgTilingPattern>(
      MatvecOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({5, 6}).setInterchange({1, 0}),
      LinalgTransformationFilter(StringAttr::get(ctx, "__with_perm__"),
                                 StringAttr::get(ctx, "L1__with_perm__")));

  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions()
          .setTileSizes({16, 8, 4})
          .setInterchange({1, 2, 0})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgTransformationFilter(
          StringAttr::get(ctx, "par__with_perm__"),
          StringAttr::get(ctx, "after_par__with_perm__")));

  //===--------------------------------------------------------------------===//
  // Linalg to loops patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgLoweringPattern<DotOp>>(
      ctx,
      /*loweringType=*/LinalgLoweringType::Loops,
      LinalgTransformationFilter(StringAttr::get(ctx, "REG")));

  //===--------------------------------------------------------------------===//
  // Linalg distribution patterns.
  //===--------------------------------------------------------------------===//
  LinalgLoopDistributionOptions distributionOptions;

  //===--------------------------------------------------------------------===//
  // Linalg to vector contraction patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<CopyVectorizationPattern>(ctx);

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

template <typename IdOp, typename NProcsOp>
static SmallVector<ProcInfo, 2>
getGpuProcIds(OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges,
              ArrayRef<linalg::DistributionMethod> distributionMethod) {
  size_t count = std::min<size_t>(3, parallelLoopRanges.size());
  SmallVector<ProcInfo, 2> procInfo(count);
  Type indexType = b.getIndexType();
  for (unsigned i = 0; i < count; ++i) {
    gpu::Dimension dim = *gpu::symbolizeDimension(i);
    procInfo[count - 1 - i] = {b.create<IdOp>(loc, indexType, dim),
                               b.create<NProcsOp>(loc, indexType, dim),
                               distributionMethod[count - 1 - i]};
  }
  return procInfo;
}

static void fillTileAndDistributePatterns(MLIRContext *context,
                                          RewritePatternSet &patterns) {
  {
    LinalgLoopDistributionOptions cyclicNprocsEqNiters;
    SmallVector<linalg::DistributionMethod> distributionMethod = {
        DistributionMethod::CyclicNumProcsEqNumIters,
        DistributionMethod::CyclicNumProcsEqNumIters};
    cyclicNprocsEqNiters.procInfo =
        [distributionMethod](OpBuilder &b, Location loc,
                             ArrayRef<Range> parallelLoopRanges) {
          return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
              b, loc, parallelLoopRanges, distributionMethod);
        };
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsEqNiters),
        LinalgTransformationFilter(
            StringAttr::get(context, "distribute1"),
            StringAttr::get(context, "after_distribute1")));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsGeNiters;
    SmallVector<linalg::DistributionMethod> distributionMethod = {
        DistributionMethod::CyclicNumProcsGeNumIters,
        DistributionMethod::CyclicNumProcsGeNumIters};
    cyclicNprocsGeNiters.procInfo =
        [distributionMethod](OpBuilder &b, Location loc,
                             ArrayRef<Range> parallelLoopRanges) {
          return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
              b, loc, parallelLoopRanges, distributionMethod);
        };
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsGeNiters),
        LinalgTransformationFilter(
            StringAttr::get(context, "distribute2"),
            StringAttr::get(context, "after_distribute2")));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsDefault;
    SmallVector<linalg::DistributionMethod> distributionMethod = {
        DistributionMethod::Cyclic, DistributionMethod::Cyclic};
    cyclicNprocsDefault.procInfo =
        [distributionMethod](OpBuilder &b, Location loc,
                             ArrayRef<Range> parallelLoopRanges) {
          return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
              b, loc, parallelLoopRanges, distributionMethod);
        };
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsDefault),
        LinalgTransformationFilter(
            StringAttr::get(context, "distribute3"),
            StringAttr::get(context, "after_distribute3")));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsMixed1;
    SmallVector<linalg::DistributionMethod> distributionMethod = {
        DistributionMethod::CyclicNumProcsEqNumIters,
        DistributionMethod::CyclicNumProcsGeNumIters};
    cyclicNprocsMixed1.procInfo =
        [distributionMethod](OpBuilder &b, Location loc,
                             ArrayRef<Range> parallelLoopRanges) {
          return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
              b, loc, parallelLoopRanges, distributionMethod);
        };
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsMixed1),
        LinalgTransformationFilter(
            StringAttr::get(context, "distribute4"),
            StringAttr::get(context, "after_distribute4")));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsMixed2;
    SmallVector<linalg::DistributionMethod> distributionMethod = {
        DistributionMethod::CyclicNumProcsGeNumIters,
        DistributionMethod::Cyclic};
    cyclicNprocsMixed2.procInfo =
        [distributionMethod](OpBuilder &b, Location loc,
                             ArrayRef<Range> parallelLoopRanges) {
          return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
              b, loc, parallelLoopRanges, distributionMethod);
        };
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsMixed2),
        LinalgTransformationFilter(
            StringAttr::get(context, "distribute5"),
            StringAttr::get(context, "after_distribute5")));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsMixed3;
    SmallVector<linalg::DistributionMethod> distributionMethod = {
        DistributionMethod::Cyclic,
        DistributionMethod::CyclicNumProcsEqNumIters};
    cyclicNprocsMixed3.procInfo =
        [distributionMethod](OpBuilder &b, Location loc,
                             ArrayRef<Range> parallelLoopRanges) {
          return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
              b, loc, parallelLoopRanges, distributionMethod);
        };

    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::ParallelLoops)
            .setDistributionOptions(cyclicNprocsMixed3),
        LinalgTransformationFilter(
            StringAttr::get(context, "distribute6"),
            StringAttr::get(context, "after_distribute6")));
  }

  {
    LinalgLoopDistributionOptions cyclicNprocsEqNiters;
    SmallVector<linalg::DistributionMethod> distributionMethod = {
        DistributionMethod::Cyclic, DistributionMethod::Cyclic};
    cyclicNprocsEqNiters.procInfo =
        [distributionMethod](OpBuilder &b, Location loc,
                             ArrayRef<Range> parallelLoopRanges) {
          return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
              b, loc, parallelLoopRanges, distributionMethod);
        };
    patterns.add<LinalgTilingPattern>(
        MatmulOp::getOperationName(), context,
        LinalgTilingOptions()
            .setTileSizes({8, 8, 4})
            .setLoopType(LinalgTilingLoopType::Loops)
            .setDistributionOptions(cyclicNprocsEqNiters),
        LinalgTransformationFilter(
            StringAttr::get(context, "tensors_distribute1"),
            StringAttr::get(context, "tensors_after_distribute1")));
  }
}

static void fillTileFuseAndDistributePatterns(MLIRContext *context,
                                              RewritePatternSet &patterns) {
  LinalgLoopDistributionOptions cyclicNprocsEqNiters;
  SmallVector<linalg::DistributionMethod> distributionMethod = {
      DistributionMethod::Cyclic, DistributionMethod::Cyclic};
  cyclicNprocsEqNiters.procInfo =
      [distributionMethod](OpBuilder &b, Location loc,
                           ArrayRef<Range> parallelLoopRanges) {
        return getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>(
            b, loc, parallelLoopRanges, distributionMethod);
      };
  patterns.add<LinalgTileAndFuseTensorOpsPattern>(
      MatmulOp::getOperationName(), context,
      LinalgTilingAndFusionOptions()
          .setTileSizes({8, 8, 4})
          .setDistributionOptions(cyclicNprocsEqNiters),
      LinalgTransformationFilter(
          StringAttr::get(context, "tensors_fuse_distribute1"),
          StringAttr::get(context, "tensors_after_fuse_distribute1")));
}

static void applyVectorTransferForwardingPatterns(func::FuncOp funcOp) {
  RewritePatternSet forwardPattern(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTRForwardingPattern>(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTWForwardingPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(forwardPattern));
}

static void applyLinalgToVectorPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  auto *ctx = funcOp.getContext();
  patterns.add<CopyVectorizationPattern>(ctx);
  populatePadOpVectorizationPatterns(patterns);
  populateConvolutionVectorizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyPadTensorToGenericPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<PadOpTransformationPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyGeneralizePadTensorPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<GeneralizePadOpPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyExtractSliceOfPadTensorSwapPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<ExtractSliceOfPadTensorSwapPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyTilePattern(func::FuncOp funcOp, const std::string &loopType,
                             ArrayRef<int64_t> tileSizes,
                             ArrayRef<int64_t> peeledLoops,
                             bool scalarizeDynamicDims) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet tilingPattern(context);
  LinalgTilingLoopType type =
      llvm::StringSwitch<LinalgTilingLoopType>(loopType)
          .Case("for", LinalgTilingLoopType::Loops)
          .Case("affine", LinalgTilingLoopType::AffineLoops)
          .Case("parallel", LinalgTilingLoopType::ParallelLoops);
  auto linalgTilingOptions = linalg::LinalgTilingOptions()
                                 .setPeeledLoops(peeledLoops)
                                 .setLoopType(type);
  if (scalarizeDynamicDims) {
    linalgTilingOptions.scalarizeDynamicDims();
    assert(tileSizes.empty() &&
           "tileSizes and scalarizeDynamicDims is mutually exclusive");
  } else {
    linalgTilingOptions.setTileSizes(tileSizes);
  }
  linalg::LinalgTransformationFilter f(StringAttr::get(context, "tile"));
  TilingPatterns<linalg::MatmulOp, linalg::GenericOp>::insert(
      tilingPattern, linalgTilingOptions, f);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(tilingPattern));
}

static void applySplitReduction(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  linalg::populateSplitReductionPattern(
      patterns,
      [](LinalgOp op) {
        unsigned insertDimIndex = op.getNumLoops() - 1;
        return std::make_pair(4, insertDimIndex);
      },
      LinalgTransformationFilter(
          ArrayRef<StringAttr>{},
          StringAttr::get(funcOp.getContext(), "SPLIT")));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyBubbleUpExtractSliceOpPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateBubbleUpExtractSliceOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

/// Apply transformations specified as patterns.
void TestLinalgTransforms::runOnOperation() {
  auto lambda = [&](void *) {
    getOperation().walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  };
  std::unique_ptr<void, decltype(lambda)> cleanupGuard{(void *)1, lambda};

  if (testTileAndDistributionOptions) {
    RewritePatternSet patterns(&getContext());
    fillTileAndDistributePatterns(&getContext(), patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
  if (testTileFuseAndDistributionOptions) {
    RewritePatternSet patterns(&getContext());
    fillTileFuseAndDistributePatterns(&getContext(), patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
  if (testPatterns)
    return applyPatterns(getOperation());
  if (testVectorTransferForwardingPatterns)
    return applyVectorTransferForwardingPatterns(getOperation());
  if (testGenericToVectorPattern)
    return applyLinalgToVectorPatterns(getOperation());
  if (testTransformPadTensor)
    return applyPadTensorToGenericPatterns(getOperation());
  if (testGeneralizePadTensor)
    return applyGeneralizePadTensorPatterns(getOperation());
  if (testSwapSubTensorPadTensor)
    return applyExtractSliceOfPadTensorSwapPattern(getOperation());
  if (testTilePattern)
    return applyTilePattern(getOperation(), loopType, tileSizes, peeledLoops,
                            /*scalarizeDynamicDims=*/false);
  if (testTileScalarizeDynamicDims)
    return applyTilePattern(getOperation(), loopType, tileSizes,
                            /*peeledLoops=*/{}, /*scalarizeDynamicDims=*/true);
  if (testSplitReduction)
    return applySplitReduction(getOperation());
  if (testBubbleUpExtractSliceOpPattern)
    return applyBubbleUpExtractSliceOpPattern(getOperation());
}

namespace mlir {
namespace test {
void registerTestLinalgTransforms() {
  PassRegistration<TestLinalgTransforms>();
}
} // namespace test
} // namespace mlir
