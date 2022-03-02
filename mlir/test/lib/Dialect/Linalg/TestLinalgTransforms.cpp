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
#include "mlir/Dialect/GPU/GPUDialect.h"
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
    : public PassWrapper<TestLinalgTransforms, OperationPass<FuncOp>> {
  TestLinalgTransforms() = default;
  TestLinalgTransforms(const TestLinalgTransforms &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
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
  Option<bool> testMatmulToVectorPatterns1dTiling{
      *this, "test-matmul-to-vector-patterns-tile-1d",
      llvm::cl::desc(
          "Test a fused pass that applies patterns from matmul to vectors via "
          "1-d tiling"),
      llvm::cl::init(false)};
  Option<bool> testMatmulToVectorPatterns2dTiling{
      *this, "test-matmul-to-vector-patterns-tile-2d",
      llvm::cl::desc(
          "Test a fused pass that applies patterns from matmul to vectors via "
          "2-d tiling"),
      llvm::cl::init(false)};
  Option<bool> testPromotionOptions{*this, "test-linalg-promotion-options",
                                    llvm::cl::desc("Test promotion options"),
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
      llvm::cl::desc("Test rewrite of subtensor(pad_tensor) into "
                     "pad_tensor(subtensor)"),
      llvm::cl::init(false)};
  ListOption<int64_t> peeledLoops{
      *this, "peeled-loops",
      llvm::cl::desc("Loops to be peeled when test-tile-pattern"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Linalg tile sizes for test-tile-pattern"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
  Option<bool> skipPartial{
      *this, "skip-partial",
      llvm::cl::desc("Skip loops inside partial iterations during peeling"),
      llvm::cl::init(false)};
  Option<std::string> loopType{
      *this, "loop-type",
      llvm::cl::desc("Specify the type of loops to generate: for, parallel or "
                     "tiled_loop"),
      llvm::cl::init("for")};
};
} // namespace

static void applyPatterns(FuncOp funcOp) {
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
  patterns.add<LinalgVectorizationPattern>(
      ctx, LinalgTransformationFilter(StringAttr::get(ctx, "VECTORIZE"))
               .addOpFilter<MatmulOp, FillOp, GenericOp>());
  patterns.add<CopyVectorizationPattern>(ctx);

  //===--------------------------------------------------------------------===//
  // Linalg generic interchange pattern.
  //===--------------------------------------------------------------------===//
  patterns.add<GenericOpInterchangePattern>(
      ctx,
      /*interchangeVector=*/ArrayRef<unsigned>{1, 2, 0},
      LinalgTransformationFilter(ArrayRef<StringAttr>{},
                                 StringAttr::get(ctx, "PERMUTED")));

  //===--------------------------------------------------------------------===//
  // Linalg subview operands promotion.
  //===--------------------------------------------------------------------===//
  patterns.add<LinalgPromotionPattern<MatmulOp>>(
      ctx, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
      LinalgTransformationFilter(StringAttr::get(ctx, "_promote_views_"),
                                 StringAttr::get(ctx, "_views_promoted_")));
  patterns.add<LinalgPromotionPattern<MatmulOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0})
          .setUseFullTileBuffersByDefault(true),
      LinalgTransformationFilter(
          StringAttr::get(ctx, "_promote_first_view_"),
          StringAttr::get(ctx, "_first_view_promoted_")));
  patterns.add<LinalgPromotionPattern<FillOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({1})
          .setUseFullTileBuffers({false, true})
          .setAlignment(32),
      LinalgTransformationFilter(
          StringAttr::get(ctx, "_promote_views_aligned_"),
          StringAttr::get(ctx, "_views_aligned_promoted_")));

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

static void fillL1TilingAndMatmulToVectorPatterns(
    FuncOp funcOp, StringRef startMarker,
    SmallVectorImpl<RewritePatternSet> &patternsVector) {
  MLIRContext *ctx = funcOp.getContext();
  patternsVector.emplace_back(
      ctx, std::make_unique<LinalgTilingPattern>(
               MatmulOp::getOperationName(), ctx,
               LinalgTilingOptions()
                   .setTileSizes({8, 12, 16})
                   .setInterchange({1, 0, 2}),
               LinalgTransformationFilter(StringAttr::get(ctx, startMarker),
                                          StringAttr::get(ctx, "L1"))));

  patternsVector.emplace_back(
      ctx,
      std::make_unique<LinalgPromotionPattern<MatmulOp>>(
          ctx, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
          LinalgTransformationFilter(StringAttr::get(ctx, "L1"),
                                     StringAttr::get(ctx, "VEC"))));

  patternsVector.emplace_back(
      ctx, std::make_unique<LinalgVectorizationPattern>(
               MatmulOp::getOperationName(), ctx, LinalgVectorizationOptions(),
               LinalgTransformationFilter(StringAttr::get(ctx, "VEC"))));
  patternsVector.back().add<LinalgVectorizationPattern>(
      ctx, LinalgTransformationFilter().addOpFilter<FillOp>());
  patternsVector.back().add<CopyVectorizationPattern>(ctx);
}

//===----------------------------------------------------------------------===//
// Test promotion callbacks
//===----------------------------------------------------------------------===//

// Allocation call back
static Optional<Value> allocCallBackFn(OpBuilder &b, memref::SubViewOp subView,
                                       ArrayRef<Value> boundingSubViewSize,
                                       DataLayout &layout) {
  SmallVector<int64_t, 4> shape(boundingSubViewSize.size(), -1);
  return b
      .create<memref::AllocOp>(
          subView.getLoc(),
          MemRefType::get(shape, subView.getType().getElementType(),
                          /*affineMapComposition =*/{}, 3),
          boundingSubViewSize)
      .getResult();
}

// Deallocation callback
static LogicalResult deallocCallBackFn(OpBuilder &b, Value buffer) {
  b.create<memref::DeallocOp>(buffer.getLoc(), buffer);
  return success();
}

// Copy in call back
static LogicalResult copyCallBackFn(OpBuilder &b, Value src, Value dst,
                                    bool isOutput) {
  auto floatType = src.getType().cast<MemRefType>().getElementType();
  if (!floatType.isa<FloatType>())
    return failure();
  if (!isOutput) {
    Value cst = b.create<arith::ConstantOp>(src.getLoc(),
                                            FloatAttr::get(floatType, 42.0));
    b.create<FillOp>(src.getLoc(), cst, dst);
  }
  b.create<memref::CopyOp>(src.getLoc(), src, dst);
  return success();
}

static void fillPromotionCallBackPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns) {
  patterns.add<LinalgTilingPattern>(
      MatmulOp::getOperationName(), ctx,
      LinalgTilingOptions().setTileSizes({16, 16, 16}),
      LinalgTransformationFilter(StringAttr::get(ctx, "START"),
                                 StringAttr::get(ctx, "PROMOTE")));
  patterns.add<LinalgPromotionPattern<MatmulOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0, 2})
          .setUseFullTileBuffers({false, false})
          .setAllocationDeallocationFns(allocCallBackFn, deallocCallBackFn)
          .setCopyInOutFns(
              [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                return copyCallBackFn(b, src, dst, false);
              },
              [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                return copyCallBackFn(b, src, dst, true);
              }),
      LinalgTransformationFilter(StringAttr::get(ctx, "PROMOTE")));
}

template <typename IdOp, typename NProcsOp>
static SmallVector<ProcInfo, 2>
getGpuProcIds(OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges) {
  size_t count = std::min<size_t>(3, parallelLoopRanges.size());
  SmallVector<ProcInfo, 2> procInfo(count);
  Type indexType = b.getIndexType();
  for (unsigned i = 0; i < count; ++i) {
    gpu::Dimension dim = *gpu::symbolizeDimension(i);
    procInfo[count - 1 - i] = {b.create<IdOp>(loc, indexType, dim),
                               b.create<NProcsOp>(loc, indexType, dim)};
  }
  return procInfo;
}

static void fillTileAndDistributePatterns(MLIRContext *context,
                                          RewritePatternSet &patterns) {
  {
    LinalgLoopDistributionOptions cyclicNprocsEqNiters;
    cyclicNprocsEqNiters.distributionMethod.resize(
        2, DistributionMethod::CyclicNumProcsEqNumIters);
    cyclicNprocsEqNiters.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
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
    cyclicNprocsGeNiters.distributionMethod.resize(
        2, DistributionMethod::CyclicNumProcsGeNumIters);
    cyclicNprocsGeNiters.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
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
    cyclicNprocsDefault.distributionMethod.resize(2,
                                                  DistributionMethod::Cyclic);
    cyclicNprocsDefault.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
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
    cyclicNprocsMixed1.distributionMethod = {
        DistributionMethod::CyclicNumProcsEqNumIters,
        DistributionMethod::CyclicNumProcsGeNumIters};
    cyclicNprocsMixed1.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
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
    cyclicNprocsMixed2.distributionMethod = {
        DistributionMethod::CyclicNumProcsGeNumIters,
        DistributionMethod::Cyclic};
    cyclicNprocsMixed2.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
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
    cyclicNprocsMixed3.distributionMethod = {
        DistributionMethod::Cyclic,
        DistributionMethod::CyclicNumProcsEqNumIters};
    cyclicNprocsMixed3.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;

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
    cyclicNprocsEqNiters.distributionMethod.resize(2,
                                                   DistributionMethod::Cyclic);
    cyclicNprocsEqNiters.procInfo =
        getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
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
  cyclicNprocsEqNiters.distributionMethod.resize(2, DistributionMethod::Cyclic);
  cyclicNprocsEqNiters.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;
  patterns.add<LinalgTileAndFuseTensorOpsPattern>(
      MatmulOp::getOperationName(), context,
      LinalgTilingAndFusionOptions()
          .setTileSizes({8, 8, 4})
          .setDistributionOptions(cyclicNprocsEqNiters),
      LinalgTransformationFilter(
          StringAttr::get(context, "tensors_fuse_distribute1"),
          StringAttr::get(context, "tensors_after_fuse_distribute1")));
}

static void
applyMatmulToVectorPatterns(FuncOp funcOp,
                            bool testMatmulToVectorPatterns1dTiling,
                            bool testMatmulToVectorPatterns2dTiling) {
  MLIRContext *ctx = funcOp.getContext();
  SmallVector<RewritePatternSet, 4> stage1Patterns;
  if (testMatmulToVectorPatterns1dTiling) {
    fillL1TilingAndMatmulToVectorPatterns(funcOp, "START", stage1Patterns);
  } else if (testMatmulToVectorPatterns2dTiling) {
    stage1Patterns.emplace_back(
        ctx, std::make_unique<LinalgTilingPattern>(
                 MatmulOp::getOperationName(), ctx,
                 LinalgTilingOptions()
                     .setTileSizes({768, 264, 768})
                     .setInterchange({1, 2, 0}),
                 LinalgTransformationFilter(StringAttr::get(ctx, "START"),
                                            StringAttr::get(ctx, "L2"))));
    fillL1TilingAndMatmulToVectorPatterns(funcOp, "L2", stage1Patterns);
  }
  {
    // Canonicalization patterns
    RewritePatternSet canonicalizationPatterns(funcOp.getContext());
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        canonicalizationPatterns);
    vector::populateVectorReductionToContractPatterns(canonicalizationPatterns);
    stage1Patterns.push_back(std::move(canonicalizationPatterns));
  }
  SmallVector<FrozenRewritePatternSet, 4> frozenStage1Patterns;
  llvm::move(stage1Patterns, std::back_inserter(frozenStage1Patterns));
  FrozenRewritePatternSet stage2Patterns =
      getLinalgTilingCanonicalizationPatterns(ctx);
  (void)applyStagedPatterns(funcOp, frozenStage1Patterns, stage2Patterns);
}

static void applyVectorTransferForwardingPatterns(FuncOp funcOp) {
  RewritePatternSet forwardPattern(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTRForwardingPattern>(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTWForwardingPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(forwardPattern));
}

static void applyLinalgToVectorPatterns(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  auto *ctx = funcOp.getContext();
  patterns.add<LinalgVectorizationPattern>(
      ctx, LinalgTransformationFilter()
               .addOpFilter<ContractionOpInterface, FillOp, GenericOp>());
  patterns.add<CopyVectorizationPattern>(ctx);
  populatePadOpVectorizationPatterns(patterns);
  populateConvolutionVectorizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyPadTensorToGenericPatterns(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<PadOpTransformationPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyGeneralizePadTensorPatterns(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<GeneralizePadOpPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyExtractSliceOfPadTensorSwapPattern(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<ExtractSliceOfPadTensorSwapPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyTilePattern(FuncOp funcOp, const std::string &loopType,
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

/// Apply transformations specified as patterns.
void TestLinalgTransforms::runOnOperation() {
  auto lambda = [&](void *) {
    getOperation().walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  };
  std::unique_ptr<void, decltype(lambda)> cleanupGuard{(void *)1, lambda};

  if (testPromotionOptions) {
    RewritePatternSet patterns(&getContext());
    fillPromotionCallBackPatterns(&getContext(), patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
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
  if (testMatmulToVectorPatterns1dTiling || testMatmulToVectorPatterns2dTiling)
    return applyMatmulToVectorPatterns(getOperation(),
                                       testMatmulToVectorPatterns1dTiling,
                                       testMatmulToVectorPatterns2dTiling);
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
}

namespace mlir {
namespace test {
void registerTestLinalgTransforms() {
  PassRegistration<TestLinalgTransforms>();
}
} // namespace test
} // namespace mlir
