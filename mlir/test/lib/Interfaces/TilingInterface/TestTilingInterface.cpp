//===- TestTilingInterface.cpp - Test tiling using `TilingInterface` -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing tiling operations using
// `TilingInterface`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

/// Pattern for testing `TileUsingSCFForOp` pattern (that tiles operations using
/// the `TilingInterface` with `scf.for` ops for iterating over the tiles) while
/// using a `filter` to avoid recursive application.
struct TestTileUsingSCFForOpWithFilter : public scf::TileUsingSCFForOp {
  TestTileUsingSCFForOpWithFilter(MLIRContext *context,
                                  scf::SCFTilingOptions options,
                                  linalg::LinalgTransformationFilter filter =
                                      linalg::LinalgTransformationFilter(),
                                  PatternBenefit benefit = 1)
      : scf::TileUsingSCFForOp(context, options, benefit), filter(filter) {}

  /// Construct a generic pattern applied to `opName`.
  TestTileUsingSCFForOpWithFilter(StringRef opName, MLIRContext *context,
                                  scf::SCFTilingOptions options,
                                  linalg::LinalgTransformationFilter filter =
                                      linalg::LinalgTransformationFilter(),
                                  PatternBenefit benefit = 1)
      : scf::TileUsingSCFForOp(context, options, benefit), filter(filter) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();

    auto tilingResult = returningMatchAndRewrite(op, rewriter);
    if (failed(tilingResult)) {
      return failure();
    }
    filter.replaceLinalgTransformationFilter(rewriter, tilingResult->tiledOp);
    return success();
  }

private:
  linalg::LinalgTransformationFilter filter;
};

/// Pattern for testing `TileConsumerAndFUseProducersUsingSCFForOp` pattern
/// (that tiles and fuses operations using the `TilingInterface` with `scf.for`
/// ops for iterating over the tiles) while using a `filter` to avoid recursive
/// application.
struct TestTileConsumerAndFuseProducersUsingSCFForOpWithFilter
    : public scf::TileConsumerAndFuseProducersUsingSCFForOp {
  TestTileConsumerAndFuseProducersUsingSCFForOpWithFilter(
      MLIRContext *context, scf::SCFTilingOptions options,
      linalg::LinalgTransformationFilter filter =
          linalg::LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : scf::TileConsumerAndFuseProducersUsingSCFForOp(context, options,
                                                       benefit),
        filter(filter) {}

  /// Construct a generic pattern applied to `opName`.
  TestTileConsumerAndFuseProducersUsingSCFForOpWithFilter(
      StringRef opName, MLIRContext *context, scf::SCFTilingOptions options,
      linalg::LinalgTransformationFilter filter =
          linalg::LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : scf::TileConsumerAndFuseProducersUsingSCFForOp(context, options,
                                                       benefit),
        filter(filter) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();

    auto tileAndFuseResult = returningMatchAndRewrite(op, rewriter);
    if (failed(tileAndFuseResult)) {
      return failure();
    }
    filter.replaceLinalgTransformationFilter(
        rewriter, tileAndFuseResult->tiledAndFusedOps.front());
    return success();
  }

private:
  linalg::LinalgTransformationFilter filter;
};

/// Test pass for testing the use of `TilingInterface`.
struct TestTilingInterfacePass
    : public PassWrapper<TestTilingInterfacePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTilingInterfacePass)

  TestTilingInterfacePass() = default;
  TestTilingInterfacePass(const TestTilingInterfacePass &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }
  StringRef getArgument() const final { return "test-tiling-interface"; }
  StringRef getDescription() const final {
    return "Test tiling using TilingInterface";
  }

  Option<bool> testTiling{
      *this, "tile-using-scf-for",
      llvm::cl::desc(
          "Test tiling using TilingInterface with scf.for operations"),
      llvm::cl::init(false)};

  Option<bool> testTileConsumerAndFuseProducer{
      *this, "tile-consumer-and-fuse-producer-using-scf-for",
      llvm::cl::desc("Test tile and fuse transformation using TilingInterface "
                     "with scf.for operations"),
      llvm::cl::init(false)};

  void runOnOperation() override;

private:
  void addTestPatterns(MLIRContext *context, RewritePatternSet &patterns);
};
} // namespace

template <class Pattern>
static void
addPatternForTiling(MLIRContext *context, ArrayRef<int64_t> tileSizes,
                    StringRef filterName, RewritePatternSet &patterns) {
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  linalg::LinalgTransformationFilter filter(
      StringAttr::get(context, filterName), StringAttr::get(context, "tiled"));
  patterns.add<Pattern>(context, tilingOptions, filter);
}

void TestTilingInterfacePass::addTestPatterns(MLIRContext *context,
                                              RewritePatternSet &patterns) {
  if (testTiling) {
    // 1. Tiling M and N dims of `linalg.matmul` on tensors.
    addPatternForTiling<TestTileUsingSCFForOpWithFilter>(
        context, {10, 20}, "simple_gemm", patterns);
    // 2. Tiling M, N and K of `linalg.matmul` on buffers.
    addPatternForTiling<TestTileUsingSCFForOpWithFilter>(
        context, {10, 20, 30}, "simple_gemm_memref", patterns);
    // 3. Tiling 3D parallel generic op which implements a transpose
    addPatternForTiling<TestTileUsingSCFForOpWithFilter>(
        context, {10, 0, 20}, "parallel_generic_transpose", patterns);
    // 4. Tiling 2D conv op.
    addPatternForTiling<TestTileUsingSCFForOpWithFilter>(
        context, {0, 0, 0, 0, 10, 20, 30}, "simple_conv", patterns);
    // 5. Tiling a simple op with `linalg.index` inside.
    addPatternForTiling<TestTileUsingSCFForOpWithFilter>(
        context, {10, 20}, "indexed_semantics", patterns);
    return;
  }
  if (testTileConsumerAndFuseProducer) {
    // 1. Tile and fuse of gemm with bias-add operation.
    addPatternForTiling<
        TestTileConsumerAndFuseProducersUsingSCFForOpWithFilter>(
        context, {10, 20}, "fusion", patterns);
    addPatternForTiling<
        TestTileConsumerAndFuseProducersUsingSCFForOpWithFilter>(
        context, {10}, "gemm_fusion", patterns);
    return;
  }
}

void TestTilingInterfacePass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet tilingPatterns(context);
  addTestPatterns(context, tilingPatterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(tilingPatterns))))
    return signalPassFailure();
}

namespace mlir {
namespace test {
void registerTestTilingInterface() {
  PassRegistration<TestTilingInterfacePass>();
}
} // namespace test
} // namespace mlir
