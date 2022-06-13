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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

/// Construct a generic pattern applied to all TilingInterface ops that verify
/// `filter`.
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

    FailureOr<scf::SCFTilingResult> tilingResult =
        returningMatchAndRewrite(op, rewriter);
    if (failed(tilingResult)) {
      return failure();
    }
    filter.replaceLinalgTransformationFilter(rewriter, tilingResult->tiledOp);
    return success();
  }

private:
  linalg::LinalgTransformationFilter filter;
};

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

  void runOnOperation() override;
};
} // namespace

static void addTestPatterns(MLIRContext *context, RewritePatternSet &patterns) {
  auto addPatternForTiling = [&](ArrayRef<int64_t> tileSizes,
                                 StringRef filterName) {
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    linalg::LinalgTransformationFilter filter(
        StringAttr::get(context, filterName),
        StringAttr::get(context, "tiled"));
    patterns.add<TestTileUsingSCFForOpWithFilter>(context, tilingOptions,
                                                  filter);
  };
  // 1. Tiling M and N dims of `linalg.matmul` on tensors.
  addPatternForTiling({10, 20}, "simple_gemm");
  // 2. Tiling M, N and K of `linalg.matmul` on buffers.
  addPatternForTiling({10, 20, 30}, "simple_gemm_memref");
  // 3. Tiling 3D parallel generic op which implements a transpose
  addPatternForTiling({10, 0, 20}, "parallel_generic_transpose");
  // 4. Tiling 2D conv op.
  addPatternForTiling({0, 0, 0, 0, 10, 20, 30}, "simple_conv");
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
