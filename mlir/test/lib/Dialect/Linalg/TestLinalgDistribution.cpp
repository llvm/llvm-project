//===- TestLinalgDistribution.cpp - Test Linalg hoisting functions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg hoisting functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

template <gpu::Dimension Dim>
static linalg::ProcInfo getGpuBlockInfo(OpBuilder &b, Location loc) {
  Type indexType = b.getIndexType();
  ProcInfo procInfo = {b.create<gpu::BlockIdOp>(loc, indexType, Dim),
                       b.create<gpu::GridDimOp>(loc, indexType, Dim)};
  return procInfo;
}

static LinalgLoopDistributionOptions getDistributionOptions() {
  LinalgLoopDistributionOptions opts;
  opts.procInfoMap.insert(
      std::make_pair("block_x", getGpuBlockInfo<gpu::Dimension::x>));
  opts.procInfoMap.insert(
      std::make_pair("block_y", getGpuBlockInfo<gpu::Dimension::y>));
  return opts;
}

namespace {
struct TestLinalgDistribution
    : public PassWrapper<TestLinalgDistribution, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-linalg-distribution"; }
  StringRef getDescription() const final { return "Test Linalg distribution."; }
  TestLinalgDistribution() = default;
  TestLinalgDistribution(const TestLinalgDistribution &pass) = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void TestLinalgDistribution::runOnOperation() {
  auto funcOp = getOperation();
  RewritePatternSet distributeTiledLoopsPatterns(&getContext());
  populateLinalgDistributeTiledLoopPattern(
      distributeTiledLoopsPatterns, getDistributionOptions(),
      LinalgTransformationFilter(
          ArrayRef<StringAttr>{},
          {StringAttr::get(funcOp.getContext(), "distributed")})
          .addFilter([](Operation *op) {
            return success(!op->getParentOfType<linalg::TiledLoopOp>());
          }));
  (void)applyPatternsAndFoldGreedily(funcOp,
                                     std::move(distributeTiledLoopsPatterns));
  // Ensure we drop the marker in the end.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

namespace mlir {
namespace test {
void registerTestLinalgDistribution() {
  PassRegistration<TestLinalgDistribution>();
}
} // namespace test
} // namespace mlir
