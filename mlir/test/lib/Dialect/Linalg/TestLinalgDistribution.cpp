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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

template <char dim>
static linalg::ProcInfo getGpuBlockInfo(OpBuilder &b, Location loc) {
  std::string d(1, dim);
  StringAttr attr = b.getStringAttr(d);

  Type indexType = b.getIndexType();
  ProcInfo procInfo = {b.create<gpu::BlockIdOp>(loc, indexType, attr),
                       b.create<gpu::GridDimOp>(loc, indexType, attr)};
  return procInfo;
}

static LinalgLoopDistributionOptions getDistributionOptions() {
  LinalgLoopDistributionOptions opts;
  opts.procInfoMap.insert(std::make_pair("block_x", getGpuBlockInfo<'x'>));
  opts.procInfoMap.insert(std::make_pair("block_y", getGpuBlockInfo<'y'>));
  return opts;
}

namespace {
struct TestLinalgDistribution
    : public PassWrapper<TestLinalgDistribution, FunctionPass> {
  TestLinalgDistribution() = default;
  TestLinalgDistribution(const TestLinalgDistribution &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }

  void runOnFunction() override;
};
} // namespace

void TestLinalgDistribution::runOnFunction() {
  auto funcOp = getFunction();
  OwningRewritePatternList distributeTiledLoopsPatterns(&getContext());
  populateLinalgDistributeTiledLoopPattern(
      distributeTiledLoopsPatterns, getDistributionOptions(),
      LinalgTransformationFilter(
          ArrayRef<Identifier>{},
          {Identifier::get("distributed", funcOp.getContext())})
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
  PassRegistration<TestLinalgDistribution> testTestLinalgDistributionPass(
      "test-linalg-distribution", "Test Linalg distribution.");
}
} // namespace test
} // namespace mlir
