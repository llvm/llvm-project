//===- TestLinalgRankReduceContractionOps.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing rank reduing patterns for named
// contraction ops with unit dims.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestLinalgRankReduceContractionOps
    : public PassWrapper<TestLinalgRankReduceContractionOps,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestLinalgRankReduceContractionOps)

  TestLinalgRankReduceContractionOps() = default;
  TestLinalgRankReduceContractionOps(
      const TestLinalgRankReduceContractionOps &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }
  StringRef getArgument() const final {
    return "test-linalg-rank-reduce-contraction-ops";
  }
  StringRef getDescription() const final {
    return "Test Linalg rank reduce contraction ops with unit dims";
  }

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    func::FuncOp funcOp = this->getOperation();

    RewritePatternSet patterns(context);
    linalg::populateContractionOpRankReducingPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp.getBody(), std::move(patterns))))
      return signalPassFailure();
    return;
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLinalgRankReduceContractionOps() {
  PassRegistration<TestLinalgRankReduceContractionOps>();
}
} // namespace test
} // namespace mlir
