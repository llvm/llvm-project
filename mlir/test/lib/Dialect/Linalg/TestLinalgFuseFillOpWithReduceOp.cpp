//===- TestLinalgFuseFillOpWithReduceOp.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing fuse linalg fill with linalg reduce
// into a new linalg generic operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestLinalgFuseFillOpWithReduceOp
    : public PassWrapper<TestLinalgFuseFillOpWithReduceOp,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgFuseFillOpWithReduceOp)

  TestLinalgFuseFillOpWithReduceOp() = default;
  TestLinalgFuseFillOpWithReduceOp(
      const TestLinalgFuseFillOpWithReduceOp &pass) = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }
  StringRef getArgument() const final {
    return "test-linalg-fuse-fill-op-with-reduce-op";
  }
  StringRef getDescription() const final {
    return "Test fuse linalg fill with linalg reduce into a new linalg generic "
           "operation";
  }

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    func::FuncOp funcOp = this->getOperation();

    RewritePatternSet patterns(context);
    linalg::populateFuseFillOpWithReduceOpPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp.getBody(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLinalgFuseFillOpWithReduceOp() {
  PassRegistration<TestLinalgFuseFillOpWithReduceOp>();
}
} // namespace test
} // namespace mlir
