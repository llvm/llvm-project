//===- TestPadFusion.cpp - Test fusion of pad op with Linalg ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing fusion of pad ops with its producer
// Linalg op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestPadFusionPass
    : public PassWrapper<TestPadFusionPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPadFusionPass)

  TestPadFusionPass() = default;
  TestPadFusionPass(const TestPadFusionPass &pass) : PassWrapper(pass) {}

  Option<bool> fillBoundaryOnly{
      *this, "fill-boundary-only",
      llvm::cl::desc("Fill only the padded boundary instead of the whole "
                     "result, for static pads whose producer overwrites the "
                     "interior"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  StringRef getArgument() const final { return "test-linalg-pad-fusion"; }
  StringRef getDescription() const final { return "Test PadOp fusion"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateFuseTensorPadWithProducerLinalgOpPatterns(patterns,
                                                              fillBoundaryOnly);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestPadFusion() { PassRegistration<TestPadFusionPass>(); }
} // namespace test
} // namespace mlir
