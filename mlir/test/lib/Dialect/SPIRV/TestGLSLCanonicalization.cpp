//===- TestGLSLCanonicalization.cpp - Pass to test GLSL-specific pattterns ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVGLSLCanonicalization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class TestGLSLCanonicalizationPass
    : public PassWrapper<TestGLSLCanonicalizationPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  TestGLSLCanonicalizationPass() = default;
  TestGLSLCanonicalizationPass(const TestGLSLCanonicalizationPass &) {}
  void runOnOperation() override;
};
} // namespace

void TestGLSLCanonicalizationPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  spirv::populateSPIRVGLSLCanonicalizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
void registerTestSpirvGLSLCanonicalizationPass() {
  PassRegistration<TestGLSLCanonicalizationPass> registration(
      "test-spirv-glsl-canonicalization",
      "Tests SPIR-V canonicalization patterns for GLSL extension.");
}
} // namespace mlir
