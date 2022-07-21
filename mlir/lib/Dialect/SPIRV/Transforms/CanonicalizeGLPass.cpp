//===- CanonicalizeGLPass.cpp - GLSL Related Canonicalization Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVGLCanonicalization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class CanonicalizeGLPass final
    : public SPIRVCanonicalizeGLBase<CanonicalizeGLPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    spirv::populateSPIRVGLCanonicalizationPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<>> spirv::createCanonicalizeGLPass() {
  return std::make_unique<CanonicalizeGLPass>();
}
