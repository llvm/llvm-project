//===- CanonicalizeGLPass.cpp - GLSL Related Canonicalization Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVGLCanonicalization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVCANONICALIZEGLPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

namespace {
class CanonicalizeGLPass final
    : public spirv::impl::SPIRVCanonicalizeGLPassBase<CanonicalizeGLPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    spirv::populateSPIRVGLCanonicalizationPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
