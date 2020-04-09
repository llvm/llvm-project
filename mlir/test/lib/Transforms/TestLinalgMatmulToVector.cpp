//===- TestLinalgMatmulToVector.cpp - Test VectorTransfers lowering -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;

namespace {
#include "TestLinalgMatmulToVectorPatterns.h.inc"

struct DeclarativeTransforms
    : public PassWrapper<DeclarativeTransforms, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    AffineApplyOp::getCanonicalizationPatterns(patterns, context);
    AffineMinOp::getCanonicalizationPatterns(patterns, context);
    AffineMaxOp::getCanonicalizationPatterns(patterns, context);
    AllocOp::getCanonicalizationPatterns(patterns, context);
    SubViewOp::getCanonicalizationPatterns(patterns, context);
    ViewOp::getCanonicalizationPatterns(patterns, context);
    populateWithGenerated(context, &patterns);
    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end anonymous namespace

namespace mlir {
void registerTestLinalgMatmulToVectorPass() {
  PassRegistration<DeclarativeTransforms> pass(
      "linalg-matmul-to-vector",
      "Test declarative transform patterns for matmul 3-D tiling + promotion"
      " + vectorization");
}
} // namespace mlir
