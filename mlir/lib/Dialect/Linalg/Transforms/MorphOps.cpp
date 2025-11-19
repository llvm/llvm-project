//===- MorphOps.cpp - conversion between named,category and generic ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversions between linalg ops:
//    named <--> category (elementwise, contraction, ..) <--> generic.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGMORPHOPSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-morphism"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct LinalgMorphOpsPass
    : public impl::LinalgMorphOpsPassBase<LinalgMorphOpsPass> {

  using impl::LinalgMorphOpsPassBase<
      LinalgMorphOpsPass>::LinalgMorphOpsPassBase;

  void runOnOperation() override;
};

void LinalgMorphOpsPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());

  // Lowering paths (named -> category -> generic)
  if (namedToCategory) {
    populateLinalgNamedToElementwisePatterns(patterns);
  }
  if (namedToGeneric || categoryToGeneric) {
    populateLinalgNamedOpsGeneralizationPatterns(patterns);
  }

  // Lifting paths (named <- category <- generic)
  if (genericToNamed) {
    populateLinalgGenericOpsSpecializationPatterns(patterns);
  }

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // namespace
