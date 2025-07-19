//===- MorphOps.cpp - conversion between named,category and generic ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversions between:
//    named <--> category (elementwise, contraction, ..) <--> generic ops.
//
// For example, a named op such `linalg.add` can also be re-written as an
// equivalent category op `linalg.elementwise` and also as a `linalg.generic`.
//
// Generic is a bigger set than named ops and so not all generics can be
// converted to single category-op or named-op. Similarly, category-ops
// are bigger in representational possiblities than named ops e.g.
// `linalg.add` has no affine maps attached, but `linalg.elementwise` does.
//
// Note:
//  Legacy converters (will be deprecated):
//    `--linalg-generalize-named-ops` is the path `named-op --> generic-op`
//    `--linalg-specialize-generic-ops` is the path `named-op <-- generic-op`
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
    // TODO: named -> contraction-op
    populateLinalgNamedToElementwisePatterns(patterns);
  }
  if (namedToGeneric || categoryToGeneric) {
    populateLinalgNamedOpsGeneralizationPatterns(patterns);
  }

  // Lifting paths (named <- category <- generic)
  if (genericToCategory) {
    // TODO.
  }
  if (categoryToNamed) {
    // TODO: if there is a case for this.
  }
  if (genericToNamed) {
    populateLinalgGenericOpsSpecializationPatterns(patterns);
  }

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // namespace
