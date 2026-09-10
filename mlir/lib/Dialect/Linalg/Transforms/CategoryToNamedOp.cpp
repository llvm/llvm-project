//===- CategoryToNamedOp.cpp - convert category ops to linalg named ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting of linalg category ops (e.g.
// `linalg.elementwise`) to their equivalent named ops (e.g. `linalg.add`).
// This is the reverse of NamedToElementwise.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-category-to-named"

namespace {
struct ElementwiseToNamedPattern : public OpRewritePattern<ElementwiseOp> {
  using OpRewritePattern<ElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    // Named elementwise ops only support identity indexing maps.
    if (!op.getIndexingMapsArray().empty() &&
        !llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }))
      return failure();

    auto inputs = op.getDpsInputs();
    auto inits = op.getDpsInits();
    auto loc = op.getLoc();

    // Helper to create a named op and replace the elementwise op.
    auto replaceWith = [&](auto namedOp) {
      using OpTy = decltype(namedOp);
      rewriter.replaceOp(op, OpTy::create(rewriter, loc, inputs, inits,
                                          ArrayRef<NamedAttribute>{}));
      return success();
    };

    switch (op.getKind()) {
    case ElementwiseKind::select:
      return replaceWith(SelectOp{});
    default:
      return failure();
    }
  }
};
} // namespace

void mlir::linalg::populateLinalgCategoryToNamedPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ElementwiseToNamedPattern>(patterns.getContext());
}
