//===- NamedToElementwise.cpp - convert linalg named op into elementwise --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting those linalg named ops that are essentially
// elementwise e.g. `linalg.add`, to `linalg.elementwise`. This allows further
// optimization on `linalg.elementwise` such as folding transpose, broadcast.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-named-to-elementwise"

namespace {
ElementwiseKind getKind(Operation *op) {
  return llvm::TypeSwitch<Operation *, ElementwiseKind>(op)
      .Case([](SelectOp) { return ElementwiseKind::select; })
      .DefaultUnreachable("unhandled case in named to elementwise");
}

template <typename NamedOpTy>
struct NamedToElementwisePattern : public OpRewritePattern<NamedOpTy> {
  using OpRewritePattern<NamedOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(NamedOpTy op,
                                PatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attrs;
    auto kindAttr = ElementwiseKindAttr::get(op.getContext(), getKind(op));
    attrs.push_back(rewriter.getNamedAttr("kind", kindAttr));
    attrs.push_back(
        rewriter.getNamedAttr("indexing_maps", op.getIndexingMaps()));

    rewriter.replaceOpWithNewOp<ElementwiseOp>(op, op.getDpsInputs(),
                                               op.getDpsInits(), attrs);
    return success();
  }
};
} // namespace

void mlir::linalg::populateLinalgNamedToElementwisePatterns(
    RewritePatternSet &patterns) {
  patterns.add<NamedToElementwisePattern<SelectOp>>(patterns.getContext());
}
