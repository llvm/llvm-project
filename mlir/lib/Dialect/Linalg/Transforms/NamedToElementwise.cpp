//===- NamedToElementwise.cpp - convert linalg named op into elementwise --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting those linalg named ops that are essentially
// elementwise e.g. `linalg.exp`, to `linalg.elementwise`. This allows further
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
      .Case([](AddOp) { return ElementwiseKind::add; })
      .Case([](SubOp) { return ElementwiseKind::sub; })
      .Case([](MulOp) { return ElementwiseKind::mul; })
      .Case([](DivOp) { return ElementwiseKind::div; })
      .Case([](DivUnsignedOp) { return ElementwiseKind::div_unsigned; })
      .Case([](PowFOp) { return ElementwiseKind::powf; })
      .Case([](ExpOp) { return ElementwiseKind::exp; })
      .Case([](LogOp) { return ElementwiseKind::log; })
      .Case([](AbsOp) { return ElementwiseKind::abs; })
      .Case([](CeilOp) { return ElementwiseKind::ceil; })
      .Case([](FloorOp) { return ElementwiseKind::floor; })
      .Case([](NegFOp) { return ElementwiseKind::negf; })
      .Case([](ReciprocalOp) { return ElementwiseKind::reciprocal; })
      .Case([](RoundOp) { return ElementwiseKind::round; })
      .Case([](SqrtOp) { return ElementwiseKind::sqrt; })
      .Case([](RsqrtOp) { return ElementwiseKind::rsqrt; })
      .Case([](SquareOp) { return ElementwiseKind::square; })
      .Case([](TanhOp) { return ElementwiseKind::tanh; })
      .Case([](ErfOp) { return ElementwiseKind::erf; })
      .Default([&](Operation *op) {
        llvm_unreachable("unhandled case in named to elementwise");
        return ElementwiseKind::sub;
      });
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
  patterns.add<NamedToElementwisePattern<AddOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<SubOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<MulOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<DivOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<DivUnsignedOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<PowFOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<ExpOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<LogOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<AbsOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<CeilOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<FloorOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<NegFOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<ReciprocalOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<RoundOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<SqrtOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<RsqrtOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<SquareOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<TanhOp>>(patterns.getContext());
  patterns.add<NamedToElementwisePattern<ErfOp>>(patterns.getContext());
}
