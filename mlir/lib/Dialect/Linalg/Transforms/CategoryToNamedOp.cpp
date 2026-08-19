//===- CategoryToNamedOp.cpp - convert category ops to linalg named ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting of linalg category ops (e.g.
// `linalg.elementwise`) to their equivalent named ops (e.g. `linalg.add`,
// `linalg.exp`). This is the reverse of NamedToElementwise.cpp.
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
    case ElementwiseKind::exp:
      return replaceWith(ExpOp{});
    case ElementwiseKind::log:
      return replaceWith(LogOp{});
    case ElementwiseKind::abs:
      return replaceWith(AbsOp{});
    case ElementwiseKind::ceil:
      return replaceWith(CeilOp{});
    case ElementwiseKind::floor:
      return replaceWith(FloorOp{});
    case ElementwiseKind::negf:
      return replaceWith(NegFOp{});
    case ElementwiseKind::reciprocal:
      return replaceWith(ReciprocalOp{});
    case ElementwiseKind::round:
      return replaceWith(RoundOp{});
    case ElementwiseKind::sqrt:
      return replaceWith(SqrtOp{});
    case ElementwiseKind::rsqrt:
      return replaceWith(RsqrtOp{});
    case ElementwiseKind::square:
      return replaceWith(SquareOp{});
    case ElementwiseKind::tanh:
      return replaceWith(TanhOp{});
    case ElementwiseKind::erf:
      return replaceWith(ErfOp{});
    case ElementwiseKind::add:
      return replaceWith(AddOp{});
    case ElementwiseKind::sub:
      return replaceWith(SubOp{});
    case ElementwiseKind::mul:
      return replaceWith(MulOp{});
    case ElementwiseKind::div:
      return replaceWith(DivOp{});
    case ElementwiseKind::div_unsigned:
      return replaceWith(DivUnsignedOp{});
    case ElementwiseKind::max_signed:
      return replaceWith(MaxOp{});
    case ElementwiseKind::min_signed:
      return replaceWith(MinOp{});
    case ElementwiseKind::powf:
      return replaceWith(PowFOp{});
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
