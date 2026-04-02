//===- CategoryToNamed.cpp - convert linalg category ops into named ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting those linalg category ops that can be
// represented by named ops, e.g. `linalg.elementwise<exp>` to `linalg.exp` or
// `linalg.contract` to `linalg.matmul`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-category-to-named"

namespace {

template <typename NamedOpTy>
static FailureOr<LinalgOp> replaceElementwiseOp(ElementwiseOp op,
                                                PatternRewriter &rewriter) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("indexing_maps", op.getIndexingMaps()));

  auto namedOp = NamedOpTy::create(rewriter, op.getLoc(), op.getDpsInputs(),
                                   op.getDpsInits(), attrs);

  {
    ScopedDiagnosticHandler handler(op.getContext(), [](Diagnostic &) {});
    if (failed(verify(namedOp.getOperation()))) {
      rewriter.eraseOp(namedOp);
      return rewriter.notifyMatchFailure(
          op, "elementwise op does not satisfy named op constraints");
    }
  }

  rewriter.replaceOp(op, namedOp->getResults());
  return cast<LinalgOp>(namedOp.getOperation());
}

static FailureOr<LinalgOp> specializeElementwiseOp(ElementwiseOp op,
                                                   PatternRewriter &rewriter) {
  switch (op.getKind()) {
  case ElementwiseKind::select:
    return replaceElementwiseOp<SelectOp>(op, rewriter);
  case ElementwiseKind::add:
    return replaceElementwiseOp<AddOp>(op, rewriter);
  case ElementwiseKind::sub:
    return replaceElementwiseOp<SubOp>(op, rewriter);
  case ElementwiseKind::mul:
    return replaceElementwiseOp<MulOp>(op, rewriter);
  case ElementwiseKind::div:
    return replaceElementwiseOp<DivOp>(op, rewriter);
  case ElementwiseKind::div_unsigned:
    return replaceElementwiseOp<DivUnsignedOp>(op, rewriter);
  case ElementwiseKind::max_signed:
    return replaceElementwiseOp<MaxOp>(op, rewriter);
  case ElementwiseKind::min_signed:
    return replaceElementwiseOp<MinOp>(op, rewriter);
  case ElementwiseKind::max_unsigned:
  case ElementwiseKind::min_unsigned:
    break;
  case ElementwiseKind::powf:
    return replaceElementwiseOp<PowFOp>(op, rewriter);
  case ElementwiseKind::exp:
    return replaceElementwiseOp<ExpOp>(op, rewriter);
  case ElementwiseKind::log:
    return replaceElementwiseOp<LogOp>(op, rewriter);
  case ElementwiseKind::abs:
    return replaceElementwiseOp<AbsOp>(op, rewriter);
  case ElementwiseKind::ceil:
    return replaceElementwiseOp<CeilOp>(op, rewriter);
  case ElementwiseKind::floor:
    return replaceElementwiseOp<FloorOp>(op, rewriter);
  case ElementwiseKind::negf:
    return replaceElementwiseOp<NegFOp>(op, rewriter);
  case ElementwiseKind::reciprocal:
    return replaceElementwiseOp<ReciprocalOp>(op, rewriter);
  case ElementwiseKind::round:
    return replaceElementwiseOp<RoundOp>(op, rewriter);
  case ElementwiseKind::sqrt:
    return replaceElementwiseOp<SqrtOp>(op, rewriter);
  case ElementwiseKind::rsqrt:
    return replaceElementwiseOp<RsqrtOp>(op, rewriter);
  case ElementwiseKind::square:
    return replaceElementwiseOp<SquareOp>(op, rewriter);
  case ElementwiseKind::tanh:
    return replaceElementwiseOp<TanhOp>(op, rewriter);
  case ElementwiseKind::erf:
    return replaceElementwiseOp<ErfOp>(op, rewriter);
  }

  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << "unsupported elementwise kind for named specialization: "
         << stringifyElementwiseKind(op.getKind());
  });
}

struct ElementwiseToNamedPattern : public OpRewritePattern<ElementwiseOp> {
  using OpRewritePattern<ElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    return succeeded(specializeElementwiseOp(op, rewriter)) ? success()
                                                            : failure();
  }
};

struct ContractToNamedPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractOp op,
                                PatternRewriter &rewriter) const override {
    // Route through a cloned generic op so we can reuse the existing
    // contraction-to-named specialization without mutating the original op
    // on unsuccessful matches.
    auto *clonedOp = rewriter.clone(*op.getOperation());
    auto clonedLinalgOp = cast<LinalgOp>(clonedOp);

    FailureOr<GenericOp> genericOp =
        generalizeNamedOp(rewriter, clonedLinalgOp);
    if (failed(genericOp)) {
      rewriter.eraseOp(clonedOp);
      return failure();
    }

    GenericOpSpecializationOptions options;
    FailureOr<LinalgOp> namedOp =
        specializeGenericOp(rewriter, *genericOp, options);
    if (failed(namedOp)) {
      rewriter.eraseOp(*genericOp);
      return failure();
    }

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOp(op, (*namedOp)->getResults());
    return success();
  }
};

} // namespace

void mlir::linalg::populateLinalgCategoryToNamedPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ElementwiseToNamedPattern, ContractToNamedPattern>(
      patterns.getContext());
}
