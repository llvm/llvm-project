//===- CategoryToNamed.cpp - convert linalg category ops into named ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting the subset of linalg category ops that can be
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

template <typename NamedOpTy>
static LogicalResult replaceElementwiseOp(ElementwiseOp op,
                                          PatternRewriter &rewriter) {
  unsigned numMaps = op.getNumDpsInputs() + op.getNumDpsInits();
  SmallVector<AffineMap> defaultMaps = ElementwiseOp::getDefaultIndexingMaps(
      numMaps, op.getResultRank(), op.getContext());
  if (!llvm::equal(op.getIndexingMapsArray(), defaultMaps))
    return rewriter.notifyMatchFailure(
        op, "named elementwise ops require default indexing maps");

  auto namedOp =
      NamedOpTy::create(rewriter, op.getLoc(), op.getDpsInputs(),
                        op.getDpsInits(), ArrayRef<NamedAttribute>{});
  rewriter.replaceOp(op, namedOp->getResults());
  return success();
}

template <typename NamedOpTy>
static LogicalResult replaceContractOp(ContractOp op, PatternRewriter &rewriter,
                                       ArrayRef<AffineMap> indexingMaps) {
  SmallVector<NamedAttribute> attrs;
  if (op.getCast() == TypeFn::cast_unsigned) {
    attrs.push_back(rewriter.getNamedAttr(
        "cast", TypeFnAttr::get(rewriter.getContext(), op.getCast())));
  }

  auto indexingMapsAttr = rewriter.getAffineMapArrayAttr(indexingMaps);
  if (!NamedOpTy::isDefaultIndexingMaps(indexingMapsAttr)) {
    attrs.push_back(rewriter.getNamedAttr("indexing_maps", indexingMapsAttr));
  }

  auto namedOp = NamedOpTy::create(rewriter, op.getLoc(), op.getDpsInputs(),
                                   op.getDpsInits(), attrs);
  rewriter.replaceOp(op, namedOp->getResults());
  return success();
}

static LogicalResult specializeElementwiseOp(ElementwiseOp op,
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
    // There are no named unsigned max/min ops yet, so these category ops
    // cannot currently be represented as named ops.
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

static LogicalResult specializeContractOp(ContractOp op,
                                          PatternRewriter &rewriter) {
  FailureOr<SmallVector<AffineMap, 3>> namedOpMaps =
      inferMatmulLikeIndexingMaps(op);
  if (failed(namedOpMaps))
    return failure();

  if ((*namedOpMaps)[0].getNumDims() > 3)
    return replaceContractOp<BatchMatmulOp>(op, rewriter, *namedOpMaps);
  return replaceContractOp<MatmulOp>(op, rewriter, *namedOpMaps);
}

namespace {

struct ElementwiseToNamedPattern : public OpRewritePattern<ElementwiseOp> {
  using OpRewritePattern<ElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    return specializeElementwiseOp(op, rewriter);
  }
};

struct ContractToNamedPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractOp op,
                                PatternRewriter &rewriter) const override {
    return specializeContractOp(op, rewriter);
  }
};

} // namespace

void mlir::linalg::populateLinalgCategoryToNamedPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ElementwiseToNamedPattern, ContractToNamedPattern>(
      patterns.getContext());
}
