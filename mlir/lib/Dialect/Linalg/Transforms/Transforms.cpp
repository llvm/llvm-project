//===- LinalgTransforms.cpp - Linalg transformations as patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic and helpers to expose Linalg transforms as rewrite
// patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

#define DEBUG_TYPE "linalg-transforms"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using llvm::dbgs;

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//
// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral mlir::linalg::LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

mlir::linalg::LinalgMarker::LinalgMarker(ArrayRef<StringRef> matchDisjunction,
                                         llvm::Optional<StringRef> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement) {}

mlir::linalg::LinalgMarker::LinalgMarker(ArrayRef<StringRef> matchDisjunction,
                                         StringRef replacement)
    : LinalgMarker(matchDisjunction, llvm::Optional<StringRef>{replacement}) {}

LogicalResult
mlir::linalg::LinalgMarker::checkAndNotify(PatternRewriter &rewriter,
                                           Operation *op) const {
  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no marker case and matchDisjunction is empty.
    if (matchDisjunction.empty())
      return success();

    // 2. Has no marker and matchDisjuntion matches the no-moarker case.
    for (auto marker : matchDisjunction)
      if (marker.empty())
        return success();

    // 3. Has no marker but was expecting a marker.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any marker from list: ";
      llvm::interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit marker.
  for (auto marker : matchDisjunction)
    if (attr.getValue() == marker)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any marker from list: ";
    llvm::interleaveComma(matchDisjunction, diag);
  });
}

void mlir::linalg::LinalgMarker::replaceLinalgMarker(PatternRewriter &rewriter,
                                                     Operation *op) const {
  if (replacement.hasValue())
    op->setAttr(LinalgTransforms::kLinalgTransformMarker,
                rewriter.getStringAttr(replacement.getValue()));
  else
    op->removeAttr(Identifier::get(LinalgTransforms::kLinalgTransformMarker,
                                   rewriter.getContext()));
}

/// Linalg base tiling pattern.
mlir::linalg::LinalgBaseTilingPattern::LinalgBaseTilingPattern(
    StringRef opName, MLIRContext *context, LinalgTilingOptions options,
    LinalgMarker marker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      options(options) {}

LogicalResult mlir::linalg::LinalgBaseTilingPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();
  Optional<TiledLinalgOp> res;
  if (options.loopType == LinalgTilingLoopType::Loops)
    res = tileLinalgOp(rewriter, linalgOp, options.tileSizes,
                       options.interchangeVector);
  else if (options.loopType == LinalgTilingLoopType::ParallelLoops)
    res = tileLinalgOpToParallelLoops(rewriter, linalgOp, options.tileSizes,
                                      options.interchangeVector);
  // TODO: Impl tiling to affine loops when it makes sense.

  if (!res)
    return failure();

  // New marker if specified.
  marker.replaceLinalgMarker(rewriter, res->op.getOperation());

  rewriter.eraseOp(op);
  return success();
}

/// Linalg base interchange pattern.
mlir::linalg::LinalgBaseInterchangePattern::LinalgBaseInterchangePattern(
    StringRef opName, MLIRContext *context,
    ArrayRef<unsigned> interchangeVector, LinalgMarker marker,
    PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      interchangeVector(interchangeVector.begin(), interchangeVector.end()) {}

LogicalResult mlir::linalg::LinalgBaseInterchangePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();
  if (failed(interchangeGenericLinalgOpPrecondition(op, interchangeVector)))
    return failure();

  // TODO: figure out how this interplays with named ops. In particular this
  // should break the named op property.
  rewriter.updateRootInPlace(op, [&]() {
    interchange(linalgOp, interchangeVector);
    // New marker if specified.
    marker.replaceLinalgMarker(rewriter, op);
  });
  return success();
}

mlir::linalg::LinalgBasePromotionPattern::LinalgBasePromotionPattern(
    StringRef opName, MLIRContext *context,
    ArrayRef<unsigned> operandsToPromote, unsigned alignment,
    LinalgMarker marker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      operandsToPromote(operandsToPromote.begin(), operandsToPromote.end()),
      alignment(alignment) {}

LogicalResult mlir::linalg::LinalgBasePromotionPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();
  if (operandsToPromote.empty()) {
    if (failed(promoteSubviewsLinalgOpPrecondition(op, llvm::None)))
      return failure();
  } else {
    DenseSet<unsigned> set;
    set.insert(operandsToPromote.begin(), operandsToPromote.end());
    if (failed(promoteSubviewsLinalgOpPrecondition(op, set)))
      return failure();
  }

  llvm::SetVector<Value> subViews;
  if (!operandsToPromote.empty()) {
    for (unsigned idx : operandsToPromote) {
      auto *op = linalgOp.getBuffer(idx).getDefiningOp();
      if (auto sv = dyn_cast_or_null<SubViewOp>(op))
        subViews.insert(sv);
    }
  } else {
    unsigned nBuffers = linalgOp.getNumInputsAndOutputBuffers();
    for (unsigned idx = 0; idx < nBuffers; ++idx) {
      auto *op = linalgOp.getBuffer(idx).getDefiningOp();
      if (auto sv = dyn_cast_or_null<SubViewOp>(op))
        subViews.insert(sv);
    }
  }

  auto promotedOp =
      promoteSubViewOperands(rewriter, op, subViews, /*dynamicBuffers=*/false,
                             /*alignment=*/alignment);
  marker.replaceLinalgMarker(rewriter, promotedOp.getOperation());
  rewriter.eraseOp(op);
  return success();
}

mlir::linalg::LinalgBaseVectorizationPattern::LinalgBaseVectorizationPattern(
    StringRef opName, MLIRContext *context, LinalgMarker marker,
    PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker) {}

LogicalResult mlir::linalg::LinalgBaseVectorizationPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();
  if (failed(vectorizeLinalgOpPrecondition(op)))
    return failure();
  vectorizeLinalgOp(rewriter, op);
  rewriter.eraseOp(op);
  return success();
}
