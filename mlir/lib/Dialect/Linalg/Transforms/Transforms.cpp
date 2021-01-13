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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

#define DEBUG_TYPE "linalg-transforms"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//
// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral mlir::linalg::LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

mlir::linalg::LinalgMarker::LinalgMarker(ArrayRef<Identifier> matchDisjunction,
                                         Optional<Identifier> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement) {}

LogicalResult
mlir::linalg::LinalgMarker::checkAndNotify(PatternRewriter &rewriter,
                                           Operation *op) const {
  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no marker case and matchDisjunction is empty.
    if (matchDisjunction.empty())
      return success();

    // 2. Has no marker but was expecting a marker.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any marker from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit marker.
  for (auto marker : matchDisjunction)
    if (attr.getValue() == marker)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any marker from list: ";
    interleaveComma(matchDisjunction, diag);
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

LinalgTilingOptions &
mlir::linalg::LinalgTilingOptions::setTileSizes(ArrayRef<int64_t> ts) {
  SmallVector<int64_t, 4> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentOfType<FuncOp>().getBody().front());
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      Value v = b.create<ConstantIndexOp>(op->getLoc(), s);
      return v;
    }));
  };
  return *this;
}

/// Linalg base tiling pattern.
mlir::linalg::LinalgBaseTilingPattern::LinalgBaseTilingPattern(
    StringRef opName, MLIRContext *context, LinalgTilingOptions options,
    LinalgMarker marker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      options(options) {}

mlir::linalg::LinalgBaseTilingPattern::LinalgBaseTilingPattern(
    LinalgTilingOptions options, LinalgMarker marker, PatternBenefit benefit)
    : RewritePattern(benefit, MatchAnyOpTypeTag()), marker(marker),
      options(options) {}

LogicalResult mlir::linalg::LinalgBaseTilingPattern::matchAndRewriteBase(
    Operation *op, PatternRewriter &rewriter, TiledLinalgOp &result) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();

  Optional<TiledLinalgOp> res = tileLinalgOp(rewriter, linalgOp, options);

  if (!res)
    return failure();

  // Return relevant information to derived pattern.
  result = *res;

  // New marker if specified.
  marker.replaceLinalgMarker(rewriter, res->op.getOperation());
  return success();
}

mlir::linalg::LinalgBaseTileAndFusePattern::LinalgBaseTileAndFusePattern(
    StringRef opName, MLIRContext *context,
    const LinalgDependenceGraph &dependenceGraph,
    LinalgTilingOptions tilingOptions, LinalgFusionOptions fusionOptions,
    LinalgMarker marker, LinalgMarker fusedOpMarker,
    LinalgMarker originalOpMarker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context),
      dependenceGraph(dependenceGraph), tilingOptions(tilingOptions),
      fusionOptions(fusionOptions), marker(marker),
      fusedOpMarker(fusedOpMarker), originalOpMarker(originalOpMarker) {}

LogicalResult mlir::linalg::LinalgBaseTileAndFusePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();
  if (!linalgOp.hasBufferSemantics())
    return failure();

  DenseSet<Operation *> producers;
  producers.insert(linalgOp);
  for (auto dependence : dependenceGraph.getDependentOperations(linalgOp)) {
    if (!fusionOptions.indicesToFuse.count(
            dependence.indexingOpView->getOperandNumber()))
      continue;
    if (isa<LinalgOp>(dependence.dependentOpView->getOwner()))
      producers.insert(dependence.dependentOpView->getOwner());
  }

  SmallVector<LinalgOp, 1> fusionOps;
  for (auto it = op->getBlock()->begin(), ie = Block::iterator(op); it != ie;
       ++it) {
    auto producerLinalgOp = dyn_cast<LinalgOp>(&(*it));
    if (producerLinalgOp && producers.count(producerLinalgOp))
      fusionOps.push_back(producerLinalgOp);
  }
  fusionOps.push_back(linalgOp);

  SmallVector<Value, 4> tileSizes =
      tilingOptions.tileSizeComputationFunction(rewriter, op);
  LinalgTilingOptions instanceTilingOptions = tilingOptions;
  instanceTilingOptions.setTileSizes(tileSizes);
  Optional<TiledAndFusedLinalgOps> tiledAndFusedOps = tileAndFuseLinalgOps(
      rewriter, fusionOps, dependenceGraph, instanceTilingOptions);
  if (!tiledAndFusedOps)
    return failure();

  // Tile the unfused loops;
  SmallVector<Value, 4> unfusedLoopTileSizes;
  Value zero = rewriter.create<ConstantIndexOp>(op->getLoc(), 0);
  for (auto tileSize : enumerate(tileSizes)) {
    if (tiledAndFusedOps->fusedLoopDims.count(tileSize.index()))
      unfusedLoopTileSizes.push_back(zero);
    else
      unfusedLoopTileSizes.push_back(tileSize.value());
  }
  // Tile the loop only if there is a non-zero tile size.
  if (unfusedLoopTileSizes.size() > linalgOp.getNumLoops())
    unfusedLoopTileSizes.resize(linalgOp.getNumLoops());
  if (llvm::any_of(unfusedLoopTileSizes, [](Value val) {
        if (auto cst = val.getDefiningOp<ConstantIndexOp>())
          return cst.getValue() != 0;
        return true;
      })) {
    LinalgTilingOptions unfusedTilingOptions = tilingOptions;
    unfusedTilingOptions.setTileSizes(unfusedLoopTileSizes);
    Optional<TiledLinalgOp> unfusedTiledOp =
        tileLinalgOp(rewriter, tiledAndFusedOps->op, unfusedTilingOptions);
    if (!unfusedTiledOp)
      return failure();
    rewriter.eraseOp(tiledAndFusedOps->op);
    tiledAndFusedOps->op = unfusedTiledOp->op;
  }

  marker.replaceLinalgMarker(rewriter, tiledAndFusedOps->op.getOperation());
  for (auto fusedOp : tiledAndFusedOps->fusedProducers) {
    fusedOpMarker.replaceLinalgMarker(rewriter, fusedOp.getOperation());
  }
  for (auto origProducerOp : ArrayRef<LinalgOp>(fusionOps).drop_back()) {
    originalOpMarker.replaceLinalgMarker(rewriter,
                                         origProducerOp.getOperation());
  }
  rewriter.updateRootInPlace(
      op, [&]() { originalOpMarker.replaceLinalgMarker(rewriter, op); });
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
    StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
    LinalgMarker marker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      options(options) {}

LogicalResult mlir::linalg::LinalgBasePromotionPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  if (failed(marker.checkAndNotify(rewriter, op)))
    return failure();
  if (failed(promoteSubviewsPrecondition(op, options)))
    return failure();

  // TODO: We cannot use root update here. This pattern is creating other ops,
  // so if the promotion fails, those need to be cleaned up, which doesnt seem
  // to be happening here. So to fail properly, we should be cloning the op and
  // deleting the previous op. This needs more investigation.
  rewriter.startRootUpdate(op);
  Optional<LinalgOp> promotedOp = promoteSubViews(rewriter, op, options);
  if (!promotedOp) {
    rewriter.cancelRootUpdate(op);
    return op->emitError("subview promotion failed");
  }
  rewriter.finalizeRootUpdate(op);
  marker.replaceLinalgMarker(rewriter, op);
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

LogicalResult mlir::linalg::applyStagedPatterns(
    Operation *op, ArrayRef<FrozenRewritePatternList> stage1Patterns,
    const FrozenRewritePatternList &stage2Patterns,
    function_ref<LogicalResult(Operation *)> stage3Lambda) {
  unsigned iteration = 0;
  (void)iteration;
  for (const auto &patterns : stage1Patterns) {
    LLVM_DEBUG(DBGS() << "Before 1st stage, iter: " << ++iteration << "\n"
                      << *op);
    if (failed(applyPatternsAndFoldGreedily(op, patterns))) {
      LLVM_DEBUG(DBGS() << "Underlying first stage rewrite did not converge");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "After 1st stage, iter: " << ++iteration << "\n"
                      << *op);
    if (failed(applyPatternsAndFoldGreedily(op, stage2Patterns))) {
      LLVM_DEBUG(DBGS() << "Underlying 2nd stage rewrite did not converge");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "After 2nd stage, iter : " << iteration << "\n"
                      << *op);
    if (stage3Lambda) {
      if (failed(stage3Lambda(op)))
        return failure();
      LLVM_DEBUG(DBGS() << "After 3rd stage, iter : " << iteration << "\n"
                        << *op);
    }
  }
  return success();
}

/// Traverse `e` and return an AffineExpr where all occurrences of `dim` have
/// been replaced by either:
///  - `min` if `positivePath` is true when we reach an occurrence of `dim`
///  - `max` if `positivePath` is true when we reach an occurrence of `dim`
/// `positivePath` is negated each time we hit a multiplicative or divisive
/// binary op with a constant negative coefficient.
static AffineExpr substWithMin(AffineExpr e, AffineExpr dim, AffineExpr min,
                               AffineExpr max, bool positivePath = true) {
  if (e == dim)
    return positivePath ? min : max;
  if (auto bin = e.dyn_cast<AffineBinaryOpExpr>()) {
    AffineExpr lhs = bin.getLHS();
    AffineExpr rhs = bin.getRHS();
    if (bin.getKind() == mlir::AffineExprKind::Add)
      return substWithMin(lhs, dim, min, max, positivePath) +
             substWithMin(rhs, dim, min, max, positivePath);

    auto c1 = bin.getLHS().dyn_cast<AffineConstantExpr>();
    auto c2 = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (c1 && c1.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), c1, substWithMin(rhs, dim, min, max, !positivePath));
    if (c2 && c2.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), substWithMin(lhs, dim, min, max, !positivePath), c2);
    return getAffineBinaryOpExpr(
        bin.getKind(), substWithMin(lhs, dim, min, max, positivePath),
        substWithMin(rhs, dim, min, max, positivePath));
  }
  return e;
}

/// Given the `lbVal`, `ubVal` and `stepVal` of a loop, append `lbVal` and
/// `ubVal` to `dims` and `stepVal` to `symbols`.
/// Create new AffineDimExpr (`%lb` and `%ub`) and AffineSymbolExpr (`%step`)
/// with positions matching the newly appended values. Substitute occurrences of
/// `dimExpr` by either the min expression (i.e. `%lb`) or the max expression
/// (i.e. `%lb + %step * floordiv(%ub -1 - %lb, %step)`), depending on whether
/// the induction variable is used with a positive or negative  coefficient.
static AffineExpr substituteLoopInExpr(AffineExpr expr, AffineExpr dimExpr,
                                       Value lbVal, Value ubVal, Value stepVal,
                                       SmallVectorImpl<Value> &dims,
                                       SmallVectorImpl<Value> &symbols) {
  MLIRContext *ctx = lbVal.getContext();
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(lbVal);
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(ubVal);
  AffineExpr step = getAffineSymbolExpr(symbols.size(), ctx);
  symbols.push_back(stepVal);
  LLVM_DEBUG(DBGS() << "Before: " << expr << "\n");
  AffineExpr ee = substWithMin(expr, dimExpr, lb,
                               lb + step * ((ub - 1) - lb).floorDiv(step));
  LLVM_DEBUG(DBGS() << "After: " << expr << "\n");
  return ee;
}

/// Traverse the `dims` and substitute known min or max expressions in place of
/// induction variables in `exprs`.
static AffineMap substitute(AffineMap map, SmallVectorImpl<Value> &dims,
                            SmallVectorImpl<Value> &symbols) {
  auto exprs = llvm::to_vector<4>(map.getResults());
  for (AffineExpr &expr : exprs) {
    bool substituted = true;
    while (substituted) {
      substituted = false;
      for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
        Value dim = dims[dimIdx];
        AffineExpr dimExpr = getAffineDimExpr(dimIdx, expr.getContext());
        LLVM_DEBUG(DBGS() << "Subst: " << dim << " @ " << dimExpr << "\n");
        AffineExpr substitutedExpr;
        if (auto forOp = scf::getForInductionVarOwner(dim))
          substitutedExpr = substituteLoopInExpr(
              expr, dimExpr, forOp.lowerBound(), forOp.upperBound(),
              forOp.step(), dims, symbols);

        if (auto parallelForOp = scf::getParallelForInductionVarOwner(dim))
          for (unsigned idx = 0, e = parallelForOp.getNumLoops(); idx < e;
               ++idx)
            substitutedExpr = substituteLoopInExpr(
                expr, dimExpr, parallelForOp.lowerBound()[idx],
                parallelForOp.upperBound()[idx], parallelForOp.step()[idx],
                dims, symbols);

        if (!substitutedExpr)
          continue;

        substituted = (substitutedExpr != expr);
        expr = substitutedExpr;
      }
    }

    // Cleanup and simplify the results.
    // This needs to happen outside of the loop iterating on dims.size() since
    // it modifies dims.
    SmallVector<Value, 4> operands(dims.begin(), dims.end());
    operands.append(symbols.begin(), symbols.end());
    auto map = AffineMap::get(dims.size(), symbols.size(), exprs,
                              exprs.front().getContext());

    LLVM_DEBUG(DBGS() << "Map to simplify: " << map << "\n");

    // Pull in affine.apply operations and compose them fully into the
    // result.
    fullyComposeAffineMapAndOperands(&map, &operands);
    canonicalizeMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    // Assign the results.
    exprs.assign(map.getResults().begin(), map.getResults().end());
    dims.assign(operands.begin(), operands.begin() + map.getNumDims());
    symbols.assign(operands.begin() + map.getNumDims(), operands.end());

    LLVM_DEBUG(DBGS() << "Map simplified: " << map << "\n");
  }

  assert(!exprs.empty() && "Unexpected empty exprs");
  return AffineMap::get(dims.size(), symbols.size(), exprs, map.getContext());
}

LogicalResult AffineMinSCFCanonicalizationPattern::matchAndRewrite(
    AffineMinOp minOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(DBGS() << "Canonicalize AffineMinSCF: " << *minOp.getOperation()
                    << "\n");

  SmallVector<Value, 4> dims(minOp.getDimOperands()),
      symbols(minOp.getSymbolOperands());
  AffineMap map = substitute(minOp.getAffineMap(), dims, symbols);

  LLVM_DEBUG(DBGS() << "Resulting map: " << map << "\n");

  // Check whether any of the expressions, when subtracted from all other
  // expressions, produces only >= 0 constants. If so, it is the min.
  for (auto e : minOp.getAffineMap().getResults()) {
    LLVM_DEBUG(DBGS() << "Candidate min: " << e << "\n");
    if (!e.isSymbolicOrConstant())
      continue;

    auto isNonPositive = [](AffineExpr e) {
      if (auto cst = e.dyn_cast<AffineConstantExpr>())
        return cst.getValue() < 0;
      return true;
    };

    // Build the subMap and check everything is statically known to be
    // positive.
    SmallVector<AffineExpr, 4> subExprs;
    subExprs.reserve(map.getNumResults());
    for (auto ee : map.getResults())
      subExprs.push_back(ee - e);
    MLIRContext *ctx = minOp.getContext();
    AffineMap subMap = simplifyAffineMap(
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), subExprs, ctx));
    LLVM_DEBUG(DBGS() << "simplified subMap: " << subMap << "\n");
    if (llvm::any_of(subMap.getResults(), isNonPositive))
      continue;

    // Static min found.
    if (auto cst = e.dyn_cast<AffineConstantExpr>()) {
      rewriter.replaceOpWithNewOp<ConstantIndexOp>(minOp, cst.getValue());
    } else {
      auto resultMap = AffineMap::get(0, map.getNumSymbols(), {e}, ctx);
      SmallVector<Value, 4> resultOperands = dims;
      resultOperands.append(symbols.begin(), symbols.end());
      canonicalizeMapAndOperands(&resultMap, &resultOperands);
      resultMap = simplifyAffineMap(resultMap);
      rewriter.replaceOpWithNewOp<AffineApplyOp>(minOp, resultMap,
                                                 resultOperands);
    }
    return success();
  }

  return failure();
}
