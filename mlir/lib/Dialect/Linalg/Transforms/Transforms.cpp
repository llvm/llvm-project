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
#include "mlir/Dialect/Affine/Utils.h"
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
#include "llvm/ADT/ScopeExit.h"
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

mlir::linalg::LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<Identifier> matchDisjunction, Optional<Identifier> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement) {}

mlir::linalg::LinalgTransformationFilter::LinalgTransformationFilter(
    FilterFunction f, ArrayRef<Identifier> matchDisjunction,
    Optional<Identifier> replacement)
    : filters(),
      matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement) {
  if (f)
    filters.push_back(f);
}

LogicalResult mlir::linalg::LinalgTransformationFilter::checkAndNotify(
    PatternRewriter &rewriter, Operation *op) const {
  if (llvm::any_of(filters,
                   [&](const FilterFunction &f) { return failed(f(op)); }))
    return failure();

  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty())
      return success();

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction)
    if (attr.getValue() == filter)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void mlir::linalg::LinalgTransformationFilter::
    replaceLinalgTransformationFilter(PatternRewriter &rewriter,
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

/// Try to compute a static bounding box for `operand`
/// Return success if either:
///   1. The operand is already statically shaped, `result` is left unchanged.
///   2. The operand is (partially) dynamic, `result` is the result of a freshly
///      created PadTensorOp.
/// Return failure if the operand cannot be padded to a static shape.
static LogicalResult padOperandToSmallestStaticBoundingBox(
    PatternRewriter &rewriter, linalg::LinalgOp opToPad, OpOperand &operand,
    const LinalgTilingOptions &options, Value &result) {
  auto tensorType = operand.get().getType().cast<RankedTensorType>();
  // Already static shape, no need to pad.
  if (tensorType.hasStaticShape())
    return success();
  auto subtensor = operand.get().getDefiningOp<SubTensorOp>();
  // Not a subtensor, cannot construct a static bounding box.
  if (!subtensor)
    return failure();
  SmallVector<int64_t> staticSizes;
  staticSizes.reserve(tensorType.getRank());
  auto shapedOp =
      cast<OffsetSizeAndStrideOpInterface>(subtensor.getOperation());
  for (auto size : shapedOp.getMixedSizes()) {
    auto indexAttr = size.is<Attribute>()
                         ? size.get<Attribute>().dyn_cast<IntegerAttr>()
                         : linalg::getSmallestBoundingIndex(size.get<Value>());
    // SmallestBoundingIndex must exist for all sizes.
    // For now return an error if we can't find it.
    if (!indexAttr)
      return rewriter.notifyMatchFailure(
          opToPad, "No constant bounding box can be found for padding");
    staticSizes.push_back(indexAttr.getInt());
  }
  Value pad = options.paddingValueComputationFunction(rewriter, operand);
  auto staticTensorType =
      RankedTensorType::get(staticSizes, tensorType.getElementType());
  result = linalg::PadTensorOp::createPadHighOp(
      staticTensorType, operand.get(), pad, opToPad->getLoc(), rewriter);
  return success();
}

// Try to create a static bounding box around each operand of `res.op`.
// If successful, `res.op` is rewritten in static form with padded operands.
// `res.op` is updated to the cloned static form of the op on success.
static LogicalResult rewriteAsPaddedOp(PatternRewriter &rewriter,
                                       TiledLinalgOp &res,
                                       const LinalgTilingOptions &options) {
  LinalgOp opToPad = res.op;
  Location loc = opToPad->getLoc();

  // If the op is fully static, it does not need padding.
  // TODO: there are cases where we may still want to pad to larger sizes.
  if (llvm::all_of(opToPad.getShapedOperands(), [](Value v) {
        return v.getType().cast<RankedTensorType>().hasStaticShape();
      }))
    return success();

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(opToPad);
  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad.getNumShapedOperands());
  for (OpOperand &operand : opToPad.getShapedOpOperands()) {
    Value paddedOperand;
    // If padding was requested but the shape cannot be bounded statically then
    // the pattern fails to apply.
    if (failed(padOperandToSmallestStaticBoundingBox(rewriter, opToPad, operand,
                                                     options, paddedOperand))) {
      return failure();
    }
    newOperands.push_back(paddedOperand ? paddedOperand : operand.get());
  }

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumOutputs()).getTypes();
  ValueRange otherOperands = opToPad.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  linalg::LinalgOp paddedOp =
      opToPad.clone(rewriter, loc, resultTensorTypes, newOperands);

  // Recover the subtensor out of the new static results. This keeps the
  // original linalg op around because it uses the dims of the original results.
  // This later folds away.
  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(opToPad->getNumResults());
  llvm::SetVector<Operation *> newUsersOfOpToPad;
  for (auto it : llvm::zip(opToPad->getResults(), paddedOp->getResults())) {
    auto rank = std::get<0>(it).getType().cast<RankedTensorType>().getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    auto sizes = llvm::to_vector<4>(llvm::map_range(
        llvm::seq<unsigned>(0, rank), [&](unsigned d) -> OpFoldResult {
          auto dimOp = rewriter.create<memref::DimOp>(loc, std::get<0>(it), d);
          newUsersOfOpToPad.insert(dimOp);
          return dimOp.getResult();
        }));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubviewResults.push_back(rewriter.create<SubTensorOp>(
        loc, std::get<1>(it), offsets, sizes, strides));
  }
  // Replace the transient `opToPad` locally, except for uses that we just
  // created for the purpose of extracting the dims.
  rewriter.replaceOpWithIf(opToPad, paddedSubviewResults, [&](OpOperand &opOp) {
    return !newUsersOfOpToPad.contains(opOp.getOwner());
  });

  res = TiledLinalgOp{paddedOp, res.loops, res.tensorResults};
  return success();
}

/// Linalg base tiling pattern.
mlir::linalg::LinalgBaseTilingPattern::LinalgBaseTilingPattern(
    StringRef opName, MLIRContext *context, LinalgTilingOptions options,
    LinalgTransformationFilter filter, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), filter(filter),
      options(options) {}

mlir::linalg::LinalgBaseTilingPattern::LinalgBaseTilingPattern(
    LinalgTilingOptions options, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(benefit, MatchAnyOpTypeTag()), filter(filter),
      options(options) {}

LogicalResult mlir::linalg::LinalgBaseTilingPattern::matchAndRewriteBase(
    Operation *op, PatternRewriter &rewriter, TiledLinalgOp &result) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();

  Optional<TiledLinalgOp> res = tileLinalgOp(rewriter, linalgOp, options);

  if (!res)
    return failure();

  // Setup RAII guard to return properly.
  bool succeeded = true;
  LinalgOp tiledOp = res->op;
  auto guard = llvm::make_scope_exit([&]() {
    if (!succeeded)
      return;
    // Return relevant information to derived pattern.
    result = *res;
    // Replace filter on both tiledOp and tiledAndPaddedOp, if necessary.
    filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    if (tiledOp != res->op)
      filter.replaceLinalgTransformationFilter(rewriter, res->op);
  });

  // Consider padding on the fly only if the op has tensor semantics.
  if (!options.paddingValueComputationFunction ||
      !linalgOp.hasTensorSemantics())
    return success();

  // Try to pad on the fly by rewriting res->op as a padded op.
  if (failed(rewriteAsPaddedOp(rewriter, *res, options))) {
    // Set so RAII guard does not propagate TiledLinalgOp to `result`.
    succeeded = false;
    return failure();
  }

  // Do not perform replacement of `linalgOp`, let the derived patterns
  // do this as they see fit, from the resulting TiledLinalgOp.
  return success();
}

static ValueRange getTiledOpResult(TiledLinalgOp tiledOp) {
  if (tiledOp.loops.empty())
    return tiledOp.op.getOperation()->getResults();
  return tiledOp.loops.front()->getResults();
}

static ValueRange
getTiledAndFusedOpResult(TiledAndFusedLinalgOps tiledAndFusedOp) {
  if (tiledAndFusedOp.fusedLoops.empty())
    return tiledAndFusedOp.op.getOperation()->getResults();
  return tiledAndFusedOp.fusedLoops.front()->getResults();
}

mlir::linalg::LinalgBaseTileAndFusePattern::LinalgBaseTileAndFusePattern(
    StringRef opName, MLIRContext *context,
    const LinalgDependenceGraph &dependenceGraph,
    LinalgTilingOptions tilingOptions, LinalgFusionOptions fusionOptions,
    LinalgTransformationFilter filter, LinalgTransformationFilter fusedOpMarker,
    LinalgTransformationFilter originalOpMarker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context),
      dependenceGraph(dependenceGraph), tilingOptions(tilingOptions),
      fusionOptions(fusionOptions), filter(filter),
      fusedOpMarker(fusedOpMarker), originalOpMarker(originalOpMarker) {}

LogicalResult mlir::linalg::LinalgBaseTileAndFusePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();

  DenseSet<Operation *> producers;
  producers.insert(linalgOp);
  for (auto dependence : dependenceGraph.getDependentOperationsInto(linalgOp)) {
    Optional<unsigned> operandNumber = dependence.getIndexingOpViewOperandNum();
    // When looking at dependences into, indexingOp is always OpOperand. We
    // could assert, but continue if this is not the case.
    if (!operandNumber)
      continue;
    if (!fusionOptions.indicesToFuse.count(operandNumber.getValue()))
      continue;
    if (isa<LinalgOp>(dependence.getDependentOp()))
      producers.insert(dependence.getDependentOp());
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
    rewriter.replaceOp(tiledAndFusedOps->op,
                       getTiledOpResult(unfusedTiledOp.getValue()));
    tiledAndFusedOps->op = unfusedTiledOp->op;
  }
  op->replaceAllUsesWith(getTiledAndFusedOpResult(tiledAndFusedOps.getValue()));

  filter.replaceLinalgTransformationFilter(rewriter,
                                           tiledAndFusedOps->op.getOperation());
  for (auto fusedOp : tiledAndFusedOps->fusedProducers) {
    fusedOpMarker.replaceLinalgTransformationFilter(rewriter,
                                                    fusedOp.getOperation());
  }
  for (auto origProducerOp : ArrayRef<LinalgOp>(fusionOps).drop_back()) {
    originalOpMarker.replaceLinalgTransformationFilter(
        rewriter, origProducerOp.getOperation());
  }
  rewriter.updateRootInPlace(op, [&]() {
    originalOpMarker.replaceLinalgTransformationFilter(rewriter, op);
  });
  return success();
}

/// Linalg base interchange pattern.
mlir::linalg::LinalgBaseInterchangePattern::LinalgBaseInterchangePattern(
    StringRef opName, MLIRContext *context,
    ArrayRef<unsigned> interchangeVector, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), filter(filter),
      interchangeVector(interchangeVector.begin(), interchangeVector.end()) {}

LogicalResult mlir::linalg::LinalgBaseInterchangePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();
  if (failed(interchangeGenericLinalgOpPrecondition(op, interchangeVector)))
    return failure();

  // TODO: figure out how this interplays with named ops. In particular this
  // should break the named op property.
  rewriter.updateRootInPlace(op, [&]() {
    interchange(linalgOp, interchangeVector);
    // New filter if specified.
    filter.replaceLinalgTransformationFilter(rewriter, op);
  });
  return success();
}

mlir::linalg::LinalgBasePromotionPattern::LinalgBasePromotionPattern(
    StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
    LinalgTransformationFilter filter, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), filter(filter),
      options(options) {}

LogicalResult mlir::linalg::LinalgBasePromotionPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, op)))
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
  filter.replaceLinalgTransformationFilter(rewriter, op);
  return success();
}

mlir::linalg::LinalgBaseVectorizationPattern::LinalgBaseVectorizationPattern(
    LinalgTransformationFilter filter, PatternBenefit benefit)
    : RewritePattern(benefit, MatchAnyOpTypeTag()), filter(filter) {}

mlir::linalg::LinalgBaseVectorizationPattern::LinalgBaseVectorizationPattern(
    StringRef opName, MLIRContext *context, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), filter(filter) {}

LogicalResult mlir::linalg::LinalgBaseVectorizationPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();
  SmallVector<Value> newResults;
  if (failed(vectorizeLinalgOp(rewriter, op, newResults)))
    return failure();
  if (!newResults.empty())
    rewriter.replaceOp(op, newResults);
  else
    rewriter.eraseOp(op);
  return success();
}

LogicalResult mlir::linalg::applyStagedPatterns(
    Operation *op, ArrayRef<FrozenRewritePatternSet> stage1Patterns,
    const FrozenRewritePatternSet &stage2Patterns,
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
static AffineMap substitute(
    AffineMap map, SmallVectorImpl<Value> &dims,
    SmallVectorImpl<Value> &symbols,
    llvm::function_ref<bool(Operation *)> substituteOperation = nullptr) {
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
          if (!substituteOperation || substituteOperation(forOp))
            substitutedExpr = substituteLoopInExpr(
                expr, dimExpr, forOp.lowerBound(), forOp.upperBound(),
                forOp.step(), dims, symbols);

        if (auto parallelForOp = scf::getParallelForInductionVarOwner(dim))
          if (!substituteOperation || substituteOperation(parallelForOp))
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

    LLVM_DEBUG({
      DBGS() << "Map to simplify: " << map << "\n";
      DBGS() << "Operands:\n";
      for (Value v : operands)
        DBGS() << v << "\n";
    });

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

/// Traverse the dims of the AffineMap of `affineMinOp` and substitute scf loop
/// induction variables by new expressions involving the lower or upper bound:
///   - If the AffineDimExpr mapped to a loop IV has a positive sign, it is
///     replaced by the loop upper bound.
///   - If the AffineDimExpr mapped to a loop IV has a negative sign, it is
///     replaced by the loop lower bound.
/// All loop induction variables are iteratively replaced, unless a
/// `substituteOperation` hook is passed to more finely determine which
/// operations are substituted.
/// This is used as an intermediate step in computing bounding boxes and
/// canonicalize AffineMinOps. All dim and symbol operands are assumed to have
/// positive values (positive orthant assumptions).
/// Return a new AffineMap, dims and symbols that have been canonicalized and
/// simplified.
AffineMapAndOperands mlir::linalg::substituteMin(
    AffineMinOp affineMinOp,
    llvm::function_ref<bool(Operation *)> substituteOperation) {
  AffineMapAndOperands res{affineMinOp.getAffineMap(),
                           SmallVector<Value>(affineMinOp.getDimOperands()),
                           SmallVector<Value>(affineMinOp.getSymbolOperands())};
  res.map = substitute(affineMinOp.getAffineMap(), res.dims, res.symbols,
                       substituteOperation);
  return res;
}

LogicalResult AffineMinSCFCanonicalizationPattern::matchAndRewrite(
    AffineMinOp minOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(DBGS() << "Canonicalize AffineMinSCF: " << *minOp.getOperation()
                    << "\n");

  auto affineMapAndOperands = substituteMin(minOp);
  AffineMap map = affineMapAndOperands.map;

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
      SmallVector<Value> resultOperands = affineMapAndOperands.dims;
      llvm::append_range(resultOperands, affineMapAndOperands.symbols);
      canonicalizeMapAndOperands(&resultMap, &resultOperands);
      resultMap = simplifyAffineMap(resultMap);
      rewriter.replaceOpWithNewOp<AffineApplyOp>(minOp, resultMap,
                                                 resultOperands);
    }
    return success();
  }

  return failure();
}
