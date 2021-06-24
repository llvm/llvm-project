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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
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
                rewriter.getStringAttr(replacement.getValue().strref()));
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
    PatternRewriter &rewriter, linalg::LinalgOp opToPad, OpOperand *opOperand,
    const LinalgTilingOptions &options, Value &result) {
  // Already static shape, no need to pad.
  if (llvm::none_of(opToPad.getShape(opOperand), ShapedType::isDynamic))
    return success();
  auto sliceOp = opOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
  // Not a slice op, cannot construct a static bounding box.
  if (!sliceOp)
    return failure();
  SmallVector<int64_t> staticSizes;
  staticSizes.reserve(opToPad.getRank(opOperand));
  auto shapedOp = cast<OffsetSizeAndStrideOpInterface>(sliceOp.getOperation());
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
  Value pad = options.paddingValueComputationFunction(rewriter, *opOperand);
  auto staticTensorType = RankedTensorType::get(
      staticSizes, getElementTypeOrSelf(opOperand->get()));
  result = linalg::PadTensorOp::createPadHighOp(
      staticTensorType, opOperand->get(), pad, opToPad->getLoc(), rewriter);
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
  assert(opToPad.hasTensorSemantics() &&
         "expected operation to have tensor semantics");
  if (!opToPad.hasDynamicShape())
    return success();

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(opToPad);
  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad.getNumInputsAndOutputs());
  for (OpOperand *opOperand : opToPad.getInputAndOutputOperands()) {
    Value paddedOperand;
    // If padding was requested but the shape cannot be bounded statically then
    // the pattern fails to apply.
    if (failed(padOperandToSmallestStaticBoundingBox(
            rewriter, opToPad, opOperand, options, paddedOperand)))
      return failure();
    newOperands.push_back(paddedOperand ? paddedOperand : opOperand->get());
  }

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumOutputs()).getTypes();
  ValueRange otherOperands = opToPad.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  linalg::LinalgOp paddedOp =
      opToPad.clone(rewriter, loc, resultTensorTypes, newOperands);

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  // This later folds away.
  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(opToPad->getNumResults());
  SetVector<Operation *> newUsersOfOpToPad;
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
    paddedSubviewResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
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
    : RewritePattern(opName, benefit, context), filter(filter),
      options(options) {}

mlir::linalg::LinalgBaseTilingPattern::LinalgBaseTilingPattern(
    MLIRContext *context, LinalgTilingOptions options,
    LinalgTransformationFilter filter, PatternBenefit benefit)
    : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter),
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
  LinalgOp tiledOp = res->op;
  auto guard = llvm::make_scope_exit([&]() {
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
    : RewritePattern(opName, benefit, context, {}),
      dependenceGraph(dependenceGraph), tilingOptions(tilingOptions),
      fusionOptions(fusionOptions), filter(filter),
      fusedOpMarker(fusedOpMarker), originalOpMarker(originalOpMarker) {}

LogicalResult mlir::linalg::LinalgBaseTileAndFusePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  // TODO: remove hasIndexSemantics check once index ops are supported.
  if (!linalgOp || linalgOp.hasIndexSemantics())
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

/// Linalg generic interchange pattern.
mlir::linalg::GenericOpInterchangePattern::GenericOpInterchangePattern(
    MLIRContext *context, ArrayRef<unsigned> interchangeVector,
    LinalgTransformationFilter filter, PatternBenefit benefit)
    : OpRewritePattern(context, benefit), filter(filter),
      interchangeVector(interchangeVector.begin(), interchangeVector.end()) {}

LogicalResult mlir::linalg::GenericOpInterchangePattern::matchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, genericOp)))
    return failure();
  if (failed(interchangeGenericOpPrecondition(genericOp, interchangeVector)))
    return failure();

  // TODO: figure out how this interplays with named ops. In particular this
  // should break the named op property.
  rewriter.updateRootInPlace(genericOp, [&]() {
    interchangeGenericOp(rewriter, genericOp, interchangeVector);
    // New filter if specified.
    filter.replaceLinalgTransformationFilter(rewriter, genericOp);
  });
  return success();
}

mlir::linalg::LinalgBasePromotionPattern::LinalgBasePromotionPattern(
    StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
    LinalgTransformationFilter filter, PatternBenefit benefit)
    : RewritePattern(opName, benefit, context, {}), filter(filter),
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
    MLIRContext *context, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter) {}

mlir::linalg::LinalgBaseVectorizationPattern::LinalgBaseVectorizationPattern(
    StringRef opName, MLIRContext *context, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(opName, benefit, context, {}), filter(filter) {}

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

/// Traverse the `dims` and substitute known min or max expressions returned by
/// the lambda |getMinMaxExpr|.
static AffineMap substitute(AffineMap map, SmallVectorImpl<Value> &dims,
                            SmallVectorImpl<Value> &symbols,
                            GetMinMaxExprFn getMinMaxExpr) {
  auto exprs = llvm::to_vector<4>(map.getResults());
  for (AffineExpr &expr : exprs) {
    bool substituted = true;
    while (substituted) {
      substituted = false;
      for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
        Value dim = dims[dimIdx];
        auto minMax = getMinMaxExpr(dim, dims, symbols);
        if (!minMax)
          continue;
        AffineExpr dimExpr = getAffineDimExpr(dimIdx, expr.getContext());
        LLVM_DEBUG(DBGS() << "Subst: " << dim << " @ " << dimExpr << "\n");
        LLVM_DEBUG(DBGS() << "Before: " << expr << "\n");
        // Substitute occurrences of `dimExpr` by either the min expression or
        // the max expression depending on whether the value is used with a
        // positive or negative  coefficient.
        AffineExpr substitutedExpr =
            substWithMin(expr, dimExpr, minMax->first, minMax->second);
        LLVM_DEBUG(DBGS() << "After: " << substitutedExpr << "\n");
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

/// Traverse the dims of the AffineMap of `affineMinOp` and substitute
/// dimensions with known range by new expressions involving the min or max
/// expression:
///   - If the AffineDimExpr mapped to a known value has a positive sign, it
///     is replaced by the min expression.
///   - If the AffineDimExpr mapped to a known value has a negative sign, it is
///     replaced by the max expression.
/// All known values are iteratively replaced.
/// This is used as an intermediate step in computing bounding boxes and
/// canonicalize AffineMinOps. All dim and symbol operands are assumed to have
/// positive values (positive orthant assumptions).
/// Return a new AffineMap, dims and symbols that have been canonicalized and
/// simplified.
AffineMapAndOperands
mlir::linalg::substituteMin(AffineMinOp affineMinOp,
                            GetMinMaxExprFn getMinMaxExpr) {
  AffineMapAndOperands res{affineMinOp.getAffineMap(),
                           SmallVector<Value>(affineMinOp.getDimOperands()),
                           SmallVector<Value>(affineMinOp.getSymbolOperands())};
  res.map = substitute(affineMinOp.getAffineMap(), res.dims, res.symbols,
                       getMinMaxExpr);
  return res;
}

LogicalResult AffineMinRangeCanonicalizationPattern::matchAndRewrite(
    AffineMinOp minOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(DBGS() << "Canonicalize AffineMinSCF: " << *minOp.getOperation()
                    << "\n");

  auto affineMapAndOperands = substituteMin(minOp, getMinMaxFn);
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

static SmallVector<StringRef> getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<StringRef>(nParallelLoops, getParallelIteratorTypeName());
}

/// Rewrite a PadTensorOp into a sequence of InitTensorOp, FillOp (to initialize
/// with pad_val) and GenericOp (to copy contents).
LogicalResult PadTensorOpTransformationPattern::matchAndRewrite(
    linalg::PadTensorOp padOp, PatternRewriter &rewriter) const {

  auto inputShapedType = padOp.source().getType().cast<ShapedType>();
  auto resultShapedType = padOp.result().getType().cast<ShapedType>();

  // Bail on non-static shapes.
  if (!inputShapedType.hasStaticShape())
    return failure();
  if (!resultShapedType.hasStaticShape())
    return failure();

  // Only support padding with a constant for now, i.e. either:
  //   1. A BBarg from a different block.
  //   2. A value defined outside of the current block.
  Block &block = padOp.region().front();
  auto yieldOp = cast<YieldOp>(block.getTerminator());
  assert(yieldOp.getNumOperands() == 1 && "expected single operand yield");
  Value padValue = yieldOp.values().front();
  Operation *definingOp = padValue.getDefiningOp();
  if (definingOp && definingOp->getBlock() == &block)
    return failure();
  if (!definingOp && padValue.cast<BlockArgument>().getOwner() == &block)
    return failure();

  // Create tensor with the padded shape
  Location loc = padOp.getLoc();
  SmallVector<Value> indices(resultShapedType.getRank(),
                             rewriter.create<ConstantIndexOp>(loc, 0));
  Value initTensor = rewriter.create<InitTensorOp>(
      loc, resultShapedType.getShape(), resultShapedType.getElementType());

  // Initialize tensor with the pad value
  Value tmpTensor =
      rewriter.create<linalg::FillOp>(loc, padValue, initTensor).result();

  // Copy original contents into new tensor
  // Uses linalg.generic, but could be done with tensor.insert_slice
  SmallVector<AffineExpr, 4> outputExprs;
  for (unsigned i = 0; i < resultShapedType.getRank(); ++i) {
    outputExprs.push_back(getAffineDimExpr(i, rewriter.getContext()) +
                          padOp.static_low()[i].cast<IntegerAttr>().getInt());
  }

  SmallVector<AffineMap, 2> transferMaps = {
      rewriter.getMultiDimIdentityMap(inputShapedType.getRank()),
      AffineMap::get(resultShapedType.getRank(),
                     /*symbolCount=*/0, outputExprs, rewriter.getContext())};

  rewriter.replaceOpWithNewOp<linalg::GenericOp>(
      padOp, resultShapedType, padOp.source(), tmpTensor, transferMaps,
      getNParallelLoopsAttrs(resultShapedType.getRank()),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  return success();
}

/// Given an OpFoldResult, return a Value. If the OpFoldResult is an Attribute,
/// it must be of type Integer.
static Value asValue(OpBuilder &builder, Location loc, OpFoldResult ofr) {
  if (auto val = ofr.dyn_cast<Value>())
    return val;
  auto intVal = getConstantIntValue(ofr);
  assert(intVal && "expected Value or IntegerAttr");
  return builder.create<ConstantIndexOp>(loc, *intVal);
}

/// Given a value, try to extract a constant index-type integer as an Attribute.
/// If this fails, return the original value.
static OpFoldResult asOpFoldResult(OpBuilder &builder, Value val) {
  if (auto constInt = getConstantIntValue(val))
    return builder.getIndexAttr(*constInt);
  return val;
}

LogicalResult ExtractSliceOfPadTensorSwapPattern::matchAndRewrite(
    tensor::ExtractSliceOp sliceOp, PatternRewriter &rewriter) const {
  auto padOp = sliceOp.source().getDefiningOp<PadTensorOp>();
  if (!padOp)
    return failure();
  // Only unit stride supported.
  if (!sliceOp.hasUnitStride())
    return failure();
  // Only constant padding value supported.
  Value padValue = padOp.getConstantPaddingValue();
  if (!padValue)
    return failure();

  // Helper variables and functions for various arithmetic operations. These are
  // used extensively for computing new offset/length and padding values.
  Location loc = sliceOp.getLoc();
  AffineExpr dim0, dim1;
  bindDims(rewriter.getContext(), dim0, dim1);
  // Add two integers.
  auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
  auto add = [&](Value v1, Value v2) {
    return rewriter.createOrFold<AffineApplyOp>(loc, addMap,
                                                ValueRange{v1, v2});
  };
  // Subtract two integers.
  auto subMap = AffineMap::get(2, 0, {dim0 - dim1});
  auto sub = [&](Value v1, Value v2) {
    return rewriter.createOrFold<AffineApplyOp>(loc, subMap,
                                                ValueRange{v1, v2});
  };
  // Take the minimum of two integers.
  auto idMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
  auto min = [&](Value v1, Value v2) {
    return rewriter.createOrFold<AffineMinOp>(loc, idMap, ValueRange{v1, v2});
  };
  // Take the maximum of two integers.
  auto max = [&](Value v1, Value v2) {
    return rewriter.createOrFold<AffineMaxOp>(loc, idMap, ValueRange{v1, v2});
  };
  // Zero index-typed integer.
  auto zero = rewriter.create<ConstantIndexOp>(loc, 0);

  // Helper function for filling static/dynamic low/high padding indices vectors
  // of PadTensorOp.
  auto appendIndex = [&](Value val, SmallVector<Value> &dynIndices,
                         SmallVector<int64_t> &staticIndices) {
    if (auto constInt = getConstantIntValue(val)) {
      staticIndices.push_back(*constInt);
    } else {
      staticIndices.push_back(ShapedType::kDynamicSize);
      dynIndices.push_back(val);
    }
  };

  // Compute new offsets, lengths, low padding, high padding.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;
  SmallVector<Value> newLows, newHighs;
  SmallVector<int64_t> staticNewLows, staticNewHighs;
  // Set to true if the original data source is not read at all.
  bool hasZeroLen = false;
  // Same as hasZeroLen, but for dynamic dimension sizes. This condition
  // is true if the original data source turns out to be unused at runtime.
  Value dynHasZeroLenCond;

  int64_t rank = padOp.getSourceType().getRank();
  for (unsigned dim = 0; dim < rank; ++dim) {
    auto low = asValue(rewriter, loc, padOp.getMixedLowPad()[dim]);
    auto offset = asValue(rewriter, loc, sliceOp.getMixedOffsets()[dim]);
    auto length = asValue(rewriter, loc, sliceOp.getMixedSizes()[dim]);
    auto srcSize = rewriter.createOrFold<memref::DimOp>(
        loc, padOp.source(), dim);

    // The new amount of low padding is `low - offset`. Except for the case
    // where none of the low padding is read. In that case, the new amount of
    // low padding is zero.
    Value newLow = max(zero, sub(low, offset));
    appendIndex(newLow, newLows, staticNewLows);

    // Start reading the data from position `offset - low`. Since the original
    // read may have started in the low padding zone, this value could be
    // negative. Therefore, start reading from:
    //
    // max(offset - low, 0)
    //
    // The original read could also have started in the high padding zone.
    // In that case, set the offset to the end of source tensor. The new
    // ExtractSliceOp length will be zero in that case. (Effectively reading no
    // data from the source.)
    Value newOffset = min(max(sub(offset, low), zero), srcSize);
    newOffsets.push_back(asOpFoldResult(rewriter, newOffset));

    // The original ExtractSliceOp was reading until position `offset + length`.
    // Therefore, the corresponding position within the source tensor is:
    //
    // offset + length - low
    //
    // In case the original ExtractSliceOp stopped reading within the low
    // padding zone, this value can be negative. In that case, the end position
    // of the read should be zero. (Similar to newOffset.)
    //
    // The original read could also have stopped in the high padding zone.
    // In that case, set the end positition of the read should be the end of the
    // source tensor. (Similar to newOffset.)
    //
    // endLoc = min(max(offset - low + length, 0), srcSize)
    //
    // The new ExtractSliceOp length is `endLoc - newOffset`.
    Value endLoc = min(max(add(sub(offset, low), length), zero), srcSize);
    Value newLength = sub(endLoc, newOffset);
    newLengths.push_back(asOpFoldResult(rewriter, newLength));

    // Check if newLength is zero. In that case, no SubTensorOp should be
    // executed.
    if (auto newLengthInt = getConstantIntValue(newLength)) {
      hasZeroLen |= *newLengthInt == 0;
    } else {
      Value check = rewriter.create<CmpIOp>(
          loc, CmpIPredicate::eq, newLength, zero);
      dynHasZeroLenCond =
          dynHasZeroLenCond
              ? rewriter.create<OrOp>(loc, check, dynHasZeroLenCond)
              : check;
    }

    // The amount of high padding is simply the number of elements remaining,
    // so that the result has the same length as the original ExtractSliceOp.
    Value newHigh = sub(sub(length, newLength), newLow);
    appendIndex(newHigh, newHighs, staticNewHighs);

    // Only unit stride supported.
    newStrides.push_back(rewriter.getIndexAttr(1));
  }

  // Insert cast to ensure that types match. (May be folded away.)
  auto castResult = [&](Value val) -> Value {
    auto castOp = rewriter.create<tensor::CastOp>(loc, sliceOp.getType(), val);
    return castOp;
  };

  // In cases where the original data source is unused: Emit a GenerateOp and
  // do not generate a SliceOp. (The result shape of the SliceOp would
  // have a dimension of size 0, the semantics of which is unclear.)
  auto createGenerateOp = [&]() {
    // The shape of the GenerateOp is the same as the existing SliceOp.
    RankedTensorType type = sliceOp.getType();
    SmallVector<Value> dynDims;
    for (unsigned i = 0; i < type.getRank(); ++i) {
      if (type.isDynamicDim(i))
        dynDims.push_back(asValue(rewriter, loc, sliceOp.getMixedOffsets()[i]));
    }

    // Create GenerateOp.
    auto generateOp  = rewriter.create<tensor::GenerateOp>(loc, type, dynDims);

    // Copy region to new op.
    BlockAndValueMapping bvm;
    padOp.region().cloneInto(&generateOp.getRegion(), bvm);
    // Rewrite linalg::YieldOp to tensor::YieldOp.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto yieldOp = dyn_cast<linalg::YieldOp>(
          generateOp.getRegion().front().getTerminator());
      assert(yieldOp && "malformed PadTensorOp: expected YieldOp terminator");
      assert(yieldOp.values().size() == 1);
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<tensor::YieldOp>(
          yieldOp, yieldOp.values()[0]);
    }

    return castResult(generateOp);
  };

  // Emit a SliceOp and a PadTensorOp. Should not be used in cases where
  // the result shape of the new SliceOp has a zero dimension.
  auto createPadTensorOfSubTensor = [&]() {
    // Create pad_tensor(subtensor(x)).
    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, padOp.source(), newOffsets, newLengths, newStrides);
    auto newPadTensorOp = rewriter.create<PadTensorOp>(
        loc, newSliceOp, staticNewLows, staticNewHighs, newLows, newHighs);

    // Copy region to new PadTensorOp.
    BlockAndValueMapping bvm;
    padOp.region().cloneInto(&newPadTensorOp.getRegion(), bvm);

    // Cast result and return.
    return castResult(newPadTensorOp);
  };

  // Rewrite subtensor(pad_tensor(x)) into a GenerateOp it is statically known
  // that the original data source x is not used.
  if (hasZeroLen) {
    rewriter.replaceOp(sliceOp, createGenerateOp());
    return success();
  }

  // If there are dynamic dimensions: Generate an scf.if check to avoid creating
  // SliceOps with result dimensions of size 0 at runtime.
  if (dynHasZeroLenCond) {
    auto result = rewriter.create<scf::IfOp>(
        loc, sliceOp.getType(), dynHasZeroLenCond,
        /*thenBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, createGenerateOp());
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, createPadTensorOfSubTensor());
        });
    rewriter.replaceOp(sliceOp, result.getResult(0));
    return success();
  }

  // All shapes are static and the data source is actually used. Rewrite into
  // pad_tensor(subtensor(x)).
  rewriter.replaceOp(sliceOp, createPadTensorOfSubTensor());
  return success();
}
