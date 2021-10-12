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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
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
  assert(!tileSizeComputationFunction && "tile sizes already set");
  SmallVector<int64_t, 4> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentOfType<FuncOp>().getBody().front());
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      Value v = b.create<arith::ConstantIndexOp>(op->getLoc(), s);
      return v;
    }));
  };
  return *this;
}

LinalgTilingOptions &mlir::linalg::LinalgTilingOptions::scalarizeDynamicDims() {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  tileSizeComputationFunction = [](OpBuilder &b, Operation *op) {
    SmallVector<Value, 4> tileSizes;
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return tileSizes;
    Location loc = linalgOp.getLoc();
    auto allShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
    AffineMap map = linalgOp.getShapesToLoopsMap();
    if (!map)
      return tileSizes;
    auto shapeSizes = applyMapToValues(b, loc, map, allShapeSizes);
    // If the shape size is dynamic, tile by 1. Otherwise, do not tile (tile
    // size 0).
    for (Value shapeSize : shapeSizes)
      tileSizes.push_back(getConstantIntValue(shapeSize).hasValue()
                              ? b.create<arith::ConstantIndexOp>(loc, 0)
                              : b.create<arith::ConstantIndexOp>(loc, 1));
    return tileSizes;
  };
  return *this;
}

/// Helper function that tries to pad `opOperand`. Exit early and return success
/// for scalar operands or if `paddingFunc` returns failure. Otherwise, try to
/// pad the operand even if it already has a static shape. Set `result` to the
/// result of the created PadTensorOp or return failure if the operand cannot be
/// padded to a static shape.
static LogicalResult padOperandToSmallestStaticBoundingBox(
    PatternRewriter &rewriter, linalg::LinalgOp opToPad, OpOperand *opOperand,
    const PaddingValueComputationFunction &paddingFunc, Value &result) {
  // Can't pad scalars.
  if (opToPad.getShape(opOperand).empty())
    return success();
  // Can't pad if no padding value is known.
  FailureOr<Value> paddingValue = paddingFunc(rewriter, *opOperand);
  if (failed(paddingValue))
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
  auto staticTensorType = RankedTensorType::get(
      staticSizes, getElementTypeOrSelf(opOperand->get()));
  result = linalg::PadTensorOp::createPadHighOp(
      staticTensorType, opOperand->get(), paddingValue.getValue(),
      /*nofold=*/true, opToPad->getLoc(), rewriter);
  return success();
}

LogicalResult
linalg::rewriteAsPaddedOp(PatternRewriter &rewriter, LinalgOp opToPad,
                          const PaddingValueComputationFunction &paddingFunc,
                          LinalgOp &paddedOp) {
  Location loc = opToPad->getLoc();

  // TODO: there are cases where we may still want to pad to larger sizes.
  assert(opToPad.hasTensorSemantics() &&
         "expected operation to have tensor semantics");

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
            rewriter, opToPad, opOperand, paddingFunc, paddedOperand)))
      return failure();
    newOperands.push_back(paddedOperand ? paddedOperand : opOperand->get());
  }

  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(cast<ReifyRankedShapedTypeOpInterface>(opToPad.getOperation())
                 .reifyResultShapes(rewriter, reifiedResultShapes)))
    return failure();
  assert(reifiedResultShapes.size() == opToPad->getNumResults() &&
         "expected same number of results");

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumOutputs()).getTypes();
  paddedOp = opToPad.clone(rewriter, loc, resultTensorTypes, newOperands);

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(opToPad->getNumResults());
  for (auto en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = paddedResult.getType().cast<RankedTensorType>().getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (Value v : reifiedResultShapes[resultNumber])
      sizes.push_back(v);
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubviewResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, sizes, strides));
  }
  rewriter.replaceOp(opToPad, paddedSubviewResults);
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

/// Try to peel a loop `op` and return the new result.
// TODO: Add support for scf.parallel and affine.for loops.
static SmallVector<Value, 4> peelLoop(RewriterBase &rewriter, Operation *op) {
  return llvm::TypeSwitch<Operation *, SmallVector<Value, 4>>(op)
      .Case<scf::ForOp>([&](scf::ForOp forOp) {
        scf::ForOp partialIteration;
        if (succeeded(scf::peelAndCanonicalizeForLoop(rewriter, forOp,
                                                      partialIteration)))
          return partialIteration->getResults();
        assert(!partialIteration && "expected that loop was not peeled");
        return forOp->getResults();
      })
      .Default([&](Operation *op) { return op->getResults(); });
}

/// Try to peel a TiledLoopOp and return the new result.
static SmallVector<Value, 4> peelLoop(RewriterBase &rewriter,
                                      TiledLoopOp tiledLoop, int64_t idx) {
  assert(idx < static_cast<int64_t>(tiledLoop.iterator_types().size()) &&
         "requested peeling of non-existing loop");
  TiledLoopOp result;
  if (succeeded(peelAndCanonicalizeTiledLoop(rewriter, tiledLoop, idx, result)))
    return result->getResults();
  assert(!result && "expected that loop was not peeled");
  return tiledLoop->getResults();
}

/// Peel loops after tiling.
static void peelLoops(RewriterBase &rewriter, TiledLinalgOp &res,
                      const LinalgTilingOptions &options) {
  for (int64_t loop : options.peeledLoops) {
    assert(loop < static_cast<int64_t>(res.loops.size()) &&
           "requested peeling of non-existing loop");
    SmallVector<Value, 4> loopResults;
    Operation *loopOp = res.loops[loop];
    if (options.loopType == LinalgTilingLoopType::TiledLoops) {
      assert(llvm::all_of(
                 res.loops,
                 [&](Operation *op) { return op == res.loops.front(); }) &&
             "expected that all loop ops are the same TiledLoopOp");
      auto tiledLoopOp = dyn_cast<TiledLoopOp>(loopOp);
      assert(tiledLoopOp && "expected TiledLoopOp");
      loopResults = peelLoop(rewriter, tiledLoopOp, loop);
    } else {
      loopResults = peelLoop(rewriter, loopOp);
    }

    // The result of the loop nest may change with peeling.
    if (res.tensorResults.size() == loopOp->getNumResults() &&
        std::equal(res.tensorResults.begin(), res.tensorResults.end(),
                   loopOp->getResults().begin()))
      res.tensorResults = loopResults;
  }
}

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
  // Clear filter to stop recursive pattern application.
  filter.replaceLinalgTransformationFilter(rewriter, res->op);

  // Peel loops.
  peelLoops(rewriter, *res, options);

  // Consider padding on the fly only if the op has tensor semantics.
  if (!options.paddingValueComputationFunction ||
      !linalgOp.hasTensorSemantics()) {
    result = *res;
    return success();
  }

  // Try to pad on the fly by rewriting res->op as a padded op. If successful,
  // `res.op` is rewritten in static form with padded operands.
  LinalgOp paddedOp;
  if (succeeded(rewriteAsPaddedOp(rewriter, res->op,
                                  options.paddingValueComputationFunction,
                                  paddedOp))) {
    filter.replaceLinalgTransformationFilter(rewriter, paddedOp);
    res->op = paddedOp;
    result = *res;
    // Do not perform replacement of `linalgOp`, let the derived patterns
    // do this as they see fit, from the resulting TiledLinalgOp.
    return success();
  }
  // Set so RAII guard does not propagate TiledLinalgOp to `result`.
  return failure();
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
  Value zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
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
        if (auto cst = val.getDefiningOp<arith::ConstantIndexOp>())
          return cst.value() != 0;
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

/// Linalg generalization pattern.
mlir::linalg::LinalgGeneralizationPattern::LinalgGeneralizationPattern(
    MLIRContext *context, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter) {}

mlir::linalg::LinalgGeneralizationPattern::LinalgGeneralizationPattern(
    StringRef opName, MLIRContext *context, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(opName, benefit, context, {}), filter(filter) {}

LogicalResult mlir::linalg::LinalgGeneralizationPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, op)))
    return failure();
  if (failed(generalizeNamedOpPrecondition(op)))
    return failure();

  GenericOp genericOp = generalizeNamedOp(rewriter, op);
  rewriter.replaceOp(op, genericOp.getResults());
  filter.replaceLinalgTransformationFilter(rewriter, genericOp);
  return success();
}

mlir::linalg::LinalgBasePromotionPattern::LinalgBasePromotionPattern(
    MLIRContext *context, LinalgTransformationFilter filter,
    LinalgPromotionOptions options, PatternBenefit benefit)
    : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter),
      options(options) {}

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
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
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

/// Filling `dest` using FillOp constant padding value if possible.
/// Otherwise, generate a tensor::GenerateOp.
Value GeneralizePadTensorOpPattern::createFillOrGenerateOp(
    PatternRewriter &rewriter, PadTensorOp padOp, Value dest,
    const SmallVector<Value> &dynSizes) const {
  auto padValue = padOp.getConstantPaddingValue();
  if (padValue)
    return rewriter.create<FillOp>(padOp.getLoc(), padValue, dest).result();

  // Fill could not be optimized: Lower to tensor::GenerateOp with region.
  auto generateOp = rewriter.create<tensor::GenerateOp>(
      padOp.getLoc(), padOp.getResultType(), dynSizes);
  // Copy region to new op.
  BlockAndValueMapping bvm;
  padOp.region().cloneInto(&generateOp.getRegion(), bvm);
  // Rewrite linalg::YieldOp to tensor::YieldOp.
  OpBuilder::InsertionGuard guard(rewriter);
  auto yieldOp =
      dyn_cast<linalg::YieldOp>(generateOp.getRegion().front().getTerminator());
  assert(yieldOp && "malformed PadTensorOp: expected YieldOp terminator");
  assert(yieldOp.values().size() == 1);
  rewriter.setInsertionPoint(yieldOp);
  rewriter.replaceOpWithNewOp<tensor::YieldOp>(yieldOp, yieldOp.values()[0]);
  return generateOp;
}

LogicalResult
GeneralizePadTensorOpPattern::matchAndRewrite(PadTensorOp padOp,
                                              PatternRewriter &rewriter) const {
  // Given an OpFoldResult, return an index-typed value.
  auto getIdxValue = [&](OpFoldResult ofr) {
    if (auto val = ofr.dyn_cast<Value>())
      return val;
    return rewriter
        .create<arith::ConstantIndexOp>(
            padOp.getLoc(), ofr.get<Attribute>().cast<IntegerAttr>().getInt())
        .getResult();
  };

  auto resultType = padOp.getResultType();
  // Compute size of InitTensorOp. Any combination of static/dynamic is
  // supported.
  SmallVector<Value> dynSizes;
  SmallVector<int64_t> staticSizes;
  for (unsigned dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim)) {
      auto srcSize = rewriter.createOrFold<tensor::DimOp>(padOp.getLoc(),
                                                          padOp.source(), dim);
      // Add low and high padding value.
      auto plusLow = rewriter.createOrFold<arith::AddIOp>(
          padOp.getLoc(), srcSize, getIdxValue(padOp.getMixedLowPad()[dim]));
      auto plusHigh = rewriter.createOrFold<arith::AddIOp>(
          padOp.getLoc(), plusLow, getIdxValue(padOp.getMixedHighPad()[dim]));
      dynSizes.push_back(plusHigh);
    }
    staticSizes.push_back(resultType.getDimSize(dim));
  }

  // Init tensor and fill it with padding.
  Value init = rewriter.create<InitTensorOp>(
      padOp.getLoc(), dynSizes, staticSizes, resultType.getElementType());
  Value fill = createFillOrGenerateOp(rewriter, padOp, init, dynSizes);

  // Try optimize the copy of source.
  if (optimizeCopyFn && optimizeCopyFn(rewriter, padOp, fill).succeeded())
    return success();

  // PadTensorOps cannot be optimized. Generate a InsertSliceOp instead
  // for copying the PadOp source.
  auto sourceType = padOp.getSourceType();
  // Compute size of source of PadTensorOp.
  SmallVector<OpFoldResult> srcSizes;
  for (unsigned dim = 0; dim < sourceType.getRank(); ++dim) {
    if (sourceType.isDynamicDim(dim)) {
      srcSizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          padOp.getLoc(), padOp.source(), dim));
    } else {
      srcSizes.push_back(rewriter.getIndexAttr(sourceType.getDimSize(dim)));
    }
  }
  // Strides of InsertSliceOp are all 1.
  SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                    rewriter.getIndexAttr(1));
  rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
      padOp, padOp.source(), fill, padOp.getMixedLowPad(), srcSizes, strides);

  return success();
}

LogicalResult ExtractSliceOfPadTensorSwapPattern::matchAndRewrite(
    tensor::ExtractSliceOp sliceOp, PatternRewriter &rewriter) const {
  auto padOp = sliceOp.source().getDefiningOp<PadTensorOp>();
  if (!padOp)
    return failure();
  // Only unit stride supported.
  if (!sliceOp.hasUnitStride())
    return failure();

  Operation *tiledPadOp = padOp.getTiledImplementation(
      rewriter, /*dest=*/ValueRange{}, sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes());
  // All shapes are static and the data source is actually used. Rewrite into
  // pad_tensor(subtensor(x)).
  rewriter.replaceOp(sliceOp, tiledPadOp->getResults());
  return success();
}
