//===- Transforms.cpp - Linalg transformations as patterns ----------------===//
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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/HoistPadding.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
#include <utility>

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
    ArrayRef<StringAttr> matchDisjunction, Optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

mlir::linalg::LinalgTransformationFilter::LinalgTransformationFilter(
    const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction,
    Optional<StringAttr> replacement)
    : filters(),
      matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {
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
    if (matchDisjunction.empty() || matchByDefault)
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
                replacement.getValue());
  else
    op->removeAttr(
        rewriter.getStringAttr(LinalgTransforms::kLinalgTransformMarker));
}

bool mlir::linalg::LinalgTransformationFilter::hasReplacementFilter(
    Operation *op) const {
  if (!replacement)
    return false;
  auto attr = op->getAttr(LinalgTransforms::kLinalgTransformMarker)
                  .dyn_cast<StringAttr>();
  return attr && attr == replacement.getValue();
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

/// Helper function that tries to pad `opOperand`. Exit early for scalar
/// operands, if `paddingFunc` returns failure, or if `opOperand` is not defined
/// by an ExtractSliceOp. Otherwise, try to pad the operand even if it already
/// has a static shape. Set `result` to the result of the created tensor::PadOp
/// or and return success if the operand either has been padded to a static
/// shape or already had a static shape and failure otherwise.
static LogicalResult padOperandToSmallestStaticBoundingBox(
    OpBuilder &b, linalg::LinalgOp opToPad, OpOperand *opOperand,
    const PaddingValueComputationFunction &paddingFunc,
    const PaddingNoFoldComputationFunction &nofoldFunc, Value &result) {
  // Get the shape of the operand and check if it has a dynamic shape. Only
  // return failure if the operand is not a scalar and has a dynamic shape.
  ArrayRef<int64_t> shape = opToPad.getShape(opOperand);
  bool hasDynamicShape = llvm::is_contained(shape, ShapedType::kDynamicSize);

  // Cannot pad scalar operands.
  if (shape.empty())
    return success();

  // Cannot pad if the padding value is unknown.
  FailureOr<Value> paddingValue = paddingFunc(b, *opOperand);
  if (failed(paddingValue))
    return failure(hasDynamicShape);

  // Cannot construct a static bounding box if the operand is not defined by an
  // ExtractSliceOp.
  auto sliceOp = opOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return failure(hasDynamicShape);

  // Compute the dropped dimensions if `sliceOp` is ranke-reducing.
  llvm::SmallDenseSet<unsigned> droppedDims = sliceOp.getDroppedDims();

  // Upper bound the `sliceOp` sizes to obtain a static bounding box.
  SmallVector<int64_t> staticSizes;
  staticSizes.reserve(shape.size());
  auto shapedOp = cast<OffsetSizeAndStrideOpInterface>(sliceOp.getOperation());
  for (const auto &en : enumerate(shapedOp.getMixedSizes())) {
    // Skip dropped dimensions.
    if (droppedDims.contains(en.index()))
      continue;
    // If the size is an attribute add it directly to `staticSizes`.
    if (en.value().is<Attribute>()) {
      staticSizes.push_back(
          en.value().get<Attribute>().dyn_cast<IntegerAttr>().getInt());
      continue;
    }
    // Otherwise, try to compute a constant upper bound for the size value.
    FailureOr<int64_t> upperBound =
        getConstantUpperBoundForIndex(en.value().get<Value>());
    if (failed(upperBound)) {
      LLVM_DEBUG(DBGS() << "No constant bounding box can be found for padding");
      return failure();
    }
    staticSizes.push_back(upperBound.getValue());
  }
  assert(staticSizes.size() == shape.size() &&
         "expect the dynamic and static ranks to match");

  // Pad the operand to the bounding box defined by `staticSizes`.
  auto staticTensorType = RankedTensorType::get(
      staticSizes, getElementTypeOrSelf(opOperand->get()));
  bool nofold = nofoldFunc ? nofoldFunc(*opOperand) : false;
  result =
      makeComposedPadHighOp(b, opToPad->getLoc(), staticTensorType,
                            opOperand->get(), paddingValue.getValue(), nofold);
  return success();
}

FailureOr<SmallVector<Value>>
linalg::rewriteAsPaddedOp(OpBuilder &b, LinalgOp opToPad,
                          const PaddingValueComputationFunction &paddingFunc,
                          const PaddingNoFoldComputationFunction &nofoldFunc,
                          LinalgOp &paddedOp) {
  Location loc = opToPad->getLoc();

  // TODO: there are cases where we may still want to pad to larger sizes.
  assert(opToPad.hasTensorSemantics() &&
         "expected operation to have tensor semantics");

  OpBuilder::InsertionGuard g(b);
  // Set IP after op because we also take the dims of the original output.
  b.setInsertionPointAfter(opToPad);
  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad.getNumInputsAndOutputs());
  for (OpOperand *opOperand : opToPad.getInputAndOutputOperands()) {
    Value paddedOperand;
    // If padding was requested but the shape cannot be bounded statically then
    // the pattern fails to apply.
    if (failed(padOperandToSmallestStaticBoundingBox(
            b, opToPad, opOperand, paddingFunc, nofoldFunc, paddedOperand)))
      return failure();
    newOperands.push_back(paddedOperand ? paddedOperand : opOperand->get());
  }

  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(cast<ReifyRankedShapedTypeOpInterface>(opToPad.getOperation())
                 .reifyResultShapes(b, reifiedResultShapes)))
    return failure();
  assert(reifiedResultShapes.size() == opToPad->getNumResults() &&
         "expected same number of results");

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumOutputs()).getTypes();
  paddedOp = opToPad.clone(b, loc, resultTensorTypes, newOperands);

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(opToPad->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = paddedResult.getType().cast<RankedTensorType>().getRank();
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (Value v : reifiedResultShapes[resultNumber])
      sizes.push_back(getAsOpFoldResult(v));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
    paddedSubviewResults.push_back(b.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, sizes, strides));
  }
  return paddedSubviewResults;
}

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
void mlir::linalg::peelTiledLinalgOp(RewriterBase &rewriter, TiledLinalgOp &res,
                                     ArrayRef<int64_t> peeledLoops,
                                     LinalgTilingLoopType loopType) {
  for (int64_t loop : peeledLoops) {
    assert(loop < static_cast<int64_t>(res.loops.size()) &&
           "requested peeling of non-existing loop");
    SmallVector<Value, 4> loopResults;
    Operation *loopOp = res.loops[loop];
    if (loopType == LinalgTilingLoopType::TiledLoops) {
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
    LinalgTransformationFilter f, LinalgTransformationFilter fusedOpMarker,
    LinalgTransformationFilter originalOpMarker, PatternBenefit benefit)
    : RewritePattern(opName, benefit, context, {}),
      dependenceGraph(dependenceGraph), tilingOptions(std::move(tilingOptions)),
      fusionOptions(std::move(fusionOptions)), filter(std::move(f)),
      fusedOpMarker(std::move(fusedOpMarker)),
      originalOpMarker(std::move(originalOpMarker)) {}

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
  for (const auto &tileSize : enumerate(tileSizes)) {
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
    FailureOr<TiledLinalgOp> unfusedTiledOp =
        tileLinalgOp(rewriter, tiledAndFusedOps->op, unfusedTilingOptions);
    if (failed(unfusedTiledOp))
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

/// Linalg tiling pattern.
mlir::linalg::LinalgTilingPattern::LinalgTilingPattern(
    MLIRContext *context, LinalgTilingOptions options,
    LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(std::move(f)), options(std::move(options)) {}

mlir::linalg::LinalgTilingPattern::LinalgTilingPattern(
    StringRef opName, MLIRContext *context, LinalgTilingOptions options,
    LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)), options(std::move(options)) {}

FailureOr<TiledLinalgOp>
mlir::linalg::LinalgTilingPattern::returningMatchAndRewrite(
    LinalgOp op, PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, op)))
    return failure();

  FailureOr<TiledLinalgOp> res = tileLinalgOp(rewriter, op, options);
  if (failed(res))
    return failure();

  // Clear filter to stop recursive pattern application.
  // This must be done here to properly propagate to peeling branches.
  filter.replaceLinalgTransformationFilter(rewriter, res->op);

  // Peel the loops of the TiledLinalgOp.
  peelTiledLinalgOp(rewriter, *res, options.peeledLoops, options.loopType);

  if (res->tensorResults.empty())
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, res->tensorResults);

  return res;
}

/// Linalg padding pattern.
mlir::linalg::LinalgPaddingPattern::LinalgPaddingPattern(
    MLIRContext *context, LinalgPaddingOptions options,
    LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(std::move(f)), options(std::move(options)) {}

mlir::linalg::LinalgPaddingPattern::LinalgPaddingPattern(
    StringRef opName, MLIRContext *context, LinalgPaddingOptions options,
    LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)), options(std::move(options)) {}

FailureOr<LinalgOp>
mlir::linalg::LinalgPaddingPattern::returningMatchAndRewrite(
    LinalgOp linalgOp, PatternRewriter &rewriter) const {
  if (!linalgOp.hasTensorSemantics())
    return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();

  // Pad the operation.
  LinalgOp paddedOp;
  FailureOr<SmallVector<Value>> newResults = rewriteAsPaddedOp(
      rewriter, linalgOp, options.paddingValueComputationFunction,
      options.paddingNoFoldComputationFunction, paddedOp);
  if (failed(newResults))
    return failure();

  // Compute the desired hoisting depths.
  SmallVector<int64_t> depths;
  if (options.paddingHoistComputationFunction) {
    for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands())
      depths.push_back(options.paddingHoistComputationFunction(*opOperand));
  }

  // Hoist the padding.
  for (const auto &en : enumerate(depths)) {
    OpOperand &opOperand = paddedOp->getOpOperand(en.index());
    auto padOp = opOperand.get().getDefiningOp<tensor::PadOp>();
    if (!padOp || en.value() == 0)
      continue;
    tensor::PadOp hoistedOp;
    SmallVector<GenericOp> transposeOps;
    SmallVector<int64_t> transposeVector =
        options.paddingTransposeComputationFunction(opOperand);

    FailureOr<Value> newResult = hoistPaddingOnTensors(
        padOp, en.value(), transposeVector, hoistedOp, transposeOps);
    if (failed(newResult))
      continue;
    rewriter.replaceOp(padOp, newResult.getValue());

    // Do not apply hoist padding to the newly introduced transpose operations.
    for (GenericOp transposeOp : transposeOps)
      filter.replaceLinalgTransformationFilter(rewriter, transposeOp);
  }

  // Replace the original operation to pad.
  rewriter.replaceOp(linalgOp, newResults.getValue());
  filter.replaceLinalgTransformationFilter(rewriter, paddedOp);

  return paddedOp;
}

/// Linalg tile and fuse tensor ops pattern.
mlir::linalg::LinalgTileAndFuseTensorOpsPattern::
    LinalgTileAndFuseTensorOpsPattern(MLIRContext *context,
                                      LinalgTilingAndFusionOptions options,
                                      LinalgTransformationFilter f,
                                      PatternBenefit benefit)
    : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
      filter(std::move(f)), options(std::move(options)) {}

mlir::linalg::LinalgTileAndFuseTensorOpsPattern::
    LinalgTileAndFuseTensorOpsPattern(StringRef opName, MLIRContext *context,
                                      LinalgTilingAndFusionOptions options,
                                      LinalgTransformationFilter f,
                                      PatternBenefit benefit)
    : RewritePattern(opName, benefit, context), filter(std::move(f)),
      options(std::move(options)) {}

LogicalResult mlir::linalg::LinalgTileAndFuseTensorOpsPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp rootOp = dyn_cast<LinalgOp>(op);
  if (!rootOp)
    return failure();
  if (failed(filter.checkAndNotify(rewriter, op)))
    return failure();

  // Check `tileSizes` contains a tile size for every `rootOp` loop dimension.
  if (options.tileSizes.size() < rootOp.getNumLoops())
    return rewriter.notifyMatchFailure(op, "expect #tile sizes >= #loops");

  // Check `tileInterchange` contains no entries or as many as `tileSizes`.
  if (!options.tileInterchange.empty() &&
      options.tileInterchange.size() != options.tileSizes.size())
    return rewriter.notifyMatchFailure(
        op, "expect the number of tile sizes and interchange dims to match");

  // Copy the `tileSizes` and `tileInterchange` prefixes needed for `rootOp`.
  SmallVector<int64_t> rootTileSizes(options.tileSizes.begin(),
                                     options.tileSizes.begin() +
                                         rootOp.getNumLoops());
  SmallVector<int64_t> rootInterchange =
      options.tileInterchange.empty()
          ? llvm::to_vector<6>(llvm::seq<int64_t>(0, rootOp.getNumLoops()))
          : SmallVector<int64_t>(options.tileInterchange.begin(),
                                 options.tileInterchange.begin() +
                                     rootOp.getNumLoops());

  // Check `rootInterchange` is a permutation of the `rootOp` loop dimensions.
  // It has to be a permutation since the tiling cannot tile the same loop
  // dimension multiple times.
  if (!isPermutation(rootInterchange))
    return rewriter.notifyMatchFailure(
        op, "expect the tile interchange permutes the root loops");

  // Tile `rootOp` and fuse its producers.
  FailureOr<TileLoopNest> tileLoopNest = tileConsumerAndFuseProducers(
      rewriter, rootOp, rootTileSizes, rootInterchange);
  if (failed(tileLoopNest))
    return rewriter.notifyMatchFailure(
        op, "tileConsumerAndFuseProducers failed unexpectedly");

  // Replace all uses of the tiled loop operation.
  rootOp->replaceAllUsesWith(tileLoopNest->getRootOpReplacementResults());

  // Apply the filter if specified.
  for (LinalgOp linalgOp : tileLoopNest->getAllTiledAndFusedOps())
    filter.replaceLinalgTransformationFilter(rewriter, linalgOp);
  return failure();
}

/// Linalg generic interchange pattern.
mlir::linalg::GenericOpInterchangePattern::GenericOpInterchangePattern(
    MLIRContext *context, ArrayRef<unsigned> interchangeVector,
    LinalgTransformationFilter f, PatternBenefit benefit)
    : OpRewritePattern(context, benefit), filter(std::move(f)),
      interchangeVector(interchangeVector.begin(), interchangeVector.end()) {}

FailureOr<GenericOp>
mlir::linalg::GenericOpInterchangePattern::returningMatchAndRewrite(
    GenericOp genericOp, PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, genericOp)))
    return failure();

  FailureOr<GenericOp> transformedOp =
      interchangeGenericOp(rewriter, genericOp, interchangeVector);
  if (failed(transformedOp))
    return failure();

  // New filter if specified.
  filter.replaceLinalgTransformationFilter(rewriter, genericOp);
  return transformedOp;
}

/// Linalg generalization pattern.
mlir::linalg::LinalgGeneralizationPattern::LinalgGeneralizationPattern(
    MLIRContext *context, LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(std::move(f)) {}

mlir::linalg::LinalgGeneralizationPattern::LinalgGeneralizationPattern(
    StringRef opName, MLIRContext *context, LinalgTransformationFilter f,
    PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)) {}

FailureOr<GenericOp>
mlir::linalg::LinalgGeneralizationPattern::returningMatchAndRewrite(
    LinalgOp linalgOp, PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();
  FailureOr<GenericOp> genericOp = generalizeNamedOp(rewriter, linalgOp);
  if (failed(genericOp))
    return failure();
  filter.replaceLinalgTransformationFilter(rewriter, *genericOp);
  return genericOp;
}

mlir::linalg::LinalgBasePromotionPattern::LinalgBasePromotionPattern(
    MLIRContext *context, LinalgTransformationFilter f,
    LinalgPromotionOptions options, PatternBenefit benefit)
    : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
      filter(std::move(f)), options(std::move(options)) {}

mlir::linalg::LinalgBasePromotionPattern::LinalgBasePromotionPattern(
    StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
    LinalgTransformationFilter f, PatternBenefit benefit)
    : RewritePattern(opName, benefit, context, {}), filter(std::move(f)),
      options(std::move(options)) {}

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

mlir::linalg::LinalgVectorizationPattern::LinalgVectorizationPattern(
    MLIRContext *context, LinalgTransformationFilter f,
    LinalgVectorizationOptions options, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(std::move(f)) {}

mlir::linalg::LinalgVectorizationPattern::LinalgVectorizationPattern(
    StringRef opName, MLIRContext *context, LinalgVectorizationOptions options,
    LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)) {}

LogicalResult mlir::linalg::LinalgVectorizationPattern::matchAndRewrite(
    LinalgOp linalgOp, PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();
  return vectorize(rewriter, linalgOp);
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

/// Rewrite a tensor::PadOp into a sequence of InitTensorOp, FillOp (to
/// initialize with pad_val) and GenericOp (to copy contents).
LogicalResult
PadOpTransformationPattern::matchAndRewrite(tensor::PadOp padOp,
                                            PatternRewriter &rewriter) const {

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
  auto yieldOp = cast<tensor::YieldOp>(block.getTerminator());
  Value padValue = yieldOp.value();
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
Value GeneralizePadOpPattern::createFillOrGenerateOp(
    PatternRewriter &rewriter, tensor::PadOp padOp, Value dest,
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
  return generateOp;
}

LogicalResult
GeneralizePadOpPattern::matchAndRewrite(tensor::PadOp padOp,
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

  // tensor::PadOps cannot be optimized. Generate a InsertSliceOp instead
  // for copying the PadOp source.
  auto sourceType = padOp.getSourceType();
  // Compute size of source of tensor::PadOp.
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
  auto padOp = sliceOp.source().getDefiningOp<tensor::PadOp>();
  if (!padOp)
    return failure();
  // Only unit stride supported.
  if (!sliceOp.hasUnitStride())
    return failure();

  TilingInterface tilingInterface =
      dyn_cast<TilingInterface>(padOp.getOperation());
  Operation *tiledPadOp =
      tilingInterface
          .getTiledImplementation(
              rewriter, /*dest=*/ValueRange{}, sliceOp.getMixedOffsets(),
              sliceOp.getMixedSizes(), /*tileDestOperands=*/false)
          .front();
  // All shapes are static and the data source is actually used. Rewrite into
  // pad_tensor(subtensor(x)).
  rewriter.replaceOp(sliceOp, tiledPadOp->getResults());
  return success();
}

namespace {
// The following are patterns for downscaling convolution ops with size-1
// window dimensions.
//
// Note that we'd eventually want to write such transformations in a generic
// way, e.g., converting to linalg.generic, removing the size-1 dimensions,
// and then turning back to named ops. But for now it's fine to have a few
// patterns matching special ops to get started.

/// Rewrites 2-D convolution ops with size-1 window dimensions into 1-D
/// convolution ops.
struct DownscaleSizeOneWindowed2DConvolution final
    : public OpRewritePattern<Conv2DNhwcHwcfOp> {
  DownscaleSizeOneWindowed2DConvolution(
      MLIRContext *context,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : OpRewritePattern<Conv2DNhwcHwcfOp>(context, benefit),
        filter(std::move(f)) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, convOp)))
      return failure();
    if (convOp.hasBufferSemantics())
      return failure(); // To be implemented

    Value input = convOp.inputs().front();
    Value kernel = convOp.inputs().back();
    Value output = convOp.outputs().front();

    auto inputType = input.getType().dyn_cast<RankedTensorType>();
    auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
    auto outputType = output.getType().dyn_cast<RankedTensorType>();

    auto kernelShape = kernelType.getShape();
    auto outputShape = outputType.getShape();

    // Only handle the case where at least one of the window dimensions is
    // of size 1. Other cases can rely on tiling to reduce to such cases.
    int64_t khSize = kernelShape[0], kwSize = kernelShape[1];
    int64_t ohSize = outputShape[1], owSize = outputShape[2];
    bool removeH = (khSize == 1 && ohSize == 1);
    bool removeW = (kwSize == 1 && owSize == 1);
    if (!removeH && !removeW)
      return failure();

    // Get new shapes and types for all operands by removing the size-1
    // dimension.
    using RTTBuilder = RankedTensorType::Builder;
    RankedTensorType newInputType =
        RTTBuilder(inputType).dropDim((removeH ? 1 : 2));
    RankedTensorType newKernelType =
        RTTBuilder(kernelType).dropDim((removeH ? 0 : 1));
    RankedTensorType newOutputType =
        RTTBuilder(outputType).dropDim(removeH ? 1 : 2);

    // Rank-reduce operands.
    Location loc = convOp.getLoc();
    Value newInput = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, input, newInputType);
    Value newKernel = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, kernel, newKernelType);
    Value newOutput = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, output, newOutputType);

    // Rank-reduce strides and dilations too.
    // TODO: dropDim 1-liner helper.
    auto strides = llvm::to_vector<4>(convOp.strides().getValues<int64_t>());
    strides.erase(strides.begin() + (removeH ? 0 : 1));
    auto stridesAttr = rewriter.getI64VectorAttr(strides);

    auto dilations =
        llvm::to_vector<4>(convOp.dilations().getValues<int64_t>());
    dilations.erase(dilations.begin() + (removeH ? 0 : 1));
    auto dilationsAttr = rewriter.getI64VectorAttr(dilations);

    auto conv1DOp = rewriter.create<linalg::Conv1DNwcWcfOp>(
        loc, newOutputType, ValueRange{newInput, newKernel},
        ValueRange{newOutput}, stridesAttr, dilationsAttr);

    // Insert back.
    Value inserted = tensor::createCanonicalRankReducingInsertSliceOp(
        rewriter, loc, conv1DOp.getResult(0), output);
    rewriter.replaceOp(convOp, inserted);

    filter.replaceLinalgTransformationFilter(rewriter, conv1DOp);
    return success();
  };

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
};

/// Rewrites 2-D depthwise convolution ops with size-1 (w, kw) or (h, kh)
/// dimensions into 1-D depthwise convolution ops.
struct DownscaleDepthwiseConv2DNhwcHwcOp final
    : public OpRewritePattern<DepthwiseConv2DNhwcHwcOp> {
  DownscaleDepthwiseConv2DNhwcHwcOp(
      MLIRContext *context,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : OpRewritePattern<DepthwiseConv2DNhwcHwcOp>(context, benefit),
        filter(std::move(f)) {}

  LogicalResult matchAndRewrite(DepthwiseConv2DNhwcHwcOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, convOp)))
      return failure();
    if (convOp.hasBufferSemantics())
      return failure(); // To be implemented

    Value input = convOp.inputs().front();
    Value kernel = convOp.inputs().back();
    Value output = convOp.outputs().front();

    auto inputType = input.getType().dyn_cast<RankedTensorType>();
    auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
    auto outputType = output.getType().dyn_cast<RankedTensorType>();

    auto kernelShape = kernelType.getShape();
    auto outputShape = outputType.getShape();

    // Only handle the case where at least one of the window dimensions is
    // of size 1. Other cases can rely on tiling to reduce to such cases.
    int64_t khSize = kernelShape[0], kwSize = kernelShape[1];
    int64_t ohSize = outputShape[1], owSize = outputShape[2];
    bool removeH = (khSize == 1 && ohSize == 1);
    bool removeW = (kwSize == 1 && owSize == 1);
    if (!removeH && !removeW)
      return failure();

    // Get new shapes and types for all operands by removing the size-1
    // dimension.
    using RTTBuilder = RankedTensorType::Builder;
    RankedTensorType newInputType =
        RTTBuilder(inputType).dropDim((removeH ? 1 : 2));
    RankedTensorType newKernelType =
        RTTBuilder(kernelType).dropDim((removeH ? 0 : 1));
    RankedTensorType newOutputType =
        RTTBuilder(outputType).dropDim(removeH ? 1 : 2);

    // Rank-reduce operands.
    Location loc = convOp.getLoc();
    Value newInput = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, input, newInputType);
    Value newKernel = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, kernel, newKernelType);
    Value newOutput = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, output, newOutputType);

    // Rank-reduce strides and dilations too.
    // TODO: dropDim 1-liner helper.
    auto strides = llvm::to_vector<4>(convOp.strides().getValues<int64_t>());
    strides.erase(strides.begin() + (removeH ? 0 : 1));
    auto stridesAttr = rewriter.getI64VectorAttr(strides);

    auto dilations =
        llvm::to_vector<4>(convOp.dilations().getValues<int64_t>());
    dilations.erase(dilations.begin() + (removeH ? 0 : 1));
    auto dilationsAttr = rewriter.getI64VectorAttr(dilations);

    auto conv1DOp = rewriter.create<DepthwiseConv1DNwcWcOp>(
        loc, newOutputType, ValueRange{newInput, newKernel},
        ValueRange{newOutput}, stridesAttr, dilationsAttr);

    // Insert back.
    Value inserted = tensor::createCanonicalRankReducingInsertSliceOp(
        rewriter, loc, conv1DOp.getResult(0), output);
    rewriter.replaceOp(convOp, inserted);

    filter.replaceLinalgTransformationFilter(rewriter, conv1DOp);
    return success();
  };

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
};

} // namespace

void linalg::populateDecomposeConvolutionPatterns(
    RewritePatternSet &patterns, const LinalgTransformationFilter &filter,
    PatternBenefit benefit) {
  patterns.add<DownscaleSizeOneWindowed2DConvolution,
               DownscaleDepthwiseConv2DNhwcHwcOp>(patterns.getContext(), filter,
                                                  benefit);
}
