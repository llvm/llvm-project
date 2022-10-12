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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/HoistPadding.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
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

LinalgTilingOptions &
mlir::linalg::LinalgTilingOptions::setTileSizes(ArrayRef<int64_t> ts) {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  SmallVector<int64_t, 4> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentOfType<func::FuncOp>().getBody().front());
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
    SmallVector<OpFoldResult> allShapeSizes =
        linalgOp.createFlatListOfOperandDims(b, loc);
    AffineMap map = linalgOp.getShapesToLoopsMap();
    if (!map)
      return tileSizes;
    IRRewriter rewriter(b);
    SmallVector<OpFoldResult> shapeSizes =
        makeComposedFoldedMultiResultAffineApply(rewriter, loc, map,
                                                 allShapeSizes);
    // If the shape size is dynamic, tile by 1. Otherwise, do not tile (tile
    // size 0).
    for (OpFoldResult shapeSize : shapeSizes)
      tileSizes.push_back(getConstantIntValue(shapeSize)
                              ? b.create<arith::ConstantIndexOp>(loc, 0)
                              : b.create<arith::ConstantIndexOp>(loc, 1));
    return tileSizes;
  };
  return *this;
}

/// Pad the `opOperand` in the `paddingDimensions` using the padding value and
/// the nofold flag found in `paddingValues` and `packPaddings`, respectively.
/// Exit early and return the `opOperand` value if the shape dimensions that
/// match `paddingDimensions` have a static size and the nofold flag is not set.
/// Otherwise, try to pad the shape dimensions that match the iterator
/// dimensions `paddingDimensions` and return the tensor::PadOp result if
/// padding succeeds or failure otherwise.
static FailureOr<Value> padOperandToSmallestStaticBoundingBox(
    OpBuilder &b, linalg::LinalgOp opToPad, OpOperand *opOperand,
    ArrayRef<int64_t> paddingDimensions, ArrayRef<Attribute> paddingValues,
    ArrayRef<bool> packPaddings) {
  AffineMap indexingMap = opToPad.getMatchingIndexingMap(opOperand);
  ArrayRef<int64_t> shape = opToPad.getShape(opOperand);

  // Collect the shape dimension that are a function of the `paddingDimensions`.
  llvm::SmallDenseSet<int64_t> shapeDimsToPad;
  for (int64_t dim : paddingDimensions)
    for (const auto &en : enumerate(indexingMap.getResults()))
      if (en.value().isFunctionOfDim(dim))
        shapeDimsToPad.insert(en.index());

  // Return the unpadded operand if padding to a static shape is not needed and
  // if the nofold flag is not set.
  bool nofold = opOperand->getOperandNumber() < packPaddings.size()
                    ? packPaddings[opOperand->getOperandNumber()]
                    : false;
  bool hasStaticShape = llvm::none_of(shapeDimsToPad, [&](int64_t dim) {
    return ShapedType::isDynamic(shape[dim]);
  });
  if (!nofold && hasStaticShape)
    return opOperand->get();

  // Fail if `paddingValues` specifies no padding value.
  if (opOperand->getOperandNumber() >= paddingValues.size())
    return failure();
  Attribute paddingAttr = paddingValues[opOperand->getOperandNumber()];
  Type paddingType = b.getType<NoneType>();
  if (auto typedAttr = paddingAttr.dyn_cast<TypedAttr>())
    paddingType = typedAttr.getType();
  Value paddingValue =
      b.create<arith::ConstantOp>(opToPad.getLoc(), paddingType, paddingAttr);

  // Follow the use-def chain if `currOpOperand` is defined by a LinalgOp.
  OpOperand *currOpOperand = opOperand;
  while (auto linalgOp = currOpOperand->get().getDefiningOp<LinalgOp>()) {
    OpResult result = currOpOperand->get().cast<OpResult>();
    currOpOperand = linalgOp.getOutputOperand(result.getResultNumber());
  }

  // Fail if `currOpOperand` is not defined by an ExtractSliceOp.
  auto sliceOp = currOpOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return failure();

  // Compute the dropped dimensions if `sliceOp` is ranke-reducing.
  llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();
  OffsetSizeAndStrideOpInterface shapedOp = sliceOp;

  // Upper bound the `sliceOp` sizes to obtain a static bounding box.
  SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
  int64_t shapeIdx = 0;
  for (const auto &en : enumerate(shapedOp.getMixedSizes())) {
    // Skip dropped dimensions.
    if (droppedDims.test(en.index()))
      continue;
    // Skip dimensions that do not require padding.
    if (!shapeDimsToPad.contains(shapeIdx)) {
      shapeIdx++;
      continue;
    }
    // If the size is an attribute add it directly to `paddedShape`.
    if (en.value().is<Attribute>()) {
      paddedShape[shapeIdx++] =
          en.value().get<Attribute>().dyn_cast<IntegerAttr>().getInt();
      continue;
    }
    // Otherwise, try to compute a constant upper bound for the size value.
    FailureOr<int64_t> upperBound =
        getConstantUpperBoundForIndex(en.value().get<Value>());
    if (failed(upperBound)) {
      LLVM_DEBUG(DBGS() << "No constant bounding box can be found for padding");
      return failure();
    }
    paddedShape[shapeIdx++] = *upperBound;
  }
  assert(shapeIdx == static_cast<int64_t>(shape.size()) &&
         "expect the dynamic and static ranks to match");

  // Pad the operand to the bounding box defined by `paddedShape`.
  auto paddedTensorType = RankedTensorType::get(
      paddedShape, getElementTypeOrSelf(opOperand->get()));
  return makeComposedPadHighOp(b, opToPad->getLoc(), paddedTensorType,
                               opOperand->get(), paddingValue, nofold);
}

FailureOr<SmallVector<Value>>
linalg::rewriteAsPaddedOp(OpBuilder &b, LinalgOp opToPad,
                          ArrayRef<int64_t> paddingDimensions,
                          ArrayRef<Attribute> paddingValues,
                          ArrayRef<bool> packPaddings, LinalgOp &paddedOp) {
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
    FailureOr<Value> paddedOperand = padOperandToSmallestStaticBoundingBox(
        b, opToPad, opOperand, paddingDimensions, paddingValues, packPaddings);
    // Exit if `paddingDimensions` cannot be bounded statically.
    if (failed(paddedOperand))
      return failure();
    newOperands.push_back(*paddedOperand);
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

/// Peel and canonicalize 'loops'.
void mlir::linalg::peelLoops(RewriterBase &rewriter,
                             ArrayRef<scf::ForOp> loops) {
  for (auto loopOp : loops) {
    SmallVector<Value, 4> loopResults;
    loopResults = peelLoop(rewriter, loopOp);
  }
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
    loopResults = peelLoop(rewriter, loopOp);

    // The result of the loop nest may change with peeling.
    if (res.tensorResults.size() == loopOp->getNumResults() &&
        std::equal(res.tensorResults.begin(), res.tensorResults.end(),
                   loopOp->getResults().begin()))
      res.tensorResults = loopResults;
  }
}

FailureOr<TiledLinalgOp>
mlir::linalg::tileWithLinalgTilingOptions(RewriterBase &rewriter, LinalgOp op,
                                          const LinalgTilingOptions &options) {
  FailureOr<TiledLinalgOp> res = tileLinalgOp(rewriter, op, options);
  if (failed(res))
    return failure();

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
    MLIRContext *context, LinalgPaddingOptions options, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      options(std::move(options)) {}

FailureOr<LinalgOp>
mlir::linalg::LinalgPaddingPattern::returningMatchAndRewrite(
    LinalgOp linalgOp, PatternRewriter &rewriter) const {
  if (!linalgOp.hasTensorSemantics())
    return failure();

  // Pad the operation.
  LinalgOp paddedOp;
  FailureOr<SmallVector<Value>> newResults =
      rewriteAsPaddedOp(rewriter, linalgOp, options.paddingDimensions,
                        options.paddingValues, options.packPaddings, paddedOp);
  if (failed(newResults))
    return failure();

  // Hoist the padding.
  for (const auto &en : enumerate(options.hoistPaddings)) {
    if (static_cast<int64_t>(en.index()) >= paddedOp.getNumInputsAndOutputs())
      break;
    OpOperand *opOperand = &paddedOp->getOpOperand(en.index());
    auto padOp = opOperand->get().getDefiningOp<tensor::PadOp>();
    if (!padOp || en.value() == 0)
      continue;

    // Fail hoisting if the operand shape is not fully static.
    if (llvm::any_of(paddedOp.getShape(opOperand), ShapedType::isDynamic))
      return failure();

    tensor::PadOp hoistedOp;
    SmallVector<GenericOp> transposeOps;
    SmallVector<int64_t> transposeVector =
        en.index() < options.transposePaddings.size()
            ? options.transposePaddings[en.index()]
            : SmallVector<int64_t>{};

    FailureOr<Value> newResult = hoistPaddingOnTensors(
        padOp, en.value(), transposeVector, hoistedOp, transposeOps);
    if (failed(newResult))
      continue;
    rewriter.replaceOp(padOp, *newResult);
  }

  // Replace the original operation to pad.
  rewriter.replaceOp(linalgOp, *newResults);

  return paddedOp;
}

LogicalResult mlir::linalg::CopyVectorizationPattern::matchAndRewrite(
    memref::CopyOp copyOp, PatternRewriter &rewriter) const {
  return vectorizeCopy(rewriter, copyOp);
}

static SmallVector<StringRef> getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<StringRef>(nParallelLoops, getParallelIteratorTypeName());
}

/// Rewrite a tensor::PadOp into a sequence of EmptyOp, FillOp (to
/// initialize with pad_val) and GenericOp (to copy contents).
LogicalResult
PadOpTransformationPattern::matchAndRewrite(tensor::PadOp padOp,
                                            PatternRewriter &rewriter) const {

  auto inputShapedType = padOp.getSource().getType().cast<ShapedType>();
  auto resultShapedType = padOp.getResult().getType().cast<ShapedType>();

  // Bail on non-static shapes.
  if (!inputShapedType.hasStaticShape())
    return failure();
  if (!resultShapedType.hasStaticShape())
    return failure();

  // Only support padding with a constant for now, i.e. either:
  //   1. A BBarg from a different block.
  //   2. A value defined outside of the current block.
  Block &block = padOp.getRegion().front();
  auto yieldOp = cast<tensor::YieldOp>(block.getTerminator());
  Value padValue = yieldOp.getValue();
  Operation *definingOp = padValue.getDefiningOp();
  if (definingOp && definingOp->getBlock() == &block)
    return failure();
  if (!definingOp && padValue.cast<BlockArgument>().getOwner() == &block)
    return failure();

  // Create tensor with the padded shape
  Location loc = padOp.getLoc();
  SmallVector<Value> indices(resultShapedType.getRank(),
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, resultShapedType.getShape(), resultShapedType.getElementType());

  // Initialize tensor with the pad value
  Value tmpTensor = rewriter
                        .create<linalg::FillOp>(loc, ValueRange{padValue},
                                                ValueRange{emptyTensor})
                        .result();

  // Copy original contents into new tensor
  // Uses linalg.generic, but could be done with tensor.insert_slice
  SmallVector<AffineExpr, 4> outputExprs;
  for (unsigned i = 0; i < resultShapedType.getRank(); ++i) {
    outputExprs.push_back(getAffineDimExpr(i, rewriter.getContext()) +
                          padOp.getStaticLow()[i].cast<IntegerAttr>().getInt());
  }

  SmallVector<AffineMap, 2> transferMaps = {
      rewriter.getMultiDimIdentityMap(inputShapedType.getRank()),
      AffineMap::get(resultShapedType.getRank(),
                     /*symbolCount=*/0, outputExprs, rewriter.getContext())};

  rewriter.replaceOpWithNewOp<linalg::GenericOp>(
      padOp, resultShapedType, padOp.getSource(), tmpTensor, transferMaps,
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
  padOp.getRegion().cloneInto(&generateOp.getRegion(), bvm);
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
  // Compute size of EmptyOp. Any combination of static/dynamic is supported.
  SmallVector<Value> dynSizes;
  SmallVector<int64_t> staticSizes;
  for (unsigned dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim)) {
      auto srcSize = rewriter.createOrFold<tensor::DimOp>(
          padOp.getLoc(), padOp.getSource(), dim);
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
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      padOp.getLoc(), staticSizes, resultType.getElementType(), dynSizes);
  Value fill = createFillOrGenerateOp(rewriter, padOp, emptyTensor, dynSizes);

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
          padOp.getLoc(), padOp.getSource(), dim));
    } else {
      srcSizes.push_back(rewriter.getIndexAttr(sourceType.getDimSize(dim)));
    }
  }
  // Strides of InsertSliceOp are all 1.
  SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                    rewriter.getIndexAttr(1));
  rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
      padOp, padOp.getSource(), fill, padOp.getMixedLowPad(), srcSizes,
      strides);

  return success();
}

LogicalResult ExtractSliceOfPadTensorSwapPattern::matchAndRewrite(
    tensor::ExtractSliceOp sliceOp, PatternRewriter &rewriter) const {
  if (!sliceOp.hasUnitStride())
    return failure();

  auto padOp = sliceOp.getSource().getDefiningOp<tensor::PadOp>();
  if (!padOp)
    return failure();

  bool zeroSliceGuard = true;
  if (controlFn) {
    if (Optional<bool> control = controlFn(sliceOp))
      zeroSliceGuard = *control;
    else
      return failure();
  }

  Operation *tiledPadOp =
      tensor::bubbleUpPadSlice(rewriter, padOp, sliceOp.getMixedOffsets(),
                               sliceOp.getMixedSizes(), zeroSliceGuard);
  // All shapes are static and the data source is actually used. Rewrite into
  // pad(extract_slice(x)).
  rewriter.replaceOp(sliceOp, tiledPadOp->getResults());
  return success();
}

// The following are patterns for downscaling convolution ops with size-1
// window dimensions.
//
// Note that we'd eventually want to write such transformations in a generic
// way, e.g., converting to linalg.generic, removing the size-1 dimensions,
// and then turning back to named ops. But for now it's fine to have a few
// patterns matching special ops to get started.

template <typename Conv2DOp, typename Conv1DOp>
FailureOr<Conv1DOp> DownscaleSizeOneWindowed2DConvolution<Conv2DOp, Conv1DOp>::
    returningMatchAndRewrite(Conv2DOp convOp, PatternRewriter &rewriter) const {
  if (convOp.hasBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
  auto outputType = output.getType().dyn_cast<RankedTensorType>();

  auto kernelShape = kernelType.getShape();
  auto outputShape = outputType.getShape();

  // Get domain indices based on conv2D layout.
  int khIndex, kwIndex, ohIndex, owIndex;

  TypeSwitch<Operation *>(convOp)
      .Case([&](linalg::Conv2DNhwcHwcfOp op) {
        khIndex = 0;
        kwIndex = 1;
        ohIndex = 1;
        owIndex = 2;
      })
      .Case([&](linalg::Conv2DNchwFchwOp op) {
        khIndex = 2;
        kwIndex = 3;
        ohIndex = 2;
        owIndex = 3;
      })
      .Default([&](Operation *op) {
        llvm_unreachable("unexpected conv2d operation.");
      });

  // Only handle the case where at least one of the window dimensions is
  // of size 1. Other cases can rely on tiling to reduce to such cases.
  int64_t khSize = kernelShape[khIndex], kwSize = kernelShape[kwIndex];
  int64_t ohSize = outputShape[ohIndex], owSize = outputShape[owIndex];
  bool removeH = (khSize == 1 && ohSize == 1);
  bool removeW = (kwSize == 1 && owSize == 1);
  if (!removeH && !removeW)
    return failure();

  // Get new shapes and types for all operands by removing the size-1
  // dimension.
  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType newInputType =
      RTTBuilder(inputType).dropDim((removeH ? ohIndex : owIndex));
  RankedTensorType newKernelType =
      RTTBuilder(kernelType).dropDim((removeH ? khIndex : kwIndex));
  RankedTensorType newOutputType =
      RTTBuilder(outputType).dropDim((removeH ? ohIndex : owIndex));

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
  auto strides =
      llvm::to_vector<4>(convOp.getStrides().template getValues<int64_t>());
  strides.erase(strides.begin() + (removeH ? 0 : 1));
  auto stridesAttr = rewriter.getI64VectorAttr(strides);

  auto dilations =
      llvm::to_vector<4>(convOp.getDilations().template getValues<int64_t>());
  dilations.erase(dilations.begin() + (removeH ? 0 : 1));
  auto dilationsAttr = rewriter.getI64VectorAttr(dilations);

  auto conv1DOp = rewriter.create<Conv1DOp>(
      loc, newOutputType, ValueRange{newInput, newKernel},
      ValueRange{newOutput}, stridesAttr, dilationsAttr);

  // Insert back.
  Value inserted = tensor::createCanonicalRankReducingInsertSliceOp(
      rewriter, loc, conv1DOp.getResult(0), output);
  rewriter.replaceOp(convOp, inserted);

  return conv1DOp;
}

template struct linalg::DownscaleSizeOneWindowed2DConvolution<Conv2DNhwcHwcfOp,
                                                              Conv1DNwcWcfOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<Conv2DNchwFchwOp,
                                                              Conv1DNcwFcwOp>;

FailureOr<DepthwiseConv1DNwcWcOp>
DownscaleDepthwiseConv2DNhwcHwcOp::returningMatchAndRewrite(
    DepthwiseConv2DNhwcHwcOp convOp, PatternRewriter &rewriter) const {
  if (convOp.hasBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

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
  auto strides = llvm::to_vector<4>(convOp.getStrides().getValues<int64_t>());
  strides.erase(strides.begin() + (removeH ? 0 : 1));
  auto stridesAttr = rewriter.getI64VectorAttr(strides);

  auto dilations =
      llvm::to_vector<4>(convOp.getDilations().getValues<int64_t>());
  dilations.erase(dilations.begin() + (removeH ? 0 : 1));
  auto dilationsAttr = rewriter.getI64VectorAttr(dilations);

  auto conv1DOp = rewriter.create<DepthwiseConv1DNwcWcOp>(
      loc, newOutputType, ValueRange{newInput, newKernel},
      ValueRange{newOutput}, stridesAttr, dilationsAttr);

  // Insert back.
  Value inserted = tensor::createCanonicalRankReducingInsertSliceOp(
      rewriter, loc, conv1DOp.getResult(0), output);
  rewriter.replaceOp(convOp, inserted);

  return conv1DOp;
}

void linalg::populateDecomposeConvolutionPatterns(RewritePatternSet &patterns,
                                                  PatternBenefit benefit) {
  patterns.add<DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNhwcHwcfOp,
                                                     Conv1DNwcWcfOp>,
               DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNchwFchwOp,
                                                     Conv1DNcwFcwOp>,
               DownscaleDepthwiseConv2DNhwcHwcOp>(patterns.getContext(),
                                                  benefit);
}
