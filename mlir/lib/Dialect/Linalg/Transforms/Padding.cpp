//===- Padding.cpp - Padding of Linalg ops --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#define DEBUG_TYPE "linalg-padding"

using namespace mlir;
using namespace mlir::linalg;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

/// Pad the `opOperand` in the `paddingDimensions` using the padding value and
/// the nofold flag found in `paddingValues` and `packPaddings`, respectively.
///
/// Exit early and return the `opOperand` value if it already has the requested
/// shape. I.e.:
/// - static shape
/// - nofold is not set
/// - dim sizes are multiples of `padToMultipleOf`
///
/// Otherwise, try to pad the shape dimensions that match the iterator
/// dimensions `paddingDimensions` and return the tensor::PadOp result if
/// padding succeeds or failure otherwise.
static FailureOr<Value> padOperandToSmallestStaticBoundingBox(
    RewriterBase &rewriter, linalg::LinalgOp opToPad, OpOperand *opOperand,
    ArrayRef<int64_t> paddingDimensions, ArrayRef<int64_t> padToMultipleOf,
    ArrayRef<Attribute> paddingValues, ArrayRef<bool> packPaddings) {
  assert(padToMultipleOf.size() == paddingDimensions.size() &&
         "invalid number of elements in padToMultipleOf");

  AffineMap indexingMap = opToPad.getMatchingIndexingMap(opOperand);
  ArrayRef<int64_t> shape = opToPad.getShape(opOperand);

  // Collect the shape dimensions that are a function of `paddingDimensions`,
  // along with the multiple that they should be padded to ("1" if none).
  bool alreadyHasRequestedShape = true;
  DenseMap<int64_t, int64_t> shapeDimToMultiple;
  for (const auto &dimEn : enumerate(paddingDimensions)) {
    for (const auto &en : enumerate(indexingMap.getResults())) {
      if (en.value().isFunctionOfDim(dimEn.value())) {
        int64_t dimSize = shape[en.index()];
        shapeDimToMultiple[en.index()] = padToMultipleOf[dimEn.index()];
        if (ShapedType::isDynamic(dimSize)) {
          alreadyHasRequestedShape = false;
        } else if (dimSize % shapeDimToMultiple[en.index()] != 0) {
          alreadyHasRequestedShape = false;
        }
      }
    }
  }

  // Return the unpadded operand if padding to a static shape is not needed and
  // if the nofold flag is not set.
  bool nofold = opOperand->getOperandNumber() < packPaddings.size()
                    ? packPaddings[opOperand->getOperandNumber()]
                    : false;
  if (!nofold && alreadyHasRequestedShape)
    return opOperand->get();

  // Fail if `paddingValues` specifies no padding value.
  if (opOperand->getOperandNumber() >= paddingValues.size()) {
    return rewriter.notifyMatchFailure(opToPad, "--no padding value specified");
  }
  Attribute paddingAttr = paddingValues[opOperand->getOperandNumber()];
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      opToPad.getLoc(), cast<TypedAttr>(paddingAttr));

  // Helper function to round a number up to a given multiple.
  auto ceil = [](int64_t val, int64_t multiple) {
    return ((val + multiple - 1) / multiple) * multiple;
  };

  // Upper bound the sizes to obtain a static bounding box.
  SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
  for (int64_t i = 0, e = shape.size(); i < e; ++i) {
    LLVM_DEBUG(DBGS() << "--compute padded size for dim " << i << "\n");
    // Skip dimensions that do not require padding.
    if (!shapeDimToMultiple.contains(i)) {
      LLVM_DEBUG(DBGS() << "----dim does not require padding, SKIP\n");
      continue;
    }
    // Otherwise, try to compute a constant upper bound for the size value.
    FailureOr<int64_t> upperBound =
        ValueBoundsConstraintSet::computeConstantBound(
            presburger::BoundType::UB, opOperand->get(),
            /*dim=*/i, /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(upperBound)) {
      LLVM_DEBUG(DBGS() << "----count not compute a bounding box for padding");
      return rewriter.notifyMatchFailure(
          opToPad, "count not compute a bounding box for padding");
    }
    paddedShape[i] = ceil(*upperBound, shapeDimToMultiple[i]);
    LLVM_DEBUG(DBGS() << "----new dim size: " << paddedShape[i] << "\n");
  }

  // Pad the operand to the bounding box defined by `paddedShape`.
  auto paddedTensorType = RankedTensorType::get(
      paddedShape, getElementTypeOrSelf(opOperand->get()));
  LLVM_DEBUG(DBGS() << "--SUCCESS, makeComposedPadHighOp with type: "
                    << paddedTensorType);
  return makeComposedPadHighOp(rewriter, opToPad->getLoc(), paddedTensorType,
                               opOperand->get(), paddingValue, nofold);
}

FailureOr<SmallVector<Value>>
linalg::rewriteAsPaddedOp(RewriterBase &rewriter, LinalgOp opToPad,
                          ArrayRef<int64_t> paddingDimensions,
                          ArrayRef<int64_t> padToMultipleOf,
                          ArrayRef<Attribute> paddingValues,
                          ArrayRef<bool> packPaddings, LinalgOp &paddedOp) {
  LLVM_DEBUG(DBGS() << "Start rewriteAsPaddedOp : " << opToPad << "\n");
  Location loc = opToPad->getLoc();

  // TODO: there are cases where we may still want to pad to larger sizes.
  if (!opToPad.hasTensorSemantics())
    return rewriter.notifyMatchFailure(opToPad,
                                       "expected operation on tensors");

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(opToPad);

  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad->getNumOperands());
  for (OpOperand &opOperand : opToPad->getOpOperands()) {
    FailureOr<Value> paddedOperand = padOperandToSmallestStaticBoundingBox(
        rewriter, opToPad, &opOperand, paddingDimensions, padToMultipleOf,
        paddingValues, packPaddings);
    // Exit if `paddingDimensions` cannot be bounded statically.
    if (failed(paddedOperand)) {
      LLVM_DEBUG(DBGS() << "--operand cannot be bound statically : "
                        << opOperand.get() << " -> FAIL\n");
      return rewriter.notifyMatchFailure(opToPad,
                                         "operand cannot be bound statically");
    }
    newOperands.push_back(*paddedOperand);
  }

  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(reifyResultShapes(rewriter, opToPad, reifiedResultShapes))) {
    LLVM_DEBUG(DBGS() << "--failed to reify result shapes -> FAIL\n");
    return rewriter.notifyMatchFailure(opToPad,
                                       "failed to reify result shapes");
  }
  assert(reifiedResultShapes.size() == opToPad->getNumResults() &&
         "expected same number of results");

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumDpsInits()).getTypes();
  // clone **should** properly notify the rewriter.
  paddedOp = clone(rewriter, opToPad, resultTensorTypes, newOperands);
  LLVM_DEBUG(DBGS() << "--cloned padded op: " << paddedOp << "\n");

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  SmallVector<Value> paddedSubtensorResults;
  paddedSubtensorResults.reserve(opToPad->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = cast<RankedTensorType>(paddedResult.getType()).getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubtensorResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, reifiedResultShapes[resultNumber],
        strides));
  }
  return paddedSubtensorResults;
}

FailureOr<LinalgOp>
mlir::linalg::padAndHoistLinalgOp(RewriterBase &rewriter, LinalgOp linalgOp,
                                  LinalgPaddingOptions options) {
  if (!linalgOp.hasTensorSemantics())
    return rewriter.notifyMatchFailure(
        linalgOp, "only applies to Linalg ops with tensor semantics");

  // Pad the operation.
  LinalgOp paddedOp;
  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  if (options.padToMultipleOf.has_value())
    padToMultipleOf.assign(options.padToMultipleOf->begin(),
                           options.padToMultipleOf->end());
  FailureOr<SmallVector<Value>> newResults = rewriteAsPaddedOp(
      rewriter, linalgOp, options.paddingDimensions, padToMultipleOf,
      options.paddingValues, options.packPaddings, paddedOp);
  if (failed(newResults))
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to rewrite as a padded op");

  // Hoist the padding.
  for (const auto &en : enumerate(options.hoistPaddings)) {
    if (static_cast<int64_t>(en.index()) >= paddedOp->getNumOperands())
      break;
    OpOperand &opOperand = paddedOp->getOpOperand(en.index());
    auto padOp = opOperand.get().getDefiningOp<tensor::PadOp>();
    if (!padOp || en.value() == 0) {
      (void)rewriter.notifyMatchFailure(linalgOp, "not a tensor.pad -- skip");
      continue;
    }

    // Fail hoisting if the operand shape is not fully static.
    if (llvm::any_of(paddedOp.getShape(&opOperand), ShapedType::isDynamic)) {
      (void)rewriter.notifyMatchFailure(linalgOp,
                                        "non static padding shape -- skip");
      continue;
    }

    tensor::PadOp hoistedOp;
    SmallVector<GenericOp> transposeOps;
    SmallVector<int64_t> transposeVector =
        en.index() < options.transposePaddings.size()
            ? options.transposePaddings[en.index()]
            : SmallVector<int64_t>{};

    FailureOr<Value> newResult = hoistPaddingOnTensors(
        padOp, en.value(), transposeVector, hoistedOp, transposeOps);
    if (failed(newResult)) {
      (void)rewriter.notifyMatchFailure(linalgOp,
                                        "failed to apply hoistPadding");
      continue;
    }
    rewriter.replaceOp(padOp, *newResult);
  }

  // Replace the original operation to pad.
  rewriter.replaceOp(linalgOp, *newResults);

  return paddedOp;
}
