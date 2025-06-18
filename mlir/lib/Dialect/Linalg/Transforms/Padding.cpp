//===- Padding.cpp - Padding of Linalg ops --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#define DEBUG_TYPE "linalg-padding"

using namespace mlir;
using namespace mlir::linalg;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

/// Compute the padded shape of the given operand, combining multiples from all
/// contributing dimensions using affine.apply operations.
static void computePaddedShape(RewriterBase &rewriter, linalg::LinalgOp opToPad,
                               OpOperand *opOperand,
                               ArrayRef<OpFoldResult> loopRanges,
                               const LinalgPaddingOptions &options,
                               SmallVector<OpFoldResult> &paddedShape,
                               bool &alreadyHasRequestedShape) {
  Location loc = opToPad.getLoc();
  AffineMap indexingMap = opToPad.getMatchingIndexingMap(opOperand);
  auto tensorType = cast<RankedTensorType>(opOperand->get().getType());
  paddedShape.resize_for_overwrite(tensorType.getRank());

  // For each dimension in the operand's shape
  for (const auto &enResults : enumerate(indexingMap.getResults())) {
    int64_t resultIndex = enResults.index();
    AffineMap partialIndexingMap = indexingMap.getSubMap(
        ArrayRef<unsigned>{static_cast<unsigned>(resultIndex)});

    LLVM_DEBUG(DBGS() << "----resultIndex: " << resultIndex
                      << " with partialIndexingMap: " << partialIndexingMap
                      << "\n");

    // TODO: mult could be a dynamic value, then we'd want a symbol.
    // For now at least the API forces us to get a staic value.
    // Note: if mult is 1 just let it go and compose properly; fill missing
    // entries with 1s.
    llvm::SmallDenseMap<int64_t, int64_t> dimsToMult;
    for (const auto &en : enumerate(options.paddingDimensions)) {
      auto mult = (*options.padToMultipleOf)[en.index()];
      dimsToMult[en.value()] = mult;
    }
    for (int64_t idx = 0, e = opToPad.getNumLoops(); idx != e; ++idx) {
      if (dimsToMult.find(idx) != dimsToMult.end())
        continue;
      dimsToMult[idx] = 1;
    }

    // Find all padding dimensions that contribute to this operand dimension
    // and compute the padded term contribution to the final padded shape.
    SmallVector<OpFoldResult> terms;
    for (const auto &[paddingDim, mult] : dimsToMult) {
      LLVM_DEBUG(DBGS() << "------apply padding of dim: " << paddingDim
                        << " to mult: " << mult << "\n");
      if (enResults.value().isFunctionOfDim(paddingDim)) {
        // Project non-'paddingDim' dimensions and compress the result.
        llvm::SmallBitVector projectedDims(partialIndexingMap.getNumDims(),
                                           true);
        projectedDims.flip(paddingDim);
        AffineMap projectedMap =
            mlir::projectDims(partialIndexingMap, projectedDims,
                              /*compressDims=*/true);

        // Compose with ceil(mult) and construct the term.
        AffineExpr d0;
        bindDims(rewriter.getContext(), d0);
        AffineMap ceilMap = AffineMap::get(1, 0, d0.ceilDiv(mult) * mult);
        AffineMap composedMap = projectedMap.compose(ceilMap);
        OpFoldResult paddingDimOfr = affine::makeComposedFoldedAffineApply(
            rewriter, loc, composedMap, loopRanges[paddingDim]);
        terms.push_back(paddingDimOfr);
        LLVM_DEBUG(DBGS() << "------new term: " << terms.back() << "\n");
      }
    }

    if (!terms.empty()) {
      SmallVector<AffineExpr> dims(terms.size());
      bindDimsList(rewriter.getContext(), MutableArrayRef{dims});
      AffineExpr sumExpr = dims.front();
      for (unsigned i = 1; i < dims.size(); ++i)
        sumExpr = sumExpr + dims[i];
      OpFoldResult paddedDimOfr =
          affine::makeComposedFoldedAffineApply(rewriter, loc, sumExpr, terms);
      paddedShape[resultIndex] = paddedDimOfr;
    } else {
      paddedShape[resultIndex] =
          createFoldedDimOp(rewriter, loc, opOperand->get(), resultIndex);
    }

    bool dimNeedsPad = false;
    auto maybeVal = getConstantIntValue(paddedShape[resultIndex]);
    dimNeedsPad = !maybeVal.has_value() ||
                  (*maybeVal != tensorType.getShape()[resultIndex]);
    alreadyHasRequestedShape &= !dimNeedsPad;

    LLVM_DEBUG(DBGS() << "----padded shape: " << paddedShape[resultIndex]
                      << " vs existing shape: "
                      << tensorType.getShape()[resultIndex]
                      << " ---> needs pad: " << dimNeedsPad << "\n");
  }
}

/// Pad the `opOperand` in the "paddingDimensions" using the padding value and
/// the nofold flag found in "paddingValues" and "nofoldFlags", respectively.
///
/// Exit early and return the `opOperand` value if it already has the requested
/// shape. i.e.:
/// - static shape
/// - nofold is not set
/// - dim sizes are multiples of "padToMultipleOf"
///
/// Otherwise, try to pad the shape dimensions that match the iterator
/// dimensions "paddingDimensions" and return the tensor::PadOp result if
/// padding succeeds or failure otherwise.
static Value padOperand(RewriterBase &rewriter, linalg::LinalgOp opToPad,
                        OpOperand *opOperand, ArrayRef<OpFoldResult> loopRanges,
                        const LinalgPaddingOptions &options) {
  assert(
      (!options.padToMultipleOf.has_value() ||
       options.padToMultipleOf->size() == options.paddingDimensions.size()) &&
      "invalid number of elements in padToMultipleOf");

  // Compute padded shape.
  SmallVector<OpFoldResult> paddedShape;
  bool alreadyHasRequestedShape = false;
  computePaddedShape(rewriter, opToPad, opOperand, loopRanges, options,
                     paddedShape, alreadyHasRequestedShape);

  // Return the unpadded operand if padding to a static shape is not needed
  // and if the nofold flag is not set.
  bool nofold = opOperand->getOperandNumber() < options.nofoldFlags.size()
                    ? bool(options.nofoldFlags[opOperand->getOperandNumber()])
                    : false;
  if (!nofold && alreadyHasRequestedShape)
    return opOperand->get();

  // Expect proper `paddingValues`.
  // TODO: we may want to allow garbage padding in the future, in which case
  // we would just not assert.
  assert(opOperand->getOperandNumber() < options.paddingValues.size() &&
         "--no padding value specified");

  Attribute paddingAttr = options.paddingValues[opOperand->getOperandNumber()];

  Value paddingValue;
  if (auto complexTy = dyn_cast<ComplexType>(
          getElementTypeOrSelf(opOperand->get().getType()))) {
    auto complexAttr = cast<ArrayAttr>(paddingAttr);
    paddingValue = rewriter.create<complex::ConstantOp>(opToPad.getLoc(),
                                                        complexTy, complexAttr);
  } else {
    paddingValue = rewriter.create<arith::ConstantOp>(
        opToPad.getLoc(), cast<TypedAttr>(paddingAttr));
  }

  // Pad the operand to the bounding box defined by `paddedShape`.
  SmallVector<int64_t> tensorShape;
  SmallVector<Value> dynDims;
  for (OpFoldResult ofr : paddedShape) {
    std::optional<int64_t> cst = getConstantIntValue(ofr);
    tensorShape.push_back(cst.has_value() ? *cst : ShapedType::kDynamic);
    if (!cst.has_value())
      dynDims.push_back(ofr.dyn_cast<Value>());
  }
  auto paddedTensorType = RankedTensorType::get(
      tensorShape, getElementTypeOrSelf(opOperand->get()));
  LLVM_DEBUG(DBGS() << "--SUCCESS, makeComposedPadHighOp with type: "
                    << paddedTensorType);
  return makeComposedPadHighOp(rewriter, opToPad->getLoc(), paddedTensorType,
                               opOperand->get(), paddingValue, nofold, dynDims);
}

LogicalResult
linalg::rewriteAsPaddedOp(RewriterBase &rewriter, LinalgOp opToPad,
                          const LinalgPaddingOptions &constOptions,
                          LinalgOp &paddedOp, SmallVector<Value> &replacements,
                          SmallVector<tensor::PadOp> &padOps) {
  LLVM_DEBUG(DBGS() << "Start rewriteAsPaddedOp : " << opToPad << "\n");
  Location loc = opToPad->getLoc();

  LinalgPaddingOptions options(constOptions);
  // Allow inference of pad values if they are not explicitly specified.
  // TODO: be mindful about the value depending on the actual operation.
  if (options.paddingValues.empty()) {
    SmallVector<Type> types(opToPad->getOperandTypes());
    llvm::append_range(types, opToPad->getResultTypes());
    for (Type t : types) {
      options.paddingValues.push_back(
          rewriter.getZeroAttr(getElementTypeOrSelf(t)));
    }
  }

  // TODO: there are cases where we may still want to pad to larger sizes.
  if (!opToPad.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(opToPad,
                                       "expected operation on tensors");

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(opToPad);

  SmallVector<OpFoldResult> allShapes =
      opToPad.createFlatListOfOperandDims(rewriter, loc);
  AffineMap shapesToLoops = opToPad.getShapesToLoopsMap();
  SmallVector<OpFoldResult> loopRanges =
      affine::makeComposedFoldedMultiResultAffineApply(
          rewriter, opToPad.getLoc(), shapesToLoops, allShapes);

  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad->getNumOperands());
  for (OpOperand &opOperand : opToPad->getOpOperands()) {
    LLVM_DEBUG(DBGS() << "--start padding operand: " << opOperand.get()
                      << "\n");
    Value paddedOperand =
        padOperand(rewriter, opToPad, &opOperand, loopRanges, options);
    LLVM_DEBUG(DBGS() << "--done padding operand: " << paddedOperand << "\n");
    newOperands.push_back(paddedOperand);
    if (auto padOp = paddedOperand.getDefiningOp<tensor::PadOp>())
      padOps.push_back(padOp);
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

  if (options.copyBackOp == LinalgPaddingOptions::CopyBackOp::None) {
    replacements = std::move(paddedSubtensorResults);
    return success();
  }

  // Copy back unpadded results to the original destination (i.e., inits of the
  // linalg op), so that the destination buffer of the computation does not
  // change. If the padding folds away, this will materialize as a memcpy
  // between two identical buffers, which will then also fold away.
  assert(static_cast<int64_t>(paddedSubtensorResults.size()) ==
             opToPad.getNumDpsInits() &&
         "expected matching number of results");
  for (auto it :
       llvm::zip(paddedSubtensorResults, opToPad.getDpsInitsMutable())) {
    if (options.copyBackOp == LinalgPaddingOptions::CopyBackOp::LinalgCopy) {
      replacements.push_back(rewriter
                                 .create<linalg::CopyOp>(loc, std::get<0>(it),
                                                         std::get<1>(it).get())
                                 .getResult(0));
    } else if (options.copyBackOp ==
               LinalgPaddingOptions::CopyBackOp::
                   BufferizationMaterializeInDestination) {
      replacements.push_back(
          rewriter
              .create<bufferization::MaterializeInDestinationOp>(
                  loc, std::get<0>(it), std::get<1>(it).get())
              ->getResult(0));
    } else {
      llvm_unreachable("unsupported copy back op");
    }
  }
  return success();
}

FailureOr<LinalgOp>
mlir::linalg::padAndHoistLinalgOp(RewriterBase &rewriter, LinalgOp linalgOp,
                                  const LinalgPaddingOptions &options) {
  assert(options.copyBackOp == LinalgPaddingOptions::CopyBackOp::None &&
         "invalid options");

  if (!linalgOp.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(
        linalgOp, "only applies to Linalg ops with tensor semantics");

  // Pad the operation.
  LinalgOp paddedOp;
  SmallVector<Value> newResults;
  SmallVector<tensor::PadOp> padOps;
  if (failed(rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                               newResults, padOps)))
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
    SmallVector<TransposeOp> transposeOps;
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
  rewriter.replaceOp(linalgOp, newResults);

  return paddedOp;
}
