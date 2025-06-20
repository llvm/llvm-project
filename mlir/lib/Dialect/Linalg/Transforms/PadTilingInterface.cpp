//===- PaddingTilingInterface.cpp - Padding of TilingInterface ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "pad-tiling-interface"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::tensor;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

/// Form a "full-rank" padding specification so that the application is easy.
static llvm::SmallDenseMap<int64_t, OpFoldResult>
getDimsToSize(Builder &b, ArrayRef<OpFoldResult> indexingSizes,
              const PadTilingInterfaceOptions &options) {
  llvm::SmallDenseMap<int64_t, OpFoldResult> dimsToSize;
  for (const auto &[paddingDim, paddingSize] :
       llvm::zip_equal(options.paddingDimensions, options.paddingSizes)) {
    dimsToSize[paddingDim] = paddingSize;
  }
  // Complete the padding specification to specify all dimensions.
  for (int64_t idx = 0, e = indexingSizes.size(); idx != e; ++idx) {
    if (dimsToSize.find(idx) != dimsToSize.end())
      continue;
    // If a dimension is not specified, either complete with:
    //   - 1 if we are padding to the next multiple of.
    //   - indexingSizes[idx] otherwise
    dimsToSize[idx] =
        options.padToMultipleOf ? b.getIndexAttr(1) : indexingSizes[idx];
  }
  for (int64_t idx = 0, e = indexingSizes.size(); idx != e; ++idx) {
    LLVM_DEBUG(DBGS() << "----idx: " << idx << " : " << dimsToSize[idx]
                      << "\n");
  }
  return dimsToSize;
}

/// Compute the padded shape of the given value `v` of `RankedTensorType` given
///   - `indexingSizes` a list of OpFoldResult.
///   - an `indexingMap` that encodes how the shape of varies with increases
///     in `indexingSizes`.
/// The `indexingMap` encodes how the shape of varies with `indexingSizes`.
/// The `indexingMap` + `indexingSizes` encoding suits StructuredOps.
/// The implementaiton below iteratively combines increases from contributing
/// dimensions using affine.apply operations.
/// In the future, more general interfaces can be devised to encode similar
/// shape evolutions and map between an op and its operands.
SmallVector<OpFoldResult> linalg::computePaddedShape(
    RewriterBase &rewriter, TypedValue<RankedTensorType> v,
    AffineMap indexingMap, ArrayRef<OpFoldResult> indexingSizes,
    const PadTilingInterfaceOptions &options) {
  Location loc = v.getLoc();
  SmallVector<OpFoldResult> paddedShape;
  auto tensorType = cast<RankedTensorType>(v.getType());
  paddedShape.resize_for_overwrite(tensorType.getRank());
  assert(tensorType.getRank() == indexingMap.getNumResults() &&
         "expect the number of results of the affine map to match the tensor "
         "rank");

  // "Full-rank" padding specification.
  llvm::SmallDenseMap<int64_t, OpFoldResult> dimsToSize =
      getDimsToSize(rewriter, indexingSizes, options);

  // For each dimension in the operand's shape, iterate over indexingSizes and
  // add
  for (const auto &enResults : enumerate(indexingMap.getResults())) {
    int64_t resultIndex = enResults.index();
    AffineMap partialIndexingMap = indexingMap.getSubMap(
        ArrayRef<unsigned>{static_cast<unsigned>(resultIndex)});

    LLVM_DEBUG(DBGS() << "----resultIndex: " << resultIndex
                      << " with partialIndexingMap: " << partialIndexingMap
                      << "\n");

    // Find all padding dimensions that contribute to this operand dimension
    // and compute the padded term contribution to the final padded shape.
    SmallVector<OpFoldResult> terms;
    for (const auto &[paddingDim, paddingSize] : dimsToSize) {
      LLVM_DEBUG(DBGS() << "------try apply padding of dim: " << paddingDim
                        << " to: " << paddingSize << "\n");
      if (!enResults.value().isFunctionOfDim(paddingDim))
        continue;

      LLVM_DEBUG(DBGS() << "------apply padding of dim: " << paddingDim
                        << " to: " << paddingSize << "\n");

      // Project non-'paddingDim' dimensions and compress the result.
      llvm::SmallBitVector projectedDims(partialIndexingMap.getNumDims(), true);
      projectedDims.flip(paddingDim);
      AffineMap projectedMap =
          mlir::projectDims(partialIndexingMap, projectedDims,
                            /*compressDims=*/true);

      // If we are padding to the next multiple of, compose with ceil(sz) * sz.
      if (options.padToMultipleOf) {
        AffineExpr d0, s0;
        bindDims(rewriter.getContext(), d0);
        bindSymbols(rewriter.getContext(), s0);
        AffineMap ceilMap = AffineMap::get(1, 1, d0.ceilDiv(s0) * s0);
        AffineMap composedMap = projectedMap.compose(ceilMap);
        OpFoldResult paddingDimOfr = affine::makeComposedFoldedAffineApply(
            rewriter, loc, composedMap,
            {indexingSizes[paddingDim], paddingSize});
        terms.push_back(paddingDimOfr);
      } else {
        // Otherwise just set to paddingSize.
        OpFoldResult paddingDimOfr = affine::makeComposedFoldedAffineApply(
            rewriter, loc, projectedMap, paddingSize);
        terms.push_back(paddingDimOfr);
      }

      LLVM_DEBUG(DBGS() << "------new term: " << terms.back() << "\n");
    }

    // If there are no terms, just return the dim.
    if (terms.empty()) {
      paddedShape[resultIndex] =
          createFoldedDimOp(rewriter, loc, v, resultIndex);
      continue;
    }

    // Sum individual terms' contributions.
    SmallVector<AffineExpr> dims(terms.size());
    bindDimsList(rewriter.getContext(), MutableArrayRef{dims});
    AffineExpr sumExpr = dims.front();
    for (unsigned i = 1; i < dims.size(); ++i)
      sumExpr = sumExpr + dims[i];
    OpFoldResult paddedDimOfr =
        affine::makeComposedFoldedAffineApply(rewriter, loc, sumExpr, terms);
    paddedShape[resultIndex] = paddedDimOfr;
  }

  return paddedShape;
}

FailureOr<SmallVector<OpFoldResult>>
linalg::computeIndexingMapOpInterfacePaddedShape(
    RewriterBase &rewriter, OpOperand &operandToPad,
    ArrayRef<Range> iterationDomain, const PadTilingInterfaceOptions &options) {
  auto transferOp =
      llvm::dyn_cast<IndexingMapOpInterface>(operandToPad.getOwner());
  if (!transferOp)
    return failure();

  // clang-format off
  assert(llvm::all_of(iterationDomain, [&rewriter](Range r) {
    return r.offset == OpFoldResult(rewriter.getIndexAttr(0)) &&
    r.stride == OpFoldResult(rewriter.getIndexAttr(1));
  }) && "expected 0-offset 1-stride loop ranges");
  // clang-format on
  SmallVector<OpFoldResult> loopUpperBounds;
  loopUpperBounds.reserve(iterationDomain.size());
  for (const Range &range : iterationDomain)
    loopUpperBounds.push_back(range.size);

  AffineMap indexingMap = transferOp.getMatchingIndexingMap(&operandToPad);
  return computePaddedShape(
      rewriter, cast<TypedValue<RankedTensorType>>(operandToPad.get()),
      indexingMap, loopUpperBounds, options);
}

/// Pad a single operand to `paddedShape` using `paddingValueAttr` as padding
/// Value.
static Value padOperand(RewriterBase &rewriter, TilingInterface opToPad,
                        TypedValue<RankedTensorType> v,
                        ArrayRef<OpFoldResult> paddedShape,
                        Attribute paddingValueAttr) {
  Value paddingValue;
  if (auto complexTy =
          dyn_cast<ComplexType>(getElementTypeOrSelf(v.getType()))) {
    auto complexAttr = cast<ArrayAttr>(paddingValueAttr);
    paddingValue = rewriter.create<complex::ConstantOp>(opToPad.getLoc(),
                                                        complexTy, complexAttr);
  } else {
    paddingValue = rewriter.create<arith::ConstantOp>(
        opToPad.getLoc(), cast<TypedAttr>(paddingValueAttr));
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
  // TODO: use dispatchIndexOpFoldResults(paddedShape, dynDims, paddedShape);

  auto paddedTensorType =
      RankedTensorType::get(tensorShape, getElementTypeOrSelf(v));
  LLVM_DEBUG(DBGS() << "--SUCCESS, makeComposedPadHighOp with type: "
                    << paddedTensorType);
  return makeComposedPadHighOp(rewriter, opToPad.getLoc(), paddedTensorType, v,
                               paddingValue, /*nofold=*/false, dynDims);
}

FailureOr<TilingInterface>
linalg::rewriteAsPaddedOp(RewriterBase &rewriter, TilingInterface opToPad,
                          const PadTilingInterfaceOptions &constOptions,
                          SmallVector<tensor::PadOp> &padOps,
                          PadSizeComputationFunction computePaddingSizeFun) {
  LLVM_DEBUG(DBGS() << "Start rewriteAsPaddedOp : " << opToPad << "\n");
  assert(constOptions.paddingSizes.size() ==
             constOptions.paddingDimensions.size() &&
         "invalid number of elements in padToMultipleOf");

  Location loc = opToPad.getLoc();
  PadTilingInterfaceOptions options(constOptions);
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

  if (llvm::any_of(opToPad->getOperands(),
                   [](Value v) { return isa<MemRefType>(v.getType()); })) {
    return rewriter.notifyMatchFailure(opToPad,
                                       "expected operation on tensors");
  }

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after opToPad because we also take the dims of opToPad's output.
  rewriter.setInsertionPointAfter(opToPad);

  // 1. Get the loopUpperBounds from the TilingInterface.
  SmallVector<Range> iterationDomain = opToPad.getIterationDomain(rewriter);

  // 2. For each operand.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad->getNumOperands());
  for (OpOperand &opOperand : opToPad->getOpOperands()) {
    Value operand = opOperand.get();
    LLVM_DEBUG(DBGS() << "--start padding oprd: " << operand << "\n");

    // 2.a. Skip scalar-like operands.
    Type operandType = operand.getType();
    if (!isa<RankedTensorType>(operandType)) {
      assert(!isa<ShapedType>(operandType) ||
             isa<VectorType>(operandType) &&
                 "Unexpected non-vector ShapedType");
      newOperands.push_back(operand);
      continue;
    }
    // 2.a. Compute padded shape.
    FailureOr<SmallVector<OpFoldResult>> maybePaddedShape =
        computePaddingSizeFun(rewriter, opOperand, iterationDomain, options);
    if (failed(maybePaddedShape)) {
      return rewriter.notifyMatchFailure(opToPad, "could not pad op");
    }

    // 2.b. Expect proper `paddingValues`.
    // TODO: we may want to allow garbage padding in the future, in which case
    // we would just not assert.
    if (opOperand.getOperandNumber() >= options.paddingValues.size()) {
      return rewriter.notifyMatchFailure(opToPad,
                                         "--no padding value specified");
    }
    Attribute paddingValueAttr =
        options.paddingValues[opOperand.getOperandNumber()];

    // 2.c. Perform actual padding.
    Value paddedOperand = padOperand(
        rewriter, opToPad, cast<TypedValue<RankedTensorType>>(operand),
        *maybePaddedShape, paddingValueAttr);
    LLVM_DEBUG(DBGS() << "--done padding operand: " << paddedOperand << "\n");

    // 2.d. Perform actual padding.
    newOperands.push_back(paddedOperand);
    if (auto padOp = paddedOperand.getDefiningOp<tensor::PadOp>())
      padOps.push_back(padOp);
  }

  // 3. Form the resulting tensor::ExtractSliceOp.
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
      ValueRange(newOperands).take_back(opToPad->getNumResults()).getTypes();
  // clone **should** properly notify the rewriter.
  TilingInterface paddedOp =
      clone(rewriter, opToPad, resultTensorTypes, newOperands);
  LLVM_DEBUG(DBGS() << "--cloned padded op: " << paddedOp << "\n");

  // Recover the slice out of the new static results. This keeps the original
  // opToPad around because it uses the dims of the original results.
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

  rewriter.replaceOp(opToPad, paddedSubtensorResults);

  return paddedOp;
}
