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
#include "mlir/Dialect/UB/IR/UBOps.h"
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
static SmallVector<OpFoldResult>
getFullRankPaddingSizes(Builder &b, ArrayRef<OpFoldResult> indexingSizes,
                        const PadTilingInterfaceOptions &options) {
  SmallVector<OpFoldResult> paddingSizes;
  // Complete the padding specification to specify all dimensions.
  for (size_t idx = 0, e = indexingSizes.size(); idx != e; ++idx) {
    // Complete to zero if needed.
    paddingSizes.push_back(options.paddingSizes.size() > idx
                               ? options.paddingSizes[idx]
                               : b.getIndexAttr(0));
    // If a dimension is zero (either specified or completed), replace by:
    //   - 1 if we are padding to the next multiple of.
    //   - indexingSizes[idx] otherwise
    if (isZeroInteger(paddingSizes[idx])) {
      paddingSizes[idx] =
          options.padToMultipleOf ? b.getIndexAttr(1) : indexingSizes[idx];
    }
    LLVM_DEBUG(DBGS() << "----idx: " << idx << " : " << paddingSizes[idx]
                      << "\n");
  }
  return paddingSizes;
}

/// Extracts the constant multiplier from an affine expression of the form
/// `d * c` or `c * d`, where `d` is an AffineDimExpr and `c` is an
/// AffineConstantExpr. Returns 1 if the expression is not a simple
/// multiplication of a dimension and a constant.
static int64_t extractConstantMultiplier(AffineExpr expr) {
  if (auto binOp = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == AffineExprKind::Mul) {
      auto lhsD = dyn_cast<AffineDimExpr>(binOp.getLHS());
      auto rhsC = dyn_cast<AffineConstantExpr>(binOp.getRHS());
      if (lhsD && rhsC) {
        return rhsC.getValue();
      }
      auto lhsC = dyn_cast<AffineConstantExpr>(binOp.getLHS());
      auto rhsD = dyn_cast<AffineDimExpr>(binOp.getRHS());
      if (lhsC && rhsD) {
        return lhsC.getValue();
      }
    }
  }
  return 1;
}

/// Compute the padded shape of the given value `v` of `RankedTensorType` given
///   - `indexingSizes` a list of OpFoldResult.
///   - an `indexingMap` that encodes how the shape of varies with increases
///     in `indexingSizes`.
/// The `indexingMap` encodes how the shape of varies with `indexingSizes`.
/// The `indexingMap` + `indexingSizes` encoding suits StructuredOps.
/// The implementaiton below iteratively combines increases from contributing
/// dimensions using affine.apply operations.
/// The padded shape is computed by evaluating the maximum accessed index per
/// dimension, which may involve multiplying by constant factors derived from
/// the affine indexing expressions. Currently, only a limited set of projected
/// permutation indexing maps are supported, such as
/// - affine_map<(d0, d1, d2) -> (d0, d1)>
/// - affine_map<(d0, d1, d2) -> (d0, d1 + d2)>
/// - affine_map<(d0, d1) -> (d0 * 3 + d1)>
/// In the future, more general interfaces can be devised to encode similar
/// shape evolutions and map between an op and its operands.
SmallVector<OpFoldResult>
linalg::computePaddedShape(OpBuilder &builder, TypedValue<RankedTensorType> v,
                           AffineMap indexingMap,
                           ArrayRef<OpFoldResult> indexingSizes,
                           const PadTilingInterfaceOptions &options) {
  Location loc = v.getLoc();
  SmallVector<OpFoldResult> paddedShape;
  auto tensorType = cast<RankedTensorType>(v.getType());
  paddedShape.resize_for_overwrite(tensorType.getRank());
  assert(tensorType.getRank() == indexingMap.getNumResults() &&
         "expect the number of results of the affine map to match the tensor "
         "rank");

  // "Full-rank" padding specification.
  SmallVector<OpFoldResult> paddingSizes =
      getFullRankPaddingSizes(builder, indexingSizes, options);

  // For each dimension in the operand's shape, iterate over indexingSizes and
  // add the various term contributions.
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
    for (size_t paddingDim = 0, e = paddingSizes.size(); paddingDim != e;
         ++paddingDim) {
      OpFoldResult paddingSize = paddingSizes[paddingDim];
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
                            /*compressDimsFlag=*/true);

      // If we are padding to the next multiple of, compose with ceil(sz) * sz.
      OpFoldResult paddingDimOfr;
      if (options.padToMultipleOf) {
        AffineExpr d0, s0;
        bindDims(builder.getContext(), d0);
        bindSymbols(builder.getContext(), s0);
        AffineMap ceilMap = AffineMap::get(1, 1, d0.ceilDiv(s0) * s0);
        AffineMap composedMap = projectedMap.compose(ceilMap);
        paddingDimOfr = affine::makeComposedFoldedAffineApply(
            builder, loc, composedMap, {indexingSizes[paddingDim], paddingSize},
            /*composeAffineMin=*/true);
      } else {
        // Otherwise just set to paddingSize.
        paddingDimOfr = affine::makeComposedFoldedAffineApply(
            builder, loc, projectedMap, paddingSize);
      }

      // Adjust for the maximum accessed index, which is (paddingSize - 1) *
      // multiplier.
      AffineExpr d0;
      bindDims(builder.getContext(), d0);
      int64_t multiplier = extractConstantMultiplier(projectedMap.getResult(0));
      AffineMap subtractMap = AffineMap::get(1, 0, d0 - multiplier);
      OpFoldResult maxAccessIdx = affine::makeComposedFoldedAffineApply(
          builder, loc, subtractMap, {paddingDimOfr});
      terms.push_back(maxAccessIdx);

      LLVM_DEBUG(DBGS() << "------new term: " << terms.back() << "\n");
    }

    // If there are no terms, just return the dim.
    if (terms.empty()) {
      paddedShape[resultIndex] =
          createFoldedDimOp(builder, loc, v, resultIndex);
      continue;
    }

    // Sum individual terms' contributions.
    SmallVector<AffineExpr> dims(terms.size());
    bindDimsList(builder.getContext(), MutableArrayRef{dims});
    AffineExpr sumExpr = dims.front();
    for (unsigned i = 1; i < dims.size(); ++i)
      sumExpr = sumExpr + dims[i];
    // Add 1 to the maximum accessed index and get the final padded size.
    OpFoldResult paddedDimOfr =
        affine::makeComposedFoldedAffineApply(builder, loc, sumExpr + 1, terms);
    paddedShape[resultIndex] = paddedDimOfr;
  }

  return paddedShape;
}

FailureOr<SmallVector<OpFoldResult>>
linalg::computeIndexingMapOpInterfacePaddedShape(
    OpBuilder &builder, OpOperand &operandToPad,
    ArrayRef<Range> iterationDomain, const PadTilingInterfaceOptions &options) {
  auto transferOp =
      llvm::dyn_cast<IndexingMapOpInterface>(operandToPad.getOwner());
  if (!transferOp)
    return failure();

  // clang-format off
  assert(llvm::all_of(iterationDomain, [&builder](Range r) {
    return r.offset == OpFoldResult(builder.getIndexAttr(0)) &&
    r.stride == OpFoldResult(builder.getIndexAttr(1));
  }) && "expected 0-offset 1-stride loop ranges");
  // clang-format on
  SmallVector<OpFoldResult> loopUpperBounds;
  loopUpperBounds.reserve(iterationDomain.size());
  for (const Range &range : iterationDomain)
    loopUpperBounds.push_back(range.size);

  AffineMap indexingMap = transferOp.getMatchingIndexingMap(&operandToPad);
  return computePaddedShape(
      builder, cast<TypedValue<RankedTensorType>>(operandToPad.get()),
      indexingMap, loopUpperBounds, options);
}

/// Pad a single operand to `paddedShape` using `paddingValueAttr` as padding
/// Value.
static Value padOperand(OpBuilder &builder, TilingInterface opToPad,
                        TypedValue<RankedTensorType> v,
                        ArrayRef<OpFoldResult> paddedShape,
                        Attribute paddingValueAttr) {
  Value paddingValue;
  if (auto complexTy =
          dyn_cast<ComplexType>(getElementTypeOrSelf(v.getType()))) {
    if (auto complexAttr = dyn_cast<ArrayAttr>(paddingValueAttr)) {
      paddingValue = complex::ConstantOp::create(builder, opToPad.getLoc(),
                                                 complexTy, complexAttr);
    }
  } else if (isa<ub::PoisonAttr>(paddingValueAttr)) {
    paddingValue = ub::PoisonOp::create(builder, opToPad.getLoc(),
                                        getElementTypeOrSelf(v.getType()));
  } else if (auto typedAttr = dyn_cast<TypedAttr>(paddingValueAttr)) {
    paddingValue =
        arith::ConstantOp::create(builder, opToPad.getLoc(), typedAttr);
  }
  assert(paddingValue && "failed to create value from padding attribute");

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
  return makeComposedPadHighOp(builder, opToPad.getLoc(), paddedTensorType, v,
                               paddingValue, /*nofold=*/false, dynDims);
}

FailureOr<PadTilingInterfaceResult> linalg::rewriteAsPaddedOp(
    OpBuilder &builder, TilingInterface toPad,
    PadTilingInterfaceOptions options,
    const PadSizeComputationFunction &computePaddingSizeFun) {
  LLVM_DEBUG(DBGS() << "Start rewriteAsPaddedOp : " << toPad << "\n");
  SmallVector<tensor::PadOp> padOps;
  Location loc = toPad.getLoc();

  // Allow inference of pad values if they are not explicitly specified.
  // TODO: be mindful about the value depending on the actual operation.
  if (options.paddingValues.empty()) {
    SmallVector<Type> types(toPad->getOperandTypes());
    llvm::append_range(types, toPad->getResultTypes());
    for (Type t : types) {
      options.paddingValues.push_back(
          builder.getZeroAttr(getElementTypeOrSelf(t)));
    }
  }

  if (llvm::any_of(toPad->getOperands(),
                   [](Value v) { return isa<MemRefType>(v.getType()); })) {
    LLVM_DEBUG(DBGS() << "Not an operation on tensors: FAIL\n");
    return failure();
  }

  OpBuilder::InsertionGuard g(builder);
  // Set IP after toPad because we also take the dims of toPad's output.
  builder.setInsertionPointAfter(toPad);

  // 1. Get the loopUpperBounds from the TilingInterface.
  SmallVector<Range> iterationDomain = toPad.getIterationDomain(builder);

  // 2. For each operand.
  SmallVector<Value> newOperands;
  newOperands.reserve(toPad->getNumOperands());
  for (OpOperand &opOperand : toPad->getOpOperands()) {
    Value operand = opOperand.get();
    LLVM_DEBUG(DBGS() << "--start padding operand: " << operand << "\n");

    // 2.a. Skip scalar-like operands.
    Type operandType = operand.getType();
    if (!isa<RankedTensorType>(operandType)) {
      assert((!isa<ShapedType>(operandType) || isa<VectorType>(operandType)) &&
             "Unexpected non-vector ShapedType");
      newOperands.push_back(operand);
      continue;
    }

    // 2.a. Compute padded shape.
    FailureOr<SmallVector<OpFoldResult>> maybePaddedShape =
        computePaddingSizeFun(builder, opOperand, iterationDomain, options);
    if (failed(maybePaddedShape)) {
      LLVM_DEBUG(DBGS() << "Could not get padded shape of operand: FAIL\n");
      return failure();
    }

    // 2.b. Expect proper `paddingValues`.
    // TODO: we may want to allow garbage padding in the future, in which case
    // we would just not assert.
    if (opOperand.getOperandNumber() >= options.paddingValues.size()) {
      LLVM_DEBUG(DBGS() << "Too few padding values specified: FAIL\n");
      return failure();
    }
    Attribute paddingValueAttr =
        options.paddingValues[opOperand.getOperandNumber()];

    // 2.c. Perform actual padding.
    Value paddedOperand =
        padOperand(builder, toPad, cast<TypedValue<RankedTensorType>>(operand),
                   *maybePaddedShape, paddingValueAttr);
    LLVM_DEBUG(DBGS() << "--done padding operand: " << paddedOperand << "\n");

    newOperands.push_back(paddedOperand);
    if (auto padOp = paddedOperand.getDefiningOp<tensor::PadOp>())
      padOps.push_back(padOp);
  }

  // 3. Form the resulting tensor::ExtractSliceOp.
  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(reifyResultShapes(builder, toPad, reifiedResultShapes))) {
    LLVM_DEBUG(DBGS() << "Failed to reify result shapes: FAIL\n");
    return failure();
  }
  assert(reifiedResultShapes.size() == toPad->getNumResults() &&
         "expected same number of results");

  // Clone `toPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(toPad->getNumResults()).getTypes();
  // clone **should** properly notify the builder.
  TilingInterface paddedOp =
      clone(builder, toPad, resultTensorTypes, newOperands);
  LLVM_DEBUG(DBGS() << "--cloned padded op: " << paddedOp << "\n");

  // Recover the slice out of the new static results.
  SmallVector<Value> paddedSubtensorResults;
  paddedSubtensorResults.reserve(toPad->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = cast<RankedTensorType>(paddedResult.getType()).getRank();
    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    paddedSubtensorResults.push_back(tensor::ExtractSliceOp::create(
        builder, loc, paddedResult, offsets, reifiedResultShapes[resultNumber],
        strides));
  }

  return PadTilingInterfaceResult{padOps, paddedOp, paddedSubtensorResults};
}
