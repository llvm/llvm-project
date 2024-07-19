//===- VectorMaskElimination.cpp - Eliminate Vector Masks -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::vector;
namespace {

/// If `value` is a constant multiple of `vector.vscale` return the multiplier.
std::optional<int64_t> getConstantVscaleMultiplier(Value value) {
  if (value.getDefiningOp<vector::VectorScaleOp>())
    return 1;
  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return {};
  auto lhs = mul.getLhs();
  auto rhs = mul.getRhs();
  if (lhs.getDefiningOp<vector::VectorScaleOp>())
    return getConstantIntValue(rhs);
  if (rhs.getDefiningOp<vector::VectorScaleOp>())
    return getConstantIntValue(lhs);
  return {};
}

/// Attempts to resolve a (scalable) CreateMaskOp to an all-true constant mask.
/// All-true masks can then be eliminated by simple folds.
LogicalResult resolveAllTrueCreateMaskOp(IRRewriter &rewriter,
                                         vector::CreateMaskOp createMaskOp,
                                         VscaleRange vscaleRange) {
  auto maskType = createMaskOp.getVectorType();
  auto maskTypeDimScalableFlags = maskType.getScalableDims();
  auto maskTypeDimSizes = maskType.getShape();

  struct UnknownMaskDim {
    size_t position;
    Value dimSize;
  };

  // Check for any dims that could be (partially) false before doing the more
  // expensive value bounds computations.
  SmallVector<UnknownMaskDim> unknownDims;
  for (auto [i, dimSize] : llvm::enumerate(createMaskOp.getOperands())) {
    if (auto intSize = getConstantIntValue(dimSize)) {
      // Mask not all-true for this dim.
      if (maskTypeDimScalableFlags[i] || intSize < maskTypeDimSizes[i])
        return failure();
    } else if (auto vscaleMultiplier = getConstantVscaleMultiplier(dimSize)) {
      // Mask not all-true for this dim.
      if (vscaleMultiplier < maskTypeDimSizes[i])
        return failure();
    } else {
      // Unknown (without further analysis).
      unknownDims.push_back(UnknownMaskDim{i, dimSize});
    }
  }

  for (auto [i, dimSize] : unknownDims) {
    // Compute the lower bound for the unknown dimension (i.e. the smallest
    // value it could be).
    auto lowerBound =
        vector::ScalableValueBoundsConstraintSet::computeScalableBound(
            dimSize, {}, vscaleRange.vscaleMin, vscaleRange.vscaleMax,
            presburger::BoundType::LB);
    if (failed(lowerBound))
      return failure();
    auto boundSize = lowerBound->getSize();
    if (failed(boundSize))
      return failure();
    if (boundSize->scalable) {
      // If the lower bound is scalable and >= to the mask dim size then this
      // dim is all-true.
      if (boundSize->baseSize < maskTypeDimSizes[i])
        return failure();
    } else {
      // If the lower bound is a constant and >= to the _fixed-size_ mask dim
      // size then this dim is all-true.
      if (maskTypeDimScalableFlags[i])
        return failure();
      if (boundSize->baseSize < maskTypeDimSizes[i])
        return failure();
    }
  }

  // Replace createMaskOp with an all-true constant. This should result in the
  // mask being removed in most cases (as xfer ops + vector.mask have folds to
  // remove all-true masks).
  auto allTrue = rewriter.create<arith::ConstantOp>(
      createMaskOp.getLoc(), maskType, DenseElementsAttr::get(maskType, true));
  rewriter.replaceAllUsesWith(createMaskOp, allTrue);
  return success();
}

} // namespace

namespace mlir::vector {

void eliminateVectorMasks(IRRewriter &rewriter, FunctionOpInterface function,
                          std::optional<VscaleRange> vscaleRange) {
  // TODO: Support fixed-size case. This is less likely to be useful as for
  // fixed-size code dimensions are all static so masks tend to fold away.
  if (!vscaleRange)
    return;

  OpBuilder::InsertionGuard g(rewriter);

  // Build worklist so we can safely insert new ops in
  // `resolveAllTrueCreateMaskOp()`.
  SmallVector<vector::CreateMaskOp> worklist;
  function.walk([&](vector::CreateMaskOp createMaskOp) {
    worklist.push_back(createMaskOp);
  });

  rewriter.setInsertionPointToStart(&function.front());
  for (auto mask : worklist)
    (void)resolveAllTrueCreateMaskOp(rewriter, mask, *vscaleRange);
}

} // namespace mlir::vector
