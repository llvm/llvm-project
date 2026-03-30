//===- VectorMaskElimination.cpp - Eliminate Vector Masks -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::vector;
namespace {

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

  // Loop over the CreateMaskOp operands and collect unknown dims (i.e. dims
  // that are not obviously constant). If any constant dimension is not all-true
  // bail out early (as this transform only trying to resolve all-true masks).
  // This avoids doing value-bounds anaylis in cases like:
  // `%mask = vector.create_mask %dynamicValue, %c2 : vector<8x4xi1>`
  // ...where it is known the mask is not all-true by looking at `%c2`.
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
    // Compute lower and upper bounds for the unknown dimension. We need both
    // to agree (same constant or same scalable expression) before treating the
    // mask as all-true: using only a lower bound is unsound when the value can
    // vary at runtime (e.g. tensor.dim of a dynamic slice that is full-sized
    // on most iterations but partial on the last). The lower bound analysis
    // may then report the full size even though the upper bound analysis (or
    // the differing tight range) shows the dimension is not a single constant.
    FailureOr<ConstantOrScalableBound> dimLowerBound =
        vector::ScalableValueBoundsConstraintSet::computeScalableBound(
            dimSize, {}, vscaleRange.vscaleMin, vscaleRange.vscaleMax,
            presburger::BoundType::LB);
    if (failed(dimLowerBound))
      return failure();
    FailureOr<ConstantOrScalableBound::BoundSize> dimLowerBoundSize =
        dimLowerBound->getSize();
    if (failed(dimLowerBoundSize))
      return failure();

    FailureOr<ConstantOrScalableBound> dimUpperBound =
        vector::ScalableValueBoundsConstraintSet::computeScalableBound(
            dimSize, {}, vscaleRange.vscaleMin, vscaleRange.vscaleMax,
            presburger::BoundType::UB);
    if (failed(dimUpperBound))
      return failure();
    FailureOr<ConstantOrScalableBound::BoundSize> dimUpperBoundSize =
        dimUpperBound->getSize();
    if (failed(dimUpperBoundSize))
      return failure();

    if (dimLowerBoundSize->scalable != dimUpperBoundSize->scalable ||
        dimLowerBoundSize->baseSize != dimUpperBoundSize->baseSize)
      return failure();

    if (dimLowerBoundSize->scalable) {
      // 1. The bound is scalable. If it is < the mask dim size then this dim
      // is not all-true.
      if (dimLowerBoundSize->baseSize < maskTypeDimSizes[i])
        return failure();
    } else {
      // 2. The bound is a constant.
      // - If the mask dim size is scalable then this dim is not all-true.
      if (maskTypeDimScalableFlags[i])
        return failure();
      // - If the constant < the _fixed-size_ mask dim size then not all-true.
      if (dimLowerBoundSize->baseSize < maskTypeDimSizes[i])
        return failure();
    }
  }

  // Replace createMaskOp with an all-true constant. This should result in the
  // mask being removed in most cases (as xfer ops + vector.mask have folds to
  // remove all-true masks).
  auto allTrue = vector::ConstantMaskOp::create(
      rewriter, createMaskOp.getLoc(), maskType, ConstantMaskKind::AllTrue);
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

  // Early exit for functions without a body.
  if (function.isExternal())
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
