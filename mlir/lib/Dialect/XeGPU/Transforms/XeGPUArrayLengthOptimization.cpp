//===- XeGPUArrayLengthOptimization.cpp - Array Length Opt -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"
#include "mlir/Dialect/XeGPU/uArch/uArchCommon.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "xegpu-array-length-optimization"

using namespace mlir;

namespace {

// Fallback subgroup size used when the target uArch cannot be resolved from
// the op (e.g. standalone unit tests with no chip attribute attached).
constexpr int64_t DEFAULT_SUBGROUP_SIZE = 16;

/// Return the subgroup size for `op`'s target uArch, falling back to
/// DEFAULT_SUBGROUP_SIZE if no chip attribute is attached or the chip is not
/// recognized.
static int64_t getSubgroupSize(Operation *op) {
  auto chipStr = xegpu::getChipStr(op);
  if (!chipStr)
    return DEFAULT_SUBGROUP_SIZE;
  const xegpu::uArch::uArch *targetUArch =
      xegpu::uArch::getUArch(chipStr.value());
  if (!targetUArch)
    return DEFAULT_SUBGROUP_SIZE;
  return targetUArch->getSubgroupSize();
}

/// Helper to compute array_length from FCD and subgroup size.
/// TODO: Currently, we are only allowing subgroupSize as our new FCD for LANE
/// level distribution simplicity. But it can be different, and in the future,
/// we can add that support.
static int64_t computeArrayLength(int64_t fcdSize, int64_t subgroupSize) {
  if (fcdSize <= subgroupSize)
    return 1;
  return fcdSize / subgroupSize;
}

/// Check if a 2D `xegpu.create_nd_tdesc` can be optimized into an
/// array-length-enabled descriptor. Applies only when the FCD is an integer
/// multiple of the subgroup size larger than the subgroup size itself and the
/// tensor desc does not already carry an array_length.
static bool needsOptimization(xegpu::TensorDescType tdescType,
                              int64_t subgroupSize) {
  auto shape = tdescType.getShape();
  if (shape.size() != 2)
    return false;

  int64_t fcd = shape[1];
  if (fcd % subgroupSize != 0)
    return false;

  return fcd > subgroupSize && tdescType.getArrayLength() == 1;
}

/// Returns true if `loadOp` carries a non-identity transpose attribute. A
/// transpose of `[0, 1]` is the identity and is therefore treated as absent.
static bool hasNonIdentityTranspose(xegpu::LoadNdOp loadOp) {
  auto transpose = loadOp.getTranspose();
  if (!transpose)
    return false;
  ArrayRef<int64_t> perm = *transpose;
  return !(perm.size() == 2 && perm[0] == 0 && perm[1] == 1);
}

/// Returns true if `tdescType` carries a lane layout that signals a
/// transpose-intent load (lane_layout = `[SG, 1]`). Such descriptors are
/// rewritten by the transpose peephole optimization and must not be touched
/// here, since stacking the array blocks along the non-FCD dimension would
/// invalidate that rewrite.
static bool hasTransposeLaneLayout(xegpu::TensorDescType tdescType) {
  auto layout = tdescType.getLayoutAttr();
  if (!layout)
    return false;
  SmallVector<int64_t> laneLayout = layout.getEffectiveLaneLayoutAsInt();
  if (laneLayout.size() != 2)
    return false;
  return laneLayout[0] != 1 && laneLayout[1] == 1;
}

/// Remaps a 2-D slice from the flattened array representation to the stacked
/// register representation. Slices within the first array block are unchanged;
/// later slices must start at a block boundary. Slices crossing a block
/// boundary and non-2-D descriptors or slices cannot be represented and return
/// failure.
static FailureOr<SmallVector<int64_t>>
getRemappedExtractOffsets(vector::ExtractStridedSliceOp op,
                          xegpu::TensorDescType tdescType) {
  if (tdescType.getRank() != 2)
    return failure();

  auto offsets = op.getOffsets().getValue();
  auto sizes = op.getSizes().getValue();
  auto strides = op.getStrides().getValue();
  if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2)
    return failure();

  int64_t origOffset0 = cast<IntegerAttr>(offsets[0]).getInt();
  int64_t origOffset1 = cast<IntegerAttr>(offsets[1]).getInt();
  int64_t size1 = cast<IntegerAttr>(sizes[1]).getInt();
  int64_t blockHeight = tdescType.getShape()[0];
  int64_t arrayWidth = tdescType.getShape()[1];

  int64_t localOffset1 = origOffset1 % arrayWidth;
  if (localOffset1 + size1 > arrayWidth)
    return failure();
  if (origOffset1 < arrayWidth)
    return SmallVector<int64_t>{origOffset0, origOffset1};
  if (origOffset1 % arrayWidth != 0)
    return failure();

  int64_t arrayIndex = origOffset1 / arrayWidth;
  return SmallVector<int64_t>{origOffset0 + arrayIndex * blockHeight,
                              /*offset1=*/0};
}

/// Rewrite `xegpu.create_nd_tdesc` to fold an array_length attribute into the
/// resulting tensor descriptor type. Supports static memref, dynamic-shape
/// memref, and raw-pointer (integer) sources — the memory region described by
/// `shape`/`strides` is unchanged; only the tensor_desc view is narrowed along
/// the FCD and tagged with `array_length`. Skipped if any consumer load_nd
/// carries a non-identity transpose, since stacking the array blocks along the
/// non-FCD dimension would invalidate that load.
class OptimizeCreateNdDescOp : public OpRewritePattern<xegpu::CreateNdDescOp> {
public:
  using OpRewritePattern<xegpu::CreateNdDescOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    // sub-byte type is not supported for now.
    if (op.getType().getElementTypeBitWidth() < 8)
      return failure();
    int64_t subgroupSize = getSubgroupSize(op);
    auto tdescType = op.getType();
    if (!needsOptimization(tdescType, subgroupSize))
      return failure();

    // A transpose lane layout marks this descriptor as a candidate for the
    // separate transpose peephole; stacking the array blocks would break it.
    if (hasTransposeLaneLayout(tdescType))
      return failure();

    Value source = op.getSource();
    if (!isa<MemRefType, IntegerType>(source.getType()))
      return failure();

    // Bail out if any consumer is a transposing load_nd.
    for (Operation *user : op.getResult().getUsers()) {
      if (auto loadOp = dyn_cast<xegpu::LoadNdOp>(user))
        if (hasNonIdentityTranspose(loadOp))
          return failure();
    }

    auto shape = tdescType.getShape();
    int64_t arrayLength = computeArrayLength(shape[1], subgroupSize);
    SmallVector<int64_t> newShape = {shape[0], shape[1] / arrayLength};
    if (auto layout = tdescType.getLayoutAttr();
        layout && !layout.isDistributable(newShape))
      return failure();

    auto newTdescType = xegpu::TensorDescType::get(
        newShape, tdescType.getElementType(), arrayLength,
        tdescType.getBoundaryCheck(), tdescType.getMemorySpace(),
        tdescType.getLayout());

    SmallVector<xegpu::LoadNdOp> loadOps;
    for (Operation *descriptorUser : op.getResult().getUsers()) {
      if (auto prefetchOp = dyn_cast<xegpu::PrefetchNdOp>(descriptorUser)) {
        if (auto layout = prefetchOp.getAnchorLayout();
            layout && !layout.isDistributable(newShape))
          return failure();
        continue;
      }

      auto loadOp = dyn_cast<xegpu::LoadNdOp>(descriptorUser);
      if (!loadOp)
        return failure();

      if (auto layout = loadOp.getAnchorLayout();
          layout && !layout.isDistributable(newShape))
        return failure();
      auto loadType = dyn_cast<VectorType>(loadOp.getType());
      if (!loadType || loadType.getRank() != 2)
        return failure();
      for (Operation *loadResultUser : loadOp.getResult().getUsers()) {
        auto extractOp =
            dyn_cast<vector::ExtractStridedSliceOp>(loadResultUser);
        if (!extractOp ||
            failed(getRemappedExtractOffsets(extractOp, newTdescType)))
          return failure();
      }
      loadOps.push_back(loadOp);
    }

    // Updating the descriptor alone temporarily invalidates its load users.
    // Keep the descriptor, load results, and extract offsets consistent within
    // this single pattern application.
    for (xegpu::LoadNdOp loadOp : loadOps) {
      for (Operation *loadResultUser : loadOp.getResult().getUsers()) {
        auto extractOp = cast<vector::ExtractStridedSliceOp>(loadResultUser);
        SmallVector<int64_t> newOffsets =
            *getRemappedExtractOffsets(extractOp, newTdescType);
        rewriter.modifyOpInPlace(extractOp, [&]() {
          extractOp.setOffsetsAttr(rewriter.getI64ArrayAttr(newOffsets));
        });
      }

      auto loadType = cast<VectorType>(loadOp.getType());
      SmallVector<int64_t> newLoadShape = {newShape[0] * arrayLength,
                                           newShape[1]};
      auto newLoadType =
          VectorType::get(newLoadShape, loadType.getElementType());
      rewriter.modifyOpInPlace(
          loadOp, [&]() { loadOp.getResult().setType(newLoadType); });
    }
    rewriter.modifyOpInPlace(op,
                             [&]() { op.getResult().setType(newTdescType); });
    return success();
  }
};

} // namespace

void xegpu::populateXeGPUArrayLengthOptimizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<OptimizeCreateNdDescOp>(patterns.getContext());
}
