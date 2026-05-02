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
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"
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

/// Rewrite `xegpu.create_nd_tdesc` to fold an array_length attribute into the
/// resulting tensor descriptor type. Only applies when the source is a static
/// memref; dynamic-shape sources are left unchanged. Skipped if any consumer
/// load_nd carries a non-identity transpose, since stacking the array blocks
/// along the non-FCD dimension would invalidate that load.
class OptimizeCreateNdDescOp : public OpRewritePattern<xegpu::CreateNdDescOp> {
public:
  using OpRewritePattern<xegpu::CreateNdDescOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    int64_t subgroupSize = getSubgroupSize(op);
    auto tdescType = op.getType();
    if (!needsOptimization(tdescType, subgroupSize))
      return failure();

    // A transpose lane layout marks this descriptor as a candidate for the
    // separate transpose peephole; stacking the array blocks would break it.
    if (hasTransposeLaneLayout(tdescType))
      return failure();

    // Only static memref sources are supported for now.
    // TODO: extend to dynamic-shape memrefs and raw pointer sources by
    // rewriting the `shape`/`strides` operands of create_nd_tdesc.
    auto memrefSource = dyn_cast<TypedValue<MemRefType>>(op.getSource());
    if (!memrefSource || !memrefSource.getType().hasStaticShape())
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

    auto newTdescType = xegpu::TensorDescType::get(
        newShape, tdescType.getElementType(), arrayLength,
        tdescType.getBoundaryCheck(), tdescType.getMemorySpace(),
        tdescType.getLayout());

    auto newOp = xegpu::CreateNdDescOp::create(rewriter, op.getLoc(),
                                               newTdescType, memrefSource);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Pattern to rewrite xegpu.load_nd operations
class OptimizeLoadNdOp : public OpRewritePattern<xegpu::LoadNdOp> {
public:
  using OpRewritePattern<xegpu::LoadNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::LoadNdOp op,
                                PatternRewriter &rewriter) const override {
    auto tdescType = op.getTensorDescType();
    int64_t arrayLength = tdescType.getArrayLength();

    if (arrayLength <= 1)
      return failure();

    // Transposing loads are not compatible with the stacked-on-non-FCD layout
    // that this pass produces.
    if (hasNonIdentityTranspose(op) || hasTransposeLaneLayout(tdescType))
      return failure();

    auto origVectorType = op.getType();
    auto origShape = origVectorType.getShape();
    if (origShape.size() != 2)
      return failure();

    // The expected vector shape is: [tdesc_non_FCD * array_length, tdesc_FCD]
    int64_t expectedNonFCD = tdescType.getShape()[0] * arrayLength;
    int64_t expectedFCD = tdescType.getShape()[1];

    // If already matches expected shape, skip
    if (origShape[0] == expectedNonFCD && origShape[1] == expectedFCD)
      return failure();

    // Compute new vector shape for register layout
    SmallVector<int64_t> newShape = {expectedNonFCD, expectedFCD};
    auto newVectorType =
        VectorType::get(newShape, origVectorType.getElementType());

    // Create new LoadNdOp with updated result type
    auto newLoadOp = xegpu::LoadNdOp::create(
        rewriter, op.getLoc(), newVectorType, op.getTensorDesc(),
        op.getMixedOffsets(), op.getPackedAttr(), op.getTransposeAttr(),
        op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr(),
        op.getLayoutAttr());

    rewriter.replaceOp(op, newLoadOp.getResult());
    return success();
  }
};

/// Rewrite `vector.extract_strided_slice` offsets so they index into the
/// stacked register layout produced by `OptimizeLoadNdOp`.
///
/// The optimized load places `arrayLength` blocks side-by-side in memory
/// but stacks them along the non-FCD dimension in registers. Given a
/// tensor desc of shape `[H, W]` with array_length = A:
///
///   memory layout (what the extract offsets refer to): `[H, W * A]`
///   register layout (what the new load returns):       `[H * A, W]`
///
/// An extract at memory offset `[r, c]` therefore maps to register offset
/// `[r + (c / W) * H, 0]` — provided the extract is block-aligned in the
/// FCD dimension, i.e. `c % W == 0`.
///
/// Example (`A = 2`, `H = 32`, `W = 16`):
///
///   // before
///   %v = xegpu.load_nd %t : ... -> vector<32x32xf16>
///   %e = vector.extract_strided_slice %v
///          {offsets = [0, 16], sizes = [16, 16], strides = [1, 1]}
///          : vector<32x32xf16> to vector<16x16xf16>
///
///   // after (load rewritten to vector<64x16>, extract offset remapped)
///   %v = xegpu.load_nd %t : ... -> vector<64x16xf16>
///   %e = vector.extract_strided_slice %v
///          {offsets = [32, 0], sizes = [16, 16], strides = [1, 1]}
///          : vector<64x16xf16> to vector<16x16xf16>
class UpdateExtractStridedSliceOp
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
public:
  using OpRewritePattern<vector::ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<VectorType>(op.getSource().getType());
    if (!sourceType || sourceType.getRank() != 2)
      return failure();

    auto loadOp = op.getSource().getDefiningOp<xegpu::LoadNdOp>();
    if (!loadOp)
      return failure();

    auto tdescType = loadOp.getTensorDescType();
    int64_t arrayLength = tdescType.getArrayLength();
    if (arrayLength <= 1)
      return failure();

    auto offsets = op.getOffsets().getValue();
    auto sizes = op.getSizes().getValue();
    auto strides = op.getStrides().getValue();

    if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2)
      return failure();

    int64_t origOffset0 = cast<IntegerAttr>(offsets[0]).getInt();
    int64_t origOffset1 = cast<IntegerAttr>(offsets[1]).getInt();

    int64_t blockHeight = tdescType.getShape()[0];
    int64_t arrayWidth = tdescType.getShape()[1];

    // Skip extracts that already live entirely inside block 0: their offsets
    // are identical in the memory and register layouts, so there is nothing
    // to rewrite.
    if (origOffset1 < arrayWidth)
      return failure();

    // The remap is only well-defined when the extract is aligned to an array
    // block along the FCD.
    assert(origOffset1 % arrayWidth == 0 &&
           "extract offset along FCD must be a multiple of the array width");

    int64_t arrayIndex = origOffset1 / arrayWidth;
    SmallVector<int64_t> newOffsets = {origOffset0 + arrayIndex * blockHeight,
                                       /*offset1=*/0};

    auto toInts = [](ArrayAttr arr) {
      return llvm::to_vector(llvm::map_range(
          arr, [](Attribute a) { return cast<IntegerAttr>(a).getInt(); }));
    };
    SmallVector<int64_t> sliceSizes = toInts(op.getSizes());
    SmallVector<int64_t> sliceStrides = toInts(op.getStrides());

    auto newOp = vector::ExtractStridedSliceOp::create(
        rewriter, op.getLoc(), op.getSource(), newOffsets, sliceSizes,
        sliceStrides);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

} // namespace

void xegpu::populateXeGPUArrayLengthOptimizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<OptimizeCreateNdDescOp, OptimizeLoadNdOp,
               UpdateExtractStridedSliceOp>(patterns.getContext());
}
