//===- XeGPUArrayLengthOptimization.cpp - Array Length Opt -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUARRAYLENGTHOPTIMIZATION
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-array-length-optimization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {

// Subgroup size is typically 16 for Intel GPUs
constexpr int64_t SUBGROUP_SIZE = 16;

/// Helper to compute array_length from FCD and subgroup size
static int64_t computeArrayLength(int64_t fcdSize) {
  if (fcdSize <= SUBGROUP_SIZE)
    return 1;
  return fcdSize / SUBGROUP_SIZE;
}

/// Helper to compute new FCD after introducing array_length
static int64_t computeNewFCD(int64_t oldFCD, int64_t arrayLength) {
  return oldFCD / arrayLength;
}

/// Check if a load_nd or prefetch_nd operation needs optimization
static bool needsOptimization(xegpu::TensorDescType tdescType) {
  auto shape = tdescType.getShape();
  if (shape.size() != 2)
    return false; // Only 2D tensors

  int64_t fcd = shape[1];
  if (fcd <= SUBGROUP_SIZE || fcd % SUBGROUP_SIZE != 0)
    return false; // FCD must be > subgroup_size and evenly divisible

  return tdescType.getArrayLength() == 1; // Skip if already optimized
}

/// Pattern to rewrite xegpu.create_nd_tdesc operations using simple
/// RewritePattern
class OptimizeCreateNdDescOp : public OpRewritePattern<xegpu::CreateNdDescOp> {
public:
  using OpRewritePattern<xegpu::CreateNdDescOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    auto tdescType = op.getType();
    if (!needsOptimization(tdescType))
      return failure();

    auto shape = tdescType.getShape();
    int64_t oldFCD = shape[1];
    int64_t arrayLength = computeArrayLength(oldFCD);
    int64_t newFCD = computeNewFCD(oldFCD, arrayLength);

    // Build new shape with updated FCD
    SmallVector<int64_t> newShape = {shape[0], newFCD};

    // Create new TensorDescType with array_length
    auto newTdescType = xegpu::TensorDescType::get(
        newShape, tdescType.getElementType(), arrayLength,
        tdescType.getBoundaryCheck(), tdescType.getMemorySpace(),
        tdescType.getLayout());

    // Check if we have a simple static memref source
    Value source = op.getSource();
    auto memrefType = dyn_cast<MemRefType>(source.getType());
    if (!memrefType || !memrefType.hasStaticShape()) {
      return failure();
    }

    // Cast to TypedValue<MemRefType> for the builder
    auto memrefSource = cast<TypedValue<MemRefType>>(source);

    // Build operation state and use the simple builder
    OperationState state(op.getLoc(),
                         xegpu::CreateNdDescOp::getOperationName());
    xegpu::CreateNdDescOp::build(rewriter, state, newTdescType, memrefSource);
    auto newOp = cast<xegpu::CreateNdDescOp>(rewriter.create(state));

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

/// Pattern to rewrite xegpu.prefetch_nd operations
/// Note: PrefetchNdOp doesn't require transformation - it automatically uses
/// the optimized tensor descriptor created by CreateNdDescOp
class OptimizePrefetchNdOp : public OpRewritePattern<xegpu::PrefetchNdOp> {
public:
  using OpRewritePattern<xegpu::PrefetchNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::PrefetchNdOp op,
                                PatternRewriter &rewriter) const override {
    // PrefetchNdOp doesn't need rewriting - it just uses the tensor descriptor
    // as-is. After CreateNdDescOp optimizes the descriptor, PrefetchNdOp
    // automatically uses the optimized version.
    return failure();
  }
};

/// Pattern to update vector.extract_strided_slice operations
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

    int64_t newFCD = tdescType.getShape()[1];
    int64_t origRows = sourceType.getShape()[0] / arrayLength;

    int64_t arrayIndex = origOffset1 / newFCD;
    int64_t newOffset0 = origOffset0 + (arrayIndex * origRows);
    int64_t newOffset1 = origOffset1 % newFCD;

    // If offsets don't change, this extract is already transformed
    if (newOffset0 == origOffset0 && newOffset1 == origOffset1)
      return failure();

    SmallVector<int64_t> newOffsets = {newOffset0, newOffset1};

    auto newOp = vector::ExtractStridedSliceOp::create(
        rewriter, op.getLoc(), op.getSource(), newOffsets,
        llvm::to_vector(llvm::map_range(
            sizes, [](Attribute a) { return cast<IntegerAttr>(a).getInt(); })),
        llvm::to_vector(llvm::map_range(strides, [](Attribute a) {
          return cast<IntegerAttr>(a).getInt();
        })));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

} // namespace

namespace mlir {
namespace xegpu {

void populateXeGPUArrayLengthOptimizationPatterns(RewritePatternSet &patterns) {
  patterns.add<OptimizeCreateNdDescOp, OptimizeLoadNdOp, OptimizePrefetchNdOp,
               UpdateExtractStridedSliceOp>(patterns.getContext());
}

} // namespace xegpu
} // namespace mlir

namespace {

struct XeGPUArrayLengthOptimizationPass final
    : public xegpu::impl::XeGPUArrayLengthOptimizationBase<
          XeGPUArrayLengthOptimizationPass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);

    xegpu::populateXeGPUArrayLengthOptimizationPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      DBGS() << "Array length optimization pass failed.\n";
      return signalPassFailure();
    }
  }
};

} // namespace
