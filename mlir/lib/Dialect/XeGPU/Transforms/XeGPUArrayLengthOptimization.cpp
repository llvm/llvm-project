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
#include "mlir/Transforms/DialectConversion.h"
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
  // Only optimize 2D tensors
  auto shape = tdescType.getShape();
  if (shape.size() != 2)
    return false;

  // Check if FCD is larger than subgroup size
  int64_t fcd = shape[1];
  if (fcd <= SUBGROUP_SIZE)
    return false;

  // Check if FCD is a multiple of subgroup size
  if (fcd % SUBGROUP_SIZE != 0)
    return false;

  // Check if array_length is already set to non-1
  if (tdescType.getArrayLength() > 1)
    return false;

  return true;
}

/// Pattern to rewrite xegpu.create_nd_tdesc operations
class OptimizeCreateNdDescOp
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
public:
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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

    // Check if the op has explicit offsets/sizes/strides or if they're inferred
    auto offsets = op.getMixedOffsets();
    auto sizes = op.getMixedSizes();
    auto strides = op.getMixedStrides();

    // Check if we have a simple static memref source
    Value source = op.getSource();
    auto memrefType = dyn_cast<MemRefType>(source.getType());
    if (!memrefType || !memrefType.hasStaticShape()) {
      // For now, only handle simple static memrefs
      return failure();
    }

    // Cast to TypedValue<MemRefType> for the builder
    auto memrefSource = cast<TypedValue<MemRefType>>(source);

    // Build operation state and use the simple builder
    OperationState state(op.getLoc(), xegpu::CreateNdDescOp::getOperationName());
    xegpu::CreateNdDescOp::build(rewriter, state, newTdescType, memrefSource);
    auto newOp = cast<xegpu::CreateNdDescOp>(rewriter.create(state));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Pattern to rewrite xegpu.load_nd operations
class OptimizeLoadNdOp : public OpConversionPattern<xegpu::LoadNdOp> {
public:
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted tensor desc type (after CreateNdDescOp conversion)
    auto adaptedTdescType =
        dyn_cast<xegpu::TensorDescType>(adaptor.getTensorDesc().getType());
    if (!adaptedTdescType)
      return failure();

    // Check if the adapted tensor desc has array_length > 1
    int64_t arrayLength = adaptedTdescType.getArrayLength();
    if (arrayLength <= 1)
      return failure();

    auto origVectorType = op.getType();
    auto origShape = origVectorType.getShape();
    if (origShape.size() != 2)
      return failure();

    // Compute new vector shape for register layout
    // New non-FCD = old non-FCD * array_length
    // New FCD = old FCD / array_length
    int64_t newNonFCD = origShape[0] * arrayLength;
    int64_t newFCD = adaptedTdescType.getShape()[1];

    SmallVector<int64_t> newShape = {newNonFCD, newFCD};
    auto newVectorType =
        VectorType::get(newShape, origVectorType.getElementType());

    // Create new LoadNdOp with updated result type
    auto newLoadOp = xegpu::LoadNdOp::create(
        rewriter, op.getLoc(), newVectorType, adaptor.getTensorDesc(),
        op.getMixedOffsets(), op.getPackedAttr(), op.getTransposeAttr(),
        op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr(),
        op.getLayoutAttr());

    rewriter.replaceOp(op, newLoadOp.getResult());
    return success();
  }
};

/// Pattern to rewrite xegpu.prefetch_nd operations
class OptimizePrefetchNdOp : public OpConversionPattern<xegpu::PrefetchNdOp> {
public:
  using OpConversionPattern<xegpu::PrefetchNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::PrefetchNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted tensor desc type (after CreateNdDescOp conversion)
    auto adaptedTdescType =
        dyn_cast<xegpu::TensorDescType>(adaptor.getTensorDesc().getType());
    if (!adaptedTdescType)
      return failure();

    // Check if the adapted tensor desc has array_length > 1
    int64_t arrayLength = adaptedTdescType.getArrayLength();
    if (arrayLength <= 1)
      return failure();

    // Create new PrefetchNdOp with adapted tensor desc
    xegpu::PrefetchNdOp::create(rewriter, op.getLoc(),
                                adaptor.getTensorDesc(), op.getMixedOffsets(),
                                op.getL1HintAttr(), op.getL2HintAttr(),
                                op.getL3HintAttr(), op.getLayoutAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to update vector.extract_strided_slice operations
/// Memory layout (32x32): [0:32][0:16] and [0:32][16:32] are side by side
/// Register layout (64x16): [0:32][0:16] and [32:64][0:16] are stacked
class UpdateExtractStridedSliceOp
    : public OpConversionPattern<vector::ExtractStridedSliceOp> {
public:
  using OpConversionPattern<
      vector::ExtractStridedSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted vector operand
    Value adaptedVector = adaptor.getOperands()[0];
    auto sourceType = dyn_cast<VectorType>(adaptedVector.getType());
    if (!sourceType || sourceType.getRank() != 2)
      return failure();

    // Check if the source comes from a load_nd that was optimized
    auto loadOp = adaptedVector.getDefiningOp<xegpu::LoadNdOp>();
    if (!loadOp)
      return failure();

    auto tdescType = loadOp.getTensorDescType();
    int64_t arrayLength = tdescType.getArrayLength();
    if (arrayLength <= 1)
      return failure();

    // Get original offsets and sizes
    auto offsets = op.getOffsets().getValue();
    auto sizes = op.getSizes().getValue();
    auto strides = op.getStrides().getValue();

    if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2)
      return failure();

    int64_t origOffset0 = cast<IntegerAttr>(offsets[0]).getInt();
    int64_t origOffset1 = cast<IntegerAttr>(offsets[1]).getInt();

    // Convert memory layout indexing to register layout indexing
    // Memory layout: blocks are side-by-side in the FCD
    // Register layout: blocks are stacked in the non-FCD
    //
    // Original memory indexing: [offset0][offset1]
    // where offset1 determines which array element we're in
    //
    // New register indexing:
    // - array_index = offset1 / new_FCD
    // - new_offset0 = offset0 + (array_index * original_rows)
    // - new_offset1 = offset1 % new_FCD

    int64_t newFCD = tdescType.getShape()[1];
    int64_t origRows = sourceType.getShape()[0] / arrayLength;

    int64_t arrayIndex = origOffset1 / newFCD;
    int64_t newOffset0 = origOffset0 + (arrayIndex * origRows);
    int64_t newOffset1 = origOffset1 % newFCD;

    // Create new offsets
    SmallVector<int64_t> newOffsets = {newOffset0, newOffset1};

    // Create new ExtractStridedSliceOp with updated offsets
    auto newOp = vector::ExtractStridedSliceOp::create(
        rewriter, op.getLoc(), adaptedVector, newOffsets,
        llvm::to_vector(llvm::map_range(
            sizes, [](Attribute a) { return cast<IntegerAttr>(a).getInt(); })),
        llvm::to_vector(llvm::map_range(
            strides,
            [](Attribute a) { return cast<IntegerAttr>(a).getInt(); })));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

} // namespace

namespace mlir {
namespace xegpu {

void populateXeGPUArrayLengthOptimizationPatterns(
    RewritePatternSet &patterns, TypeConverter &converter) {
  patterns.add<OptimizeCreateNdDescOp, OptimizeLoadNdOp, OptimizePrefetchNdOp,
               UpdateExtractStridedSliceOp>(converter,
                                            patterns.getContext());
}

} // namespace xegpu
} // namespace mlir

namespace {

struct XeGPUArrayLengthOptimizationPass final
    : public xegpu::impl::XeGPUArrayLengthOptimizationBase<
          XeGPUArrayLengthOptimizationPass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    TypeConverter converter;
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    // Mark CreateNdDescOp as legal only if it doesn't need optimization
    target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
        [](xegpu::CreateNdDescOp op) {
          return !needsOptimization(op.getType());
        });

    // Mark LoadNdOp as legal only if its tensor desc doesn't need optimization
    target.addDynamicallyLegalOp<xegpu::LoadNdOp>([](xegpu::LoadNdOp op) {
      return !needsOptimization(op.getTensorDescType());
    });

    // Mark PrefetchNdOp as legal only if its tensor desc doesn't need
    // optimization
    target.addDynamicallyLegalOp<xegpu::PrefetchNdOp>(
        [](xegpu::PrefetchNdOp op) {
          return !needsOptimization(op.getTensorDescType());
        });

    // Mark ExtractStridedSliceOp as legal if it doesn't extract from an
    // optimized load
    target.addDynamicallyLegalOp<vector::ExtractStridedSliceOp>(
        [](vector::ExtractStridedSliceOp op) {
          auto loadOp = op.getSource().getDefiningOp<xegpu::LoadNdOp>();
          if (!loadOp)
            return true;
          auto tdescType = loadOp.getTensorDescType();
          return tdescType.getArrayLength() <= 1;
        });

    // Identity type conversion
    converter.addConversion([](Type type) { return type; });

    target.addLegalDialect<xegpu::XeGPUDialect, vector::VectorDialect>();

    xegpu::populateXeGPUArrayLengthOptimizationPatterns(patterns, converter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      DBGS() << "Array length optimization pass failed.\n";
      return signalPassFailure();
    }
  }
};

} // namespace
