//===- BubbleUpExtractSliceThroughTilingInterface.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to bubble up `tensor.extract_slice` operations
// through producers that implement the `TilingInterface`. Unlike the Linalg-
// specific bubble up pattern, this works with any operation implementing
// TilingInterface and supports multiple non-overlapping extract_slice
// consumers.
//
// The transformation reduces computation by creating smaller/tiled operations
// that compute only the slices actually needed by consumers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bubble-up-extract-slice-tiling-interface"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Check if any two slices in the list overlap.
/// Returns failure() if overlap analysis cannot be determined.
/// Returns false if slices are guaranteed non-overlapping.
/// Returns true if slices may overlap.
static FailureOr<bool>
hasOverlappingSlices(MLIRContext *ctx,
                     SmallVectorImpl<tensor::ExtractSliceOp> &slices) {
  for (size_t i = 0; i < slices.size(); ++i) {
    for (size_t j = i + 1; j < slices.size(); ++j) {
      HyperrectangularSlice slice1(
          cast<OffsetSizeAndStrideOpInterface>(slices[i].getOperation()));
      HyperrectangularSlice slice2(
          cast<OffsetSizeAndStrideOpInterface>(slices[j].getOperation()));

      FailureOr<bool> overlapping =
          ValueBoundsConstraintSet::areOverlappingSlices(ctx, slice1, slice2);
      if (failed(overlapping)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Could not determine if slices overlap at indices " << i
                   << " and " << j << "\n");
        return failure();
      }
      if (*overlapping) {
        LLVM_DEBUG(llvm::dbgs() << "Found overlapping slices at indices " << i
                                << " and " << j << "\n");
        return true;
      }
    }
  }
  return false;
}

/// Pattern to bubble up extract_slice through operations implementing
/// TilingInterface.
///
/// Matches: TilingInterface op whose output is consumed only by non-overlapping
/// extract_slice ops.
///
/// Transforms to: tiled operations (via
/// TilingInterface::generateResultTileValue) that each compute only a specific
/// output slice, with extract_slice operations applied to the inputs.
///
/// For example:
///
/// Before:
/// ```mlir
/// %0 = "some.tiling_interface_op"(%input) : (tensor<1x9450x256xf32>)
///                                           -> tensor<1x9450x256xf32>
/// %1 = tensor.extract_slice %0[0, 0, 0] [1, 7200, 256] [1, 1, 1]
///      : tensor<1x9450x256xf32> to tensor<1x7200x256xf32>
/// %2 = tensor.extract_slice %0[0, 7200, 0] [1, 2250, 256] [1, 1, 1]
///      : tensor<1x9450x256xf32> to tensor<1x2250x256xf32>
/// ```
///
/// After:
/// ```mlir
/// %input0 = tensor.extract_slice %input[0, 0, 0] [1, 7200, 256] [1, 1, 1]
///           : tensor<1x9450x256xf32> to tensor<1x7200x256xf32>
/// %0 = "some.tiling_interface_op"(%input0) : (tensor<1x7200x256xf32>)
///                                            -> tensor<1x7200x256xf32>
/// %input1 = tensor.extract_slice %input[0, 7200, 0] [1, 2250, 256] [1, 1, 1]
///           : tensor<1x9450x256xf32> to tensor<1x2250x256xf32>
/// %1 = "some.tiling_interface_op"(%input1) : (tensor<1x2250x256xf32>)
///                                            -> tensor<1x2250x256xf32>
/// ```
struct BubbleUpExtractSliceThroughTilingInterface
    : public OpInterfaceRewritePattern<TilingInterface> {
  BubbleUpExtractSliceThroughTilingInterface(
      MLIRContext *context,
      function_ref<LogicalResult(TilingInterface, tensor::ExtractSliceOp)>
          controlFn = nullptr,
      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(TilingInterface producerOp,
                                PatternRewriter &rewriter) const override {
    // Only support operations with a single result for now.
    if (producerOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(producerOp,
                                         "expected single result operation");

    OpResult output = producerOp->getResult(0);
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(producerOp,
                                         "expected ranked tensor result");

    LLVM_DEBUG(llvm::dbgs()
               << "Checking TilingInterface op: " << *producerOp << "\n");

    // Collect all extract_slice users.
    SmallVector<tensor::ExtractSliceOp> extractSlices;
    for (Operation *user : output.getUsers()) {
      auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!extractSlice)
        return rewriter.notifyMatchFailure(
            producerOp, "result has non-extract_slice consumer");
      extractSlices.push_back(extractSlice);
    }

    // Sort slices by their position in the block to ensure deterministic
    // processing order. This maintains the correspondence between original
    // extract_slice ops and their replacements.
    llvm::sort(extractSlices,
               [](tensor::ExtractSliceOp a, tensor::ExtractSliceOp b) {
                 return a->isBeforeInBlock(b);
               });

    if (extractSlices.empty())
      return rewriter.notifyMatchFailure(producerOp,
                                         "no extract_slice consumers");

    LLVM_DEBUG(llvm::dbgs() << "Found " << extractSlices.size()
                            << " extract_slice consumers\n");

    // Check for overlapping slices when there are multiple consumers.
    // Overlapping slices would cause redundant computation.
    if (extractSlices.size() > 1) {
      FailureOr<bool> hasOverlaps =
          hasOverlappingSlices(rewriter.getContext(), extractSlices);
      if (failed(hasOverlaps))
        return rewriter.notifyMatchFailure(
            producerOp, "could not determine slice overlaps");
      if (*hasOverlaps)
        return rewriter.notifyMatchFailure(
            producerOp, "extract_slices have overlapping regions");

      LLVM_DEBUG(llvm::dbgs() << "No overlapping slices detected\n");
    }

    // Apply the control function to verify each slice is suitable for tiling.
    if (controlFn) {
      for (tensor::ExtractSliceOp extractSlice : extractSlices) {
        if (failed(controlFn(producerOp, extractSlice)))
          return rewriter.notifyMatchFailure(
              producerOp, "slice rejected by control function");
      }
    }

    // For each extract_slice, create a tiled producer.
    // Collect all results first before replacing to avoid iterator
    // invalidation.
    SmallVector<std::pair<tensor::ExtractSliceOp, Value>> replacements;
    for (tensor::ExtractSliceOp extractSlice : extractSlices) {
      FailureOr<TilingResult> tilingResult =
          tensor::replaceExtractSliceWithTiledProducer(rewriter, extractSlice,
                                                       output);
      if (failed(tilingResult))
        return rewriter.notifyMatchFailure(
            producerOp, "failed to generate tiled implementation");

      if (tilingResult->tiledValues.empty())
        return rewriter.notifyMatchFailure(producerOp,
                                           "tiling produced no values");

      replacements.emplace_back(extractSlice, tilingResult->tiledValues[0]);
    }

    // Replace all extract_slices with tiled values.
    for (auto &[extractSlice, tiledValue] : replacements) {
      rewriter.replaceOp(extractSlice, tiledValue);
      LLVM_DEBUG(llvm::dbgs() << "Replaced extract_slice with tiled value\n");
    }

    return success();
  }

private:
  /// Optional callback to control which slices should be transformed.
  /// Called for each extract_slice with the producer op and the slice op.
  /// Return success() to allow the transformation, failure() to skip.
  function_ref<LogicalResult(TilingInterface, tensor::ExtractSliceOp)>
      controlFn;
};

} // namespace

void mlir::tensor::populateBubbleUpExtractSliceThroughTilingInterfacePatterns(
    RewritePatternSet &patterns,
    function_ref<LogicalResult(TilingInterface, tensor::ExtractSliceOp)>
        controlFn) {
  patterns.add<BubbleUpExtractSliceThroughTilingInterface>(
      patterns.getContext(), controlFn);
}
