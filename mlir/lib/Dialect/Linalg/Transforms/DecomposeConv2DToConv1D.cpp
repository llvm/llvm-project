//===- DecomposeConv2DToConv1D.cpp ---------------------- -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Converts a conv2d into a series of conv1d ops using row-wise decomposition
/// (also known as shift-and-add)

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGDECOMPOSECONV2DTOCONV1D
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Decomposes a linalg.conv_2d_nhwc_hwcf op into a sequence of
/// linalg.conv_1d_nwc_wcf ops using a shift-and-add approach.
///
/// Constraints:
/// - Height stride and dilation must be 1 (to allow contiguous reshaping).
/// - Width stride and dilation are preserved in the 1D convolution.
struct DecomposeConv2DToConv1DPattern final : public OpRewritePattern<Conv2DNhwcHwcfOp> {
  using OpRewritePattern<Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();
    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto inputType = cast<RankedTensorType>(input.getType());
    auto filterType = cast<RankedTensorType>(filter.getType());

    // 1. Validate Strides and Dilations
    // We only support Stride_H = 1 and Dilation_H = 1 for this specific
    // reshape-based decomposition.
    auto stridesAttr = convOp.getStrides();
    auto dilationsAttr = convOp.getDilations();

    SmallVector<int64_t> strides = llvm::to_vector(stridesAttr.getValues<int64_t>());
    SmallVector<int64_t> dilations = llvm::to_vector(dilationsAttr.getValues<int64_t>());

    if (strides[0] != 1 || dilations[0] != 1) {
      return rewriter.notifyMatchFailure(convOp, "requires stride_h=1 and dilation_h=1");
    }

    // 2. Get Dimensions
    // Input: [N, H, W, C_in]
    // Filter: [Kh, Kw, C_in, C_out]
    // Output: [N, H_out, W_out, C_out]

    // Helper to get a Value for a dimension size (static or dynamic)
    auto getDim = [&](Value v, int64_t idx) -> Value {
      return tensor::DimOp::create(rewriter, loc, v, idx);
    };

    Value N = getDim(input, 0);
    Value H_out = getDim(output, 1);
    Value W_in = getDim(input, 2);
    Value C_in = getDim(input, 3);

    Value Kh = getDim(filter, 0);
    Value Kw = getDim(filter, 1);
    Value C_out = getDim(filter, 3);

    // 3. Iterate over the Kernel Height (Kh)
    // We will accumulate results into 'output'.
    // Lower bound = 0, Upper bound = Kh, Step = 1
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto scfLoop = scf::ForOp::create(rewriter,
        loc, zero, Kh, one, ValueRange{output},
        [&](OpBuilder &b, Location loc, Value r, ValueRange args) {
          Value currentAccumulator = args[0];

          // --- A. Extract Filter Slice ---
          // Filter shape: [Kh, Kw, Cin, Cout] -> Slice at r: [1, Kw, Cin, Cout]
          // We need to rank-reduce this to [Kw, Cin, Cout] for conv_1d.
          SmallVector<OpFoldResult> filterOffsets = {r, b.getIndexAttr(0), b.getIndexAttr(0), b.getIndexAttr(0)};
          SmallVector<OpFoldResult> filterSizes = {b.getIndexAttr(1), Kw, C_in, C_out};
          SmallVector<OpFoldResult> filterStrides = {b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1)};

          // Explicitly specify the desired result type (Rank 3)
          auto filterSliceType = RankedTensorType::get(
              {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic},
              filterType.getElementType());

          Value filterSlice = tensor::ExtractSliceOp::create(b,
              loc, filterSliceType, filter, filterOffsets, filterSizes, filterStrides);

          // --- B. Extract Input Slice ---
          // We need a view of the input shifted by 'r' along Height.
          // Input: [N, H, W, C]. Slice starts at [0, r, 0, 0].
          // Size: [N, H_out, W, C].
          // (Recall H_in = H_out + Kh - 1 generally, so H_out fits starting at r).
          SmallVector<OpFoldResult> inputOffsets = {b.getIndexAttr(0), r, b.getIndexAttr(0), b.getIndexAttr(0)};
          SmallVector<OpFoldResult> inputSizes = {N, H_out, W_in, C_in};
          SmallVector<OpFoldResult> inputStrides = {b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1)};

          Value inputSlice = tensor::ExtractSliceOp::create(b,
              loc, input, inputOffsets, inputSizes, inputStrides);

          // --- C. Reshape Input for Conv1D ---
          // Conv1D expects [Batch, Width, Channels].
          // We have [N, H_out, W_in, C_in].
          // We collapse N and H_out into a single Batch dimension.
          SmallVector<ReassociationIndices> collapseIndicesInput = {{0, 1}, {2}, {3}};
          Value reshapedInput = tensor::CollapseShapeOp::create(b,
              loc, inputSlice, collapseIndicesInput);

          // --- D. Reshape Accumulator for Conv1D ---
          // Current Accumulator: [N, H_out, W_out, C_out].
          // Target: [N * H_out, W_out, C_out].
          Value reshapedAcc = tensor::CollapseShapeOp::create(b,
              loc, currentAccumulator, collapseIndicesInput);

          // --- E. Perform Conv1D ---
          // Op: linalg.conv_1d_nwc_wcf
          // Strides and Dilations for W are passed through from the original Op.
          // Original Strides: [Stride_H, Stride_W]. We take Stride_W.
          auto strideW = strides[1];
          auto dilationW = dilations[1];

          auto conv1d = Conv1DNwcWcfOp::create(b, loc,
            TypeRange{reshapedAcc.getType()},
              ValueRange{reshapedInput, filterSlice},
              ValueRange{reshapedAcc},
              b.getDenseI64ArrayAttr({strideW}),
              b.getDenseI64ArrayAttr({dilationW}));

          // --- F. Expand Result back to 4D ---
          // Result: [N * H_out, W_out, C_out] -> [N, H_out, W_out, C_out]
          // We use the Type of the currentAccumulator to ensure correct dynamic dim reconstruction.
          Value expandedResult = tensor::ExpandShapeOp::create(b,
              loc, currentAccumulator.getType(), conv1d.getResult(0), collapseIndicesInput);

          scf::YieldOp::create(b, loc, expandedResult);
        });

    rewriter.replaceOp(convOp, scfLoop.getResult(0));
    return success();
  }
};

} // namespace

struct LinalgDecomposeConv2DtoConv1D final : public impl::LinalgDecomposeConv2DToConv1DBase<LinalgDecomposeConv2DtoConv1D> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<DecomposeConv2DToConv1DPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};