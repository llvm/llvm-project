//===- PromoteShuffleToAMDGPU.cpp - Promote shuffle to AMDGPU -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to try to promote `gpu.shuffle`s to specialized
// AMDGPU intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
/// Try to promote `gpu.shuffle` to `amdgpu.swizzle_bitmode`, width must be 64
/// and offset must be a constant integer in the range [0, 31].
struct PromoteShuffleToSwizzlePattern
    : public OpRewritePattern<gpu::ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getMode() != gpu::ShuffleMode::XOR)
      return rewriter.notifyMatchFailure(op,
                                         "only xor shuffle mode is supported");

    if (!isConstantIntValue(op.getWidth(), 64))
      return rewriter.notifyMatchFailure(op,
                                         "only 64 width shuffle is supported");

    std::optional<int64_t> offset = getConstantIntValue(op.getOffset());
    if (!offset)
      return rewriter.notifyMatchFailure(op,
                                         "offset must be a constant integer");

    int64_t offsetValue = *offset;
    if (offsetValue < 0 || offsetValue >= 32)
      return rewriter.notifyMatchFailure(op,
                                         "offset must be in the range [0, 31]");

    Location loc = op.getLoc();
    Value res = amdgpu::SwizzleBitModeOp::create(
        rewriter, loc, op.getResult(0).getType(), op.getValue(), /*andMask=*/31,
        /*orMask=*/0, /*xorMask=*/offsetValue);
    Value valid = arith::ConstantIntOp::create(rewriter, loc, 1, /*width*/ 1);
    rewriter.replaceOp(op, {res, valid});
    return success();
  }
};
} // namespace

void mlir::populateGpuPromoteShuffleToAMDGPUPatterns(
    RewritePatternSet &patterns) {
  patterns.add<PromoteShuffleToSwizzlePattern>(patterns.getContext());
}
