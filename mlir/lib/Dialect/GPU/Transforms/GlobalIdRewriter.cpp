//===- GlobalIdRewriter.cpp - Implementation of GlobalId rewriting  -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements in-dialect rewriting of the global_id op for archs
// where global_id.x = threadId.x + blockId.x * blockDim.x
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct GpuGlobalIdRewriter : public OpRewritePattern<gpu::GlobalIdOp> {
  using OpRewritePattern<gpu::GlobalIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::GlobalIdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dim = op.getDimension();
    auto blockId = rewriter.create<gpu::BlockIdOp>(loc, dim);
    auto blockDim = rewriter.create<gpu::BlockDimOp>(loc, dim);
    // Compute blockId.x * blockDim.x
    auto tmp = rewriter.create<index::MulOp>(op.getLoc(), blockId, blockDim);
    auto threadId = rewriter.create<gpu::ThreadIdOp>(loc, dim);
    // Compute threadId.x + blockId.x * blockDim.x
    rewriter.replaceOpWithNewOp<index::AddOp>(op, threadId, tmp);
    return success();
  }
};
} // namespace

void mlir::populateGpuGlobalIdPatterns(RewritePatternSet &patterns) {
  patterns.add<GpuGlobalIdRewriter>(patterns.getContext());
}
