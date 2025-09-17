//===- SubgroupIdRewriter.cpp - Implementation of SubgroupId rewriting ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements in-dialect rewriting of the gpu.subgroup_id op for archs
// where:
// subgroup_id = (tid.x + dim.x * (tid.y + dim.y * tid.z)) / subgroup_size
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
struct GpuSubgroupIdRewriter final : OpRewritePattern<gpu::SubgroupIdOp> {
  using OpRewritePattern<gpu::SubgroupIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupIdOp op,
                                PatternRewriter &rewriter) const override {
    // Calculation of the thread's subgroup identifier.
    //
    // The process involves mapping the thread's 3D identifier within its
    // block (b_id.x, b_id.y, b_id.z) to a 1D linear index.
    // This linearization assumes a layout where the x-dimension (w_dim.x)
    // varies most rapidly (i.e., it is the innermost dimension).
    //
    // The formula for the linearized thread index is:
    // L = tid.x + dim.x * (tid.y + (dim.y * tid.z))
    //
    // Subsequently, the range of linearized indices [0, N_threads-1] is
    // divided into consecutive, non-overlapping segments, each representing
    // a subgroup of size 'subgroup_size'.
    //
    // Example Partitioning (N = subgroup_size):
    // | Subgroup 0      | Subgroup 1      | Subgroup 2      | ... |
    // | Indices 0..N-1  | Indices N..2N-1 | Indices 2N..3N-1| ... |
    //
    // The subgroup identifier is obtained via integer division of the
    // linearized thread index by the predefined 'subgroup_size'.
    //
    // subgroup_id = floor( L / subgroup_size )
    //             = (tid.x + dim.x * (tid.y + dim.y * tid.z)) /
    //             subgroup_size

    Location loc = op->getLoc();
    Type indexType = rewriter.getIndexType();

    Value dimX = gpu::BlockDimOp::create(rewriter, loc, gpu::Dimension::x);
    Value dimY = gpu::BlockDimOp::create(rewriter, loc, gpu::Dimension::y);
    Value tidX = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value tidY = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::y);
    Value tidZ = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::z);

    Value dimYxIdZ =
        arith::MulIOp::create(rewriter, loc, indexType, dimY, tidZ);
    Value dimYxIdZPlusIdY =
        arith::AddIOp::create(rewriter, loc, indexType, dimYxIdZ, tidY);
    Value dimYxIdZPlusIdYTimesDimX =
        arith::MulIOp::create(rewriter, loc, indexType, dimX, dimYxIdZPlusIdY);
    Value IdXPlusDimYxIdZPlusIdYTimesDimX = arith::AddIOp::create(
        rewriter, loc, indexType, tidX, dimYxIdZPlusIdYTimesDimX);
    Value subgroupSize = gpu::SubgroupSizeOp::create(
        rewriter, loc, rewriter.getIndexType(), /*upper_bound = */ nullptr);
    Value subgroupIdOp =
        arith::DivUIOp::create(rewriter, loc, indexType,
                               IdXPlusDimYxIdZPlusIdYTimesDimX, subgroupSize);
    rewriter.replaceOp(op, {subgroupIdOp});
    return success();
  }
};

} // namespace

void mlir::populateGpuSubgroupIdPatterns(RewritePatternSet &patterns) {
  patterns.add<GpuSubgroupIdRewriter>(patterns.getContext());
}
