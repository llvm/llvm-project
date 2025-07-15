//===- ResolveStridedMetadata.cpp - AMDGPU expand_strided_metadata ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::amdgpu {
#define GEN_PASS_DEF_AMDGPURESOLVESTRIDEDMETADATAPASS
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h.inc"
} // namespace mlir::amdgpu

using namespace mlir;
using namespace mlir::amdgpu;

namespace {
struct AmdgpuResolveStridedMetadataPass
    : public amdgpu::impl::AmdgpuResolveStridedMetadataPassBase<
          AmdgpuResolveStridedMetadataPass> {
  void runOnOperation() override;
};

struct ExtractStridedMetadataOnFatRawBufferCastFolder final
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp metadataOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = metadataOp.getSource().getDefiningOp<FatRawBufferCastOp>();
    if (!castOp)
      return rewriter.notifyMatchFailure(metadataOp,
                                         "not a fat raw buffer cast");
    Location loc = castOp.getLoc();
    auto sourceMetadata = rewriter.create<memref::ExtractStridedMetadataOp>(
        loc, castOp.getSource());
    SmallVector<Value> results;
    if (metadataOp.getBaseBuffer().use_empty()) {
      results.push_back(nullptr);
    } else {
      auto baseBufferType =
          cast<MemRefType>(metadataOp.getBaseBuffer().getType());
      if (baseBufferType == castOp.getResult().getType()) {
        results.push_back(castOp.getResult());
      } else {
        results.push_back(rewriter.create<memref::ReinterpretCastOp>(
            loc, baseBufferType, castOp.getResult(), /*offset=*/0,
            /*sizes=*/ArrayRef<int64_t>{}, /*strides=*/ArrayRef<int64_t>{}));
      }
    }
    if (castOp.getResetOffset())
      results.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    else
      results.push_back(sourceMetadata.getOffset());
    llvm::append_range(results, sourceMetadata.getSizes());
    llvm::append_range(results, sourceMetadata.getStrides());
    rewriter.replaceOp(metadataOp, results);
    return success();
  }
};
} // namespace

void mlir::amdgpu::populateAmdgpuResolveStridedMetadataPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ExtractStridedMetadataOnFatRawBufferCastFolder>(
      patterns.getContext(), benefit);
}

void AmdgpuResolveStridedMetadataPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateAmdgpuResolveStridedMetadataPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
