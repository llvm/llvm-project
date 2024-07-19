//===- XeGPUFoldAliasOps.cpp - XeGPU alias ops folders ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUFOLDALIASOPS
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-fold-alias-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {
/// Merges subview operation with xegpu.create_nd_tdesc operation.
class XegpuCreateNdDescOpSubViewOpFolder final
    : public OpRewritePattern<xegpu::CreateNdDescOp> {
public:
  using OpRewritePattern<xegpu::CreateNdDescOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp descOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult XegpuCreateNdDescOpSubViewOpFolder::matchAndRewrite(
    xegpu::CreateNdDescOp descOp, PatternRewriter &rewriter) const {
  auto subViewOp = descOp.getSource().getDefiningOp<memref::SubViewOp>();

  if (!subViewOp)
    return rewriter.notifyMatchFailure(descOp, "not a subview producer");
  if (!subViewOp.hasUnitStride())
    return rewriter.notifyMatchFailure(descOp, "requires unit strides");
  if (!subViewOp.getSource().getType().hasStaticShape())
    return rewriter.notifyMatchFailure(descOp, "requires static shape");

  SmallVector<Value> resolvedOffsets;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, descOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(),
      descOp.getMixedOffsets(), resolvedOffsets);

  auto updatedSource = subViewOp.getSource();
  // If the source memref rank is greater than 2, we need to cast the source to
  // 2D and compute the height, width offsets relative to that.
  if (resolvedOffsets.size() > 2) {
    // Cast the source to 2D. This will become the new source.
    auto sourceTy = subViewOp.getSource().getType();
    int64_t newWidth = sourceTy.getShape().back();
    int64_t newHeight = 1;
    for (int64_t dim : sourceTy.getShape().drop_back())
      newHeight *= dim;
    auto newSourceTy =
        MemRefType::get({newHeight, newWidth}, sourceTy.getElementType());
    int64_t offset = 0;
    updatedSource = rewriter.create<memref::ReinterpretCastOp>(
        descOp.getLoc(), newSourceTy, subViewOp.getSource(), offset,
        llvm::SmallVector<int64_t>({newHeight, newWidth}),
        llvm::SmallVector<int64_t>({newWidth, 1}));
    // Get source strides.
    llvm::SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;
    std::tie(sourceStrides, sourceOffset) = mlir::getStridesAndOffset(sourceTy);
    // Compute height offset.
    mlir::Value heightOffset = resolvedOffsets[resolvedOffsets.size() - 2];
    for (int64_t i = resolvedOffsets.size() - 3; i >= 0; --i) {
      auto constStrideOp = rewriter.create<arith::ConstantIndexOp>(
          descOp.getLoc(), sourceStrides[i]);
      auto mulOp = rewriter.create<arith::MulIOp>(
          descOp.getLoc(), resolvedOffsets[i], constStrideOp);
      heightOffset =
          rewriter.create<arith::AddIOp>(descOp.getLoc(), mulOp, heightOffset);
    }
    resolvedOffsets = {heightOffset, resolvedOffsets.back()};
  }

  rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
      descOp, descOp.getTensorDesc().getType(), updatedSource,
      getAsOpFoldResult(resolvedOffsets));

  return success();
}

void xegpu::populateXeGPUFoldAliasOpsPatterns(RewritePatternSet &patterns) {
  patterns.add<XegpuCreateNdDescOpSubViewOpFolder>(patterns.getContext());
}

namespace {

struct XeGPUFoldAliasOpsPass final
    : public xegpu::impl::XeGPUFoldAliasOpsBase<XeGPUFoldAliasOpsPass> {
  void runOnOperation() override;
};

} // namespace

void XeGPUFoldAliasOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  xegpu::populateXeGPUFoldAliasOpsPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
