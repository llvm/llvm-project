//===- XeGPUFoldAliasOps.cpp - XeGPU alias ops folders ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
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

  SmallVector<Value> resolvedOffsets;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, descOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(),
      descOp.getMixedOffsets(), resolvedOffsets);

  rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
      descOp, descOp.getTensorDesc().getType(), subViewOp.getSource(),
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
