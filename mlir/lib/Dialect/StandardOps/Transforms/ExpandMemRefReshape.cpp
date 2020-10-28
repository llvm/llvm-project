//===- ExpandMemRefReshape.cpp - Code to perform expanding memref_reshape -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of MemRefReshapeOp into
// MemRefReinterpretCastOp.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Converts `memref_reshape` that has a target shape of a statically-known
/// size to `memref_reinterpret_cast`.
struct MemRefReshapeOpConverter : public OpRewritePattern<MemRefReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefReshapeOp op,
                                PatternRewriter &rewriter) const final {
    auto shapeType = op.shape().getType().cast<MemRefType>();
    if (!shapeType.hasStaticShape())
      return failure();

    int64_t rank = shapeType.cast<MemRefType>().getDimSize(0);
    SmallVector<Value, 4> sizes, strides;
    sizes.resize(rank);
    strides.resize(rank);

    Location loc = op.getLoc();
    Value stride = rewriter.create<ConstantIndexOp>(loc, 1);
    for (int i = rank - 1; i >= 0; --i) {
      Value index = rewriter.create<ConstantIndexOp>(loc, i);
      Value size = rewriter.create<LoadOp>(loc, op.shape(), index);
      if (!size.getType().isa<IndexType>())
        size = rewriter.create<IndexCastOp>(loc, size, rewriter.getIndexType());
      sizes[i] = size;
      strides[i] = stride;
      if (i > 0)
        stride = rewriter.create<MulIOp>(loc, stride, size);
    }
    SmallVector<int64_t, 2> staticSizes(rank, ShapedType::kDynamicSize);
    SmallVector<int64_t, 2> staticStrides(rank,
                                          ShapedType::kDynamicStrideOrOffset);
    rewriter.replaceOpWithNewOp<MemRefReinterpretCastOp>(
        op, op.getType(), op.source(), /*staticOffset = */ 0, staticSizes,
        staticStrides, /*offset=*/llvm::None, sizes, strides);
    return success();
  }
};

} // namespace

void mlir::populateExpandMemRefReshapePattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<MemRefReshapeOpConverter>(ctx);
}
