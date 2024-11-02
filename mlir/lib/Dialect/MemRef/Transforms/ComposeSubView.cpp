//===- ComposeSubView.cpp - Combining composed subview ops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns for combining composed subview ops (i.e. subview
// of a subview becomes a single subview).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Replaces a subview of a subview with a single subview. Only supports subview
// ops with static sizes and static strides of 1 (both static and dynamic
// offsets are supported).
struct ComposeSubViewOpPattern : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    // 'op' is the 'SubViewOp' we're rewriting. 'sourceOp' is the op that
    // produces the input of the op we're rewriting (for 'SubViewOp' the input
    // is called the "source" value). We can only combine them if both 'op' and
    // 'sourceOp' are 'SubViewOp'.
    auto sourceOp = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!sourceOp)
      return failure();

    // A 'SubViewOp' can be "rank-reducing" by eliminating dimensions of the
    // output memref that are statically known to be equal to 1. We do not
    // allow 'sourceOp' to be a rank-reducing subview because then our two
    // 'SubViewOp's would have different numbers of offset/size/stride
    // parameters (just difficult to deal with, not impossible if we end up
    // needing it).
    if (sourceOp.getSourceType().getRank() != sourceOp.getType().getRank()) {
      return failure();
    }

    // Offsets, sizes and strides OpFoldResult for the combined 'SubViewOp'.
    SmallVector<OpFoldResult> offsets, sizes, strides;

    // Because we only support input strides of 1, the output stride is also
    // always 1.
    if (llvm::all_of(strides, [](OpFoldResult &valueOrAttr) {
          Attribute attr = valueOrAttr.dyn_cast<Attribute>();
          return attr && attr.cast<IntegerAttr>().getInt() == 1;
        })) {
      strides = SmallVector<OpFoldResult>(sourceOp.getMixedStrides().size(),
                                          rewriter.getI64IntegerAttr(1));
    } else {
      return failure();
    }

    // The rules for calculating the new offsets and sizes are:
    // * Multiple subview offsets for a given dimension compose additively.
    //   ("Offset by m" followed by "Offset by n" == "Offset by m + n")
    // * Multiple sizes for a given dimension compose by taking the size of the
    //   final subview and ignoring the rest. ("Take m values" followed by "Take
    //   n values" == "Take n values") This size must also be the smallest one
    //   by definition (a subview needs to be the same size as or smaller than
    //   its source along each dimension; presumably subviews that are larger
    //   than their sources are disallowed by validation).
    for (auto it : llvm::zip(op.getMixedOffsets(), sourceOp.getMixedOffsets(),
                             op.getMixedSizes())) {
      auto opOffset = std::get<0>(it);
      auto sourceOffset = std::get<1>(it);
      auto opSize = std::get<2>(it);

      // We only support static sizes.
      if (opSize.is<Value>()) {
        return failure();
      }

      sizes.push_back(opSize);
      Attribute opOffsetAttr = opOffset.dyn_cast<Attribute>(),
                sourceOffsetAttr = sourceOffset.dyn_cast<Attribute>();

      if (opOffsetAttr && sourceOffsetAttr) {
        // If both offsets are static we can simply calculate the combined
        // offset statically.
        offsets.push_back(rewriter.getI64IntegerAttr(
            opOffsetAttr.cast<IntegerAttr>().getInt() +
            sourceOffsetAttr.cast<IntegerAttr>().getInt()));
      } else {
        // When either offset is dynamic, we must emit an additional affine
        // transformation to add the two offsets together dynamically.
        AffineExpr expr = rewriter.getAffineConstantExpr(0);
        SmallVector<Value> affineApplyOperands;
        for (auto valueOrAttr : {opOffset, sourceOffset}) {
          if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
            expr = expr + attr.cast<IntegerAttr>().getInt();
          } else {
            expr =
                expr + rewriter.getAffineSymbolExpr(affineApplyOperands.size());
            affineApplyOperands.push_back(valueOrAttr.get<Value>());
          }
        }

        AffineMap map = AffineMap::get(0, affineApplyOperands.size(), expr);
        Value result = rewriter.create<AffineApplyOp>(op.getLoc(), map,
                                                      affineApplyOperands);
        offsets.push_back(result);
      }
    }

    // This replaces 'op' but leaves 'sourceOp' alone; if it no longer has any
    // uses it can be removed by a (separate) dead code elimination pass.
    rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, sourceOp.getSource(),
                                                   offsets, sizes, strides);
    return success();
  }
};

} // namespace

void mlir::memref::populateComposeSubViewPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<ComposeSubViewOpPattern>(context);
}
