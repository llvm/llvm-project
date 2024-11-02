//===- SplitPaddingPatterns.cpp - Splitting tensor.pad Op -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to wrap a tensor.pad op with an scf.if op
/// to separate the cases where we don't need padding (all pad sizes are
/// actually zeros) and where we indeed need padding.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-tensor-split-padding"

using namespace mlir;

/// Returns true if the the given `attrOrValue` is a constant zero.
static bool isZero(OpFoldResult attrOrValue) {
  if (Optional<int64_t> val = getConstantIntValue(attrOrValue))
    return *val == 0;
  return false;
}

/// Gets the given `attrOrValue` as a Value by creating constant ops for
/// attributes.
static Value getAsValue(OpFoldResult attrOrValue, OpBuilder &builder,
                        Location loc) {
  if (Value val = attrOrValue.dyn_cast<Value>())
    return val;
  auto attr = attrOrValue.get<Attribute>().cast<IntegerAttr>();
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
}

namespace {

struct SplitPadding final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Avoid infinitely applying this pattern.
    if (padOp->getParentOfType<scf::IfOp>())
      return failure();

    // If all padding sizes are zero, we don't need to do anything.
    SmallVector<OpFoldResult> lowPads = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPads = padOp.getMixedHighPad();
    if (llvm::all_of(lowPads, isZero) && llvm::all_of(highPads, isZero))
      return failure();

    // Build the condition for the scf.if op: all pad sizes are zero.
    Location loc = padOp.getLoc();
    Value cstZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> eqZeroCmpVals;
    for (OpFoldResult pad : llvm::concat<OpFoldResult>(lowPads, highPads)) {
      if (!isZero(pad))
        eqZeroCmpVals.push_back(rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, getAsValue(pad, rewriter, loc),
            cstZero));
    }
    Value ifCond = eqZeroCmpVals.front();
    for (Value cmp : llvm::makeArrayRef(eqZeroCmpVals).drop_front())
      ifCond = rewriter.create<arith::AndIOp>(loc, ifCond, cmp);

    // Build the scf.if op itself. For the "then" branch, we can elide the
    // padding. For the "else" branch, we retain the clone op.
    auto thenBuilder = [&padOp](OpBuilder &builder, Location loc) {
      builder.create<scf::YieldOp>(loc, padOp.getSource());
    };
    auto elseBuilder = [&padOp](OpBuilder &builder, Location loc) {
      Operation *newOp = builder.clone(*padOp);
      builder.create<scf::YieldOp>(loc, newOp->getResults());
    };
    rewriter.replaceOpWithNewOp<scf::IfOp>(padOp, padOp.getType(), ifCond,
                                           thenBuilder, elseBuilder);
    return success();
  }
};

} // namespace

void tensor::populateSplitPaddingPatterns(RewritePatternSet &patterns,
                                          PatternBenefit baseBenefit) {
  patterns.add<SplitPadding>(patterns.getContext(), baseBenefit);
}
