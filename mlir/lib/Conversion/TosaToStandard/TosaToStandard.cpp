//===- TosaToStandard.cpp - Lowering Tosa to Standard Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Standard dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToStandard/TosaToStandard.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace tosa;

namespace {

class ConstOpConverter : public OpRewritePattern<tosa::ConstOp> {
public:
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<::ConstantOp>(op, op.value());
    return success();
  }
};

class SliceOpConverter : public OpRewritePattern<tosa::SliceOp> {
public:
  using OpRewritePattern<tosa::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SliceOp sliceOp,
                                PatternRewriter &rewriter) const final {
    Value input = sliceOp.input();
    SmallVector<int64_t> strides;
    strides.resize(sliceOp.getType().template cast<ShapedType>().getRank(), 1);

    rewriter.replaceOpWithNewOp<SubTensorOp>(
        sliceOp, sliceOp.getType(), input, ValueRange({}), ValueRange({}),
        ValueRange({}), sliceOp.start(), sliceOp.size(),
        rewriter.getI64ArrayAttr(strides));

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToStandardConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<ConstOpConverter, SliceOpConverter>(context);
}
