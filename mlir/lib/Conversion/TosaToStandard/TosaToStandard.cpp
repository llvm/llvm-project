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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        sliceOp, sliceOp.getType(), input, ValueRange({}), ValueRange({}),
        ValueRange({}), sliceOp.start(), sliceOp.size(),
        rewriter.getI64ArrayAttr(strides));
    return success();
  }
};

// This converts the TOSA ApplyScale operator to a set of StandardOps ops,
// using 64-bit operations to perform the necessary multiply, bias, and shift.
// Multiple types are used to use minimal bit width operations.
class ApplyScaleOpConverter : public OpRewritePattern<tosa::ApplyScaleOp> {
public:
  using OpRewritePattern<tosa::ApplyScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ApplyScaleOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value value32 = op.value();
    Value multiplier32 = op.multiplier();
    Value shift8 = op.shift();
    bool doubleRound = op.double_round();
    Type inType = op.value().getType();

    Value one8 = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(8), 1));
    Value one64 = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 1));

    Value shiftSubOne8 = rewriter.create<SubIOp>(loc, shift8, one8);

    // The rounding value semantics below equate to the following code:
    //    int64_t round = 1 << (shift - 1);
    //    if (double_round) {
    //      if (shift > 31 && value >= 0) round += 1<<30;
    //      if (shift > 31 && value < 0) round -= 1<<30;
    //    }
    //
    // Note that minimal bitwidth operators are used throughout the block.

    Value round64 = rewriter.create<mlir::ShiftLeftOp>(
        loc, one64,
        rewriter.create<SignExtendIOp>(loc, rewriter.getI64Type(),
                                       shiftSubOne8));

    // Double rounding is performing a round operation before the shift
    if (doubleRound) {
      Value one32 = rewriter.create<ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getI32Type(), 1));
      Value shift32 = rewriter.create<mlir::SignExtendIOp>(
          loc, rewriter.getI32Type(), shift8);
      Value thirty32 = rewriter.create<ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getI32Type(), 30));

      Value shiftThirty32 =
          rewriter.create<mlir::ShiftLeftOp>(loc, one32, thirty32);
      Value shiftThirty64 = rewriter.create<mlir::SignExtendIOp>(
          loc, rewriter.getI64Type(), shiftThirty32);

      // Round value needs to with be added or subtracted depending on the sign
      // of the input value.
      Value roundAdd64 =
          rewriter.create<mlir::AddIOp>(loc, round64, shiftThirty64);
      Value roundSub64 =
          rewriter.create<mlir::SubIOp>(loc, round64, shiftThirty64);

      Value zero32 =
          rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(inType));
      Value valueGreaterThanZero = rewriter.create<mlir::CmpIOp>(
          loc, CmpIPredicate::sge, value32, zero32);

      Value doubleRound64 = rewriter.create<mlir::SelectOp>(
          loc, valueGreaterThanZero, roundAdd64, roundSub64);

      // We only perform double rounding if the shift value is greater than 32.
      Value thirtyTwo32 = rewriter.create<ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getI32Type(), 32));
      Value shiftGreaterThanThirtyTwo = rewriter.create<mlir::CmpIOp>(
          loc, CmpIPredicate::sge, shift32, thirtyTwo32);
      round64 = rewriter.create<mlir::SelectOp>(loc, shiftGreaterThanThirtyTwo,
                                                doubleRound64, round64);
    }

    // The computation below equates to the following pseudocode:
    //    int64_t result = (int64_t)value * multiplier + round;
    //    result = result >> shift;
    //
    // Note that multiply and shift need to be perform in i64 to preserve bits.

    Value value64 =
        rewriter.create<SignExtendIOp>(loc, rewriter.getI64Type(), value32);
    Value multiplier64 = rewriter.create<SignExtendIOp>(
        loc, rewriter.getI64Type(), multiplier32);
    Value shift64 =
        rewriter.create<SignExtendIOp>(loc, rewriter.getI64Type(), shift8);

    // Multiply as a pair of i64 values to guarantee the end value fits.
    Value result64 = rewriter.create<MulIOp>(loc, value64, multiplier64);
    result64 = rewriter.create<AddIOp>(loc, result64, round64);
    result64 =
        rewriter.create<mlir::SignedShiftRightOp>(loc, result64, shift64);

    Value result32 = rewriter.create<mlir::TruncateIOp>(
        loc, rewriter.getI32Type(), result64);

    rewriter.replaceOp(op, result32);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToStandardConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<ApplyScaleOpConverter, ConstOpConverter, SliceOpConverter>(
      patterns->getContext());
}

void mlir::tosa::populateTosaRescaleToStandardConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<ApplyScaleOpConverter>(patterns->getContext());
}
