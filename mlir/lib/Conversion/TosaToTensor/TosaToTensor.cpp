//===- TosaToTensor.cpp - Lowering Tosa to Tensor Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace tosa;

namespace {

class SliceOpConverter : public OpRewritePattern<tosa::SliceOp> {
public:
  using OpRewritePattern<tosa::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SliceOp sliceOp,
                                PatternRewriter &rewriter) const final {
    Location loc = sliceOp.getLoc();
    Value input = sliceOp.getInput();
    SmallVector<int64_t> strides;
    auto starts = sliceOp.getStart();
    auto sizes = sliceOp.getSize();
    strides.resize(sliceOp.getType().template cast<ShapedType>().getRank(), 1);

    SmallVector<Value> dynSizes;
    for (const auto &i : llvm::enumerate(sizes)) {
      int64_t size = i.value().cast<IntegerAttr>().getInt();
      size_t index = i.index();
      if (size != ShapedType::kDynamicSize)
        continue;

      auto dim = rewriter.create<tensor::DimOp>(loc, input, index);
      auto offset = rewriter.create<arith::ConstantOp>(
          loc,
          rewriter.getIndexAttr(starts[index].cast<IntegerAttr>().getInt()));
      dynSizes.push_back(rewriter.create<arith::SubIOp>(loc, dim, offset));
    }

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), input, ValueRange({}), dynSizes,
        ValueRange({}), starts, sizes, rewriter.getI64ArrayAttr(strides));

    rewriter.replaceOp(sliceOp, newSliceOp.getResult());
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToTensorConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<SliceOpConverter>(patterns->getContext());
}
