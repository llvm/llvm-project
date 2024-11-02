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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    SmallVector<int64_t> strides, sizes, starts;
    starts = extractFromI64ArrayAttr(sliceOp.getStart());
    strides.resize(sliceOp.getType().template cast<ShapedType>().getRank(), 1);

    SmallVector<Value> dynSizes;
    for (const auto &i : llvm::enumerate(sliceOp.getSize())) {
      int64_t size = i.value().cast<IntegerAttr>().getInt();
      size_t index = i.index();
      sizes.push_back(size == -1 ? ShapedType::kDynamic : size);
      if (!ShapedType::isDynamic(sizes.back()))
        continue;

      auto dim = rewriter.create<tensor::DimOp>(loc, input, index);
      auto offset = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(starts[index]));
      dynSizes.push_back(rewriter.create<arith::SubIOp>(loc, dim, offset));
    }

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), input, ValueRange({}), dynSizes,
        ValueRange({}), rewriter.getDenseI64ArrayAttr(starts),
        rewriter.getDenseI64ArrayAttr(sizes),
        rewriter.getDenseI64ArrayAttr(strides));

    rewriter.replaceOp(sliceOp, newSliceOp.getResult());
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToTensorConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<SliceOpConverter>(patterns->getContext());
}
