//===- ShapeToSCF.cpp - conversion from Shape to SCF dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToSCF/ShapeToSCF.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::shape;

namespace {
/// Converts `shape.reduce` to `scf.for`.
struct ReduceOpConverter : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
ReduceOpConverter::matchAndRewrite(ReduceOp reduceOp,
                                   PatternRewriter &rewriter) const {
  auto loc = reduceOp.getLoc();

  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  Value extentTensor = rewriter.create<ToExtentTensorOp>(
      loc,
      RankedTensorType::get({ShapedType::kDynamicSize},
                            rewriter.getIndexType()),
      reduceOp.shape());
  Value size =
      rewriter.create<DimOp>(loc, rewriter.getIndexType(), extentTensor, zero);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, size, one, reduceOp.initVals(),
      [&](OpBuilder &b, Location nestedLoc, Value iv, ValueRange args) {
        Value indexExtent = b.create<ExtractElementOp>(loc, extentTensor, iv);
        Value sizeExtent = b.create<IndexToSizeOp>(loc, indexExtent);

        SmallVector<Value, 2> mapped_values{iv, sizeExtent};
        mapped_values.append(args.begin(), args.end());

        BlockAndValueMapping mapping;
        Block *reduceBody = reduceOp.getBody();
        mapping.map(reduceBody->getArguments(), mapped_values);
        for (auto &nested : reduceBody->without_terminator())
          b.clone(nested, mapping);

        SmallVector<Value, 2> mappedResults;
        for (auto result : reduceBody->getTerminator()->getOperands())
          mappedResults.push_back(mapping.lookup(result));
        b.create<scf::YieldOp>(loc, mappedResults);
      });

  rewriter.replaceOp(reduceOp, loop.getResults());
  return success();
}

namespace {
struct ConvertShapeToSCFPass
    : public ConvertShapeToSCFBase<ConvertShapeToSCFPass> {
  void runOnFunction() override;
};
} // namespace

void ConvertShapeToSCFPass::runOnFunction() {
  MLIRContext &ctx = getContext();

  OwningRewritePatternList patterns;
  populateShapeToSCFConversionPatterns(patterns, &ctx);

  ConversionTarget target(getContext());
  target.addLegalDialect<ShapeDialect, scf::SCFDialect, StandardOpsDialect>();
  target.addIllegalOp<ReduceOp>();
  if (failed(mlir::applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

void mlir::populateShapeToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ReduceOpConverter>(ctx);
}

std::unique_ptr<FunctionPass> mlir::createConvertShapeToSCFPass() {
  return std::make_unique<ConvertShapeToSCFPass>();
}
