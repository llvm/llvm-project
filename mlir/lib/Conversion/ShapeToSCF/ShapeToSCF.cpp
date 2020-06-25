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
/// Converts `shape_of` to for loop for unranked tensors.
class ShapeOfOpConverter : public OpConversionPattern<ShapeOfOp> {
public:
  using OpConversionPattern<ShapeOfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeOfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ShapeOfOpConverter::matchAndRewrite(ShapeOfOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  ShapeOfOp::Adaptor transformed(operands);
  auto tensorVal = transformed.arg();
  auto tensorTy = tensorVal.getType();

  // For ranked tensors `shape_of` lowers to `std` and the pattern can be
  // found in the corresponding pass.
  if (tensorTy.isa<RankedTensorType>())
    return failure();

  // Allocate stack memory.
  auto loc = op.getLoc();
  auto rankVal = rewriter.create<RankOp>(loc, tensorVal);
  auto i64Ty = rewriter.getI64Type();
  auto memTy = MemRefType::get({ShapedType::kDynamicSize}, i64Ty);
  auto memVal = rewriter.create<AllocaOp>(loc, memTy, ValueRange({rankVal}));

  // Copy shape extents to stack-allocated memory.
  auto zeroVal = rewriter.create<ConstantIndexOp>(loc, 0);
  auto oneVal = rewriter.create<ConstantIndexOp>(loc, 1);
  rewriter.create<scf::ForOp>(
      loc, zeroVal, rankVal, oneVal, ValueRange(),
      [&](OpBuilder &b, Location loc, Value iVal, ValueRange args) {
        auto dimVal = b.create<DimOp>(loc, tensorVal, iVal);
        auto dimIntVal = b.create<IndexCastOp>(loc, dimVal, i64Ty);
        b.create<StoreOp>(loc, dimIntVal, memVal, ValueRange({iVal}));
        b.create<scf::YieldOp>(loc);
      });

  // Load extents to tensor value.
  auto shapeIntVal = rewriter.create<TensorLoadOp>(loc, memVal);
  auto indexTy = rewriter.getIndexType();
  auto shapeTy = RankedTensorType::get({ShapedType::kDynamicSize}, indexTy);
  rewriter.replaceOpWithNewOp<IndexCastOp>(op.getOperation(), shapeIntVal,
                                           shapeTy);
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

  // Populate conversion patterns.
  OwningRewritePatternList patterns;
  populateShapeToSCFConversionPatterns(patterns, &ctx);

  // Setup target legality.
  ConversionTarget target(getContext());
  target.addLegalDialect<ShapeDialect, scf::SCFDialect, StandardOpsDialect>();
  target.addIllegalOp<ReduceOp, ShapeOfOp>();

  // Apply conversion.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

void mlir::populateShapeToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ReduceOpConverter, ShapeOfOpConverter>(ctx);
}

std::unique_ptr<FunctionPass> mlir::createConvertShapeToSCFPass() {
  return std::make_unique<ConvertShapeToSCFPass>();
}
