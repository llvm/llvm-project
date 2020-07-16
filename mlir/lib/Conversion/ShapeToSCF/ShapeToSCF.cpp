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
using namespace mlir::scf;

namespace {
/// Converts `shape.reduce` to `scf.for`.
struct ReduceOpConverter : public OpConversionPattern<shape::ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::ReduceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
ReduceOpConverter::matchAndRewrite(shape::ReduceOp op, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands.
  if (!op.shape().getType().isa<RankedTensorType>())
    return failure();

  auto loc = op.getLoc();
  shape::ReduceOp::Adaptor transformed(operands);

  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  Type indexTy = rewriter.getIndexType();
  Value rank = rewriter.create<DimOp>(loc, indexTy, transformed.shape(), zero);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, rank, one, op.initVals(),
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Value extent = b.create<ExtractElementOp>(loc, transformed.shape(), iv);

        SmallVector<Value, 2> mappedValues{iv, extent};
        mappedValues.append(args.begin(), args.end());

        BlockAndValueMapping mapping;
        Block *reduceBody = op.getBody();
        mapping.map(reduceBody->getArguments(), mappedValues);
        for (auto &nested : reduceBody->without_terminator())
          b.clone(nested, mapping);

        SmallVector<Value, 2> mappedResults;
        for (auto result : reduceBody->getTerminator()->getOperands())
          mappedResults.push_back(mapping.lookup(result));
        b.create<scf::YieldOp>(loc, mappedResults);
      });

  rewriter.replaceOp(op, loop.getResults());
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
  auto rankVal = rewriter.create<mlir::RankOp>(loc, tensorVal);
  auto i64Ty = rewriter.getI64Type();
  auto memTy = MemRefType::get({ShapedType::kDynamicSize}, i64Ty);
  auto memVal = rewriter.create<AllocaOp>(loc, memTy, ValueRange({rankVal}));

  // Copy shape extents to stack-allocated memory.
  auto zeroVal = rewriter.create<ConstantIndexOp>(loc, 0);
  auto oneVal = rewriter.create<ConstantIndexOp>(loc, 1);
  auto loop = rewriter.create<scf::ForOp>(loc, zeroVal, rankVal, oneVal);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto iVal = loop.getInductionVar();
    auto dimVal = rewriter.create<DimOp>(loc, tensorVal, iVal);
    auto dimIntVal = rewriter.create<IndexCastOp>(loc, dimVal, i64Ty);
    rewriter.create<StoreOp>(loc, dimIntVal, memVal, ValueRange{iVal});
  }

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
  target.addLegalDialect<SCFDialect, StandardOpsDialect>();
  target.addLegalOp<ModuleOp, FuncOp>();

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
