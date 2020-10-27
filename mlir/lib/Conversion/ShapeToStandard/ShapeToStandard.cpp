//===- ShapeToStandard.cpp - conversion from Shape to Standard dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::shape;

/// Conversion patterns.
namespace {
class AnyOpConversion : public OpConversionPattern<AnyOp> {
public:
  using OpConversionPattern<AnyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AnyOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
AnyOpConversion::matchAndRewrite(AnyOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  AnyOp::Adaptor transformed(operands);

  // Replace `any` with its first operand.
  // Any operand would be a valid substitution.
  rewriter.replaceOp(op, {transformed.inputs().front()});
  return success();
}

namespace {
template <typename SrcOpTy, typename DstOpTy>
class BinaryOpConversion : public OpConversionPattern<SrcOpTy> {
public:
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor transformed(operands);

    // For now, only error-free types are supported by this lowering.
    if (op.getType().template isa<SizeType>())
      return failure();

    rewriter.replaceOpWithNewOp<DstOpTy>(op, transformed.lhs(),
                                         transformed.rhs());
    return success();
  }
};
} // namespace

namespace {
class ConstSizeOpConversion : public OpConversionPattern<ConstSizeOp> {
public:
  using OpConversionPattern<ConstSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstSizeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<ConstantIndexOp>(op, op.value().getSExtValue());
    return success();
  }
};
} // namespace

namespace {
class ShapeOfOpConversion : public OpConversionPattern<ShapeOfOp> {
public:
  using OpConversionPattern<ShapeOfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeOfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ShapeOfOpConversion::matchAndRewrite(
    ShapeOfOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  // For now, only error-free types are supported by this lowering.
  if (op.getType().isa<ShapeType>())
    return failure();

  // For unranked tensors `shape_of` lowers to `scf` and the pattern can be
  // found in the corresponding pass.
  ShapeOfOp::Adaptor transformed(operands);
  Value tensorVal = transformed.arg();
  Type tensorTy = tensorVal.getType();
  if (tensorTy.isa<UnrankedTensorType>())
    return failure();

  // Build values for individual dimensions.
  SmallVector<Value, 8> dimValues;
  RankedTensorType rankedTensorTy = tensorTy.cast<RankedTensorType>();
  int64_t rank = rankedTensorTy.getRank();
  auto loc = op.getLoc();
  for (int64_t i = 0; i < rank; i++) {
    if (rankedTensorTy.isDynamicDim(i)) {
      Value dimVal = rewriter.create<DimOp>(loc, tensorVal, i);
      dimValues.push_back(dimVal);
    } else {
      int64_t dim = rankedTensorTy.getDimSize(i);
      Value dimVal = rewriter.create<ConstantIndexOp>(loc, dim);
      dimValues.push_back(dimVal);
    }
  }

  // Materialize extent tensor.
  Value staticExtentTensor =
      rewriter.create<TensorFromElementsOp>(loc, dimValues);
  rewriter.replaceOpWithNewOp<TensorCastOp>(op, staticExtentTensor,
                                            op.getType());
  return success();
}

namespace {
class ConstShapeOpConverter : public OpConversionPattern<ConstShapeOp> {
public:
  using OpConversionPattern<ConstShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ConstShapeOpConverter::matchAndRewrite(
    ConstShapeOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  // For now, this lowering supports only extent tensors, not `shape.shape`
  // types.
  if (op.getType().isa<ShapeType>())
    return failure();

  auto loc = op.getLoc();
  SmallVector<Value, 4> extentOperands;
  for (auto extent : op.shape()) {
    extentOperands.push_back(
        rewriter.create<ConstantIndexOp>(loc, extent.getLimitedValue()));
  }
  Value tensor = rewriter.create<TensorFromElementsOp>(loc, extentOperands);
  Type indexTy = rewriter.getIndexType();
  Type resultTy = RankedTensorType::get({ShapedType::kDynamicSize}, indexTy);
  rewriter.replaceOpWithNewOp<TensorCastOp>(op, tensor, resultTy);
  return success();
}

namespace {
class ToExtentTensorOpConversion
    : public OpConversionPattern<ToExtentTensorOp> {
public:
  using OpConversionPattern<ToExtentTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToExtentTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ToExtentTensorOpAdaptor adaptor(operands);

    if (!adaptor.input().getType().isa<RankedTensorType>())
      return rewriter.notifyMatchFailure(op, "input needs to be a tensor");

    rewriter.replaceOpWithNewOp<TensorCastOp>(op, adaptor.input(),
                                              op.getType());
    return success();
  }
};
} // namespace

namespace {
class GetExtentOpConverter : public OpConversionPattern<GetExtentOp> {
  using OpConversionPattern<GetExtentOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetExtentOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult GetExtentOpConverter::matchAndRewrite(
    GetExtentOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  GetExtentOp::Adaptor transformed(operands);

  // For now, only error-free types are supported by this lowering.
  if (op.getType().isa<SizeType>())
    return failure();

  // Derive shape extent directly from shape origin if possible. This
  // circumvents the necessity to materialize the shape in memory.
  if (auto shapeOfOp = op.shape().getDefiningOp<ShapeOfOp>()) {
    if (shapeOfOp.arg().getType().isa<ShapedType>()) {
      rewriter.replaceOpWithNewOp<DimOp>(op, shapeOfOp.arg(),
                                         transformed.dim());
      return success();
    }
  }

  rewriter.replaceOpWithNewOp<ExtractElementOp>(op, rewriter.getIndexType(),
                                                transformed.shape(),
                                                ValueRange{transformed.dim()});
  return success();
}

namespace {
class RankOpConverter : public OpConversionPattern<shape::RankOp> {
public:
  using OpConversionPattern<shape::RankOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::RankOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
RankOpConverter::matchAndRewrite(shape::RankOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  // For now, this lowering supports only error-free types.
  if (op.getType().isa<SizeType>())
    return failure();

  shape::RankOp::Adaptor transformed(operands);
  rewriter.replaceOpWithNewOp<DimOp>(op, transformed.shape(), 0);
  return success();
}

namespace {
/// Conversion pass.
class ConvertShapeToStandardPass
    : public ConvertShapeToStandardBase<ConvertShapeToStandardPass> {

  void runOnOperation() override;
};
} // namespace

void ConvertShapeToStandardPass::runOnOperation() {
  // Setup target legality.
  MLIRContext &ctx = getContext();
  ConversionTarget target(ctx);
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<FuncOp, ModuleOp, ModuleTerminatorOp>();

  // Setup conversion patterns.
  OwningRewritePatternList patterns;
  populateShapeToStandardConversionPatterns(patterns, &ctx);

  // Apply conversion.
  auto module = getOperation();
  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

void mlir::populateShapeToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
      AnyOpConversion,
      BinaryOpConversion<AddOp, AddIOp>,
      ConstShapeOpConverter,
      BinaryOpConversion<MulOp, MulIOp>,
      ConstSizeOpConversion,
      GetExtentOpConverter,
      RankOpConverter,
      ShapeOfOpConversion,
      ToExtentTensorOpConversion>(ctx);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertShapeToStandardPass() {
  return std::make_unique<ConvertShapeToStandardPass>();
}
