//===- XeGPUSgToWiDistributeExperimental.cpp - XeGPU SG to WI Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUSGTOWIDISTRIBUTEEXPERIMENTAL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

namespace {

static Value resolveTy(ConversionPatternRewriter &rewriter,
                       TypedValue<VectorType> v, VectorType expectedTy) {
  if (v.getType() == expectedTy)
    return v;
  assert(v.getType().getElementType() == expectedTy.getElementType() &&
         "element types must match");
  assert(v.getType().getNumElements() == expectedTy.getNumElements() &&
         "total number of elements must match");
  return vector::ShapeCastOp::create(rewriter, v.getLoc(), expectedTy, v);
}

struct CreateNdDescOpPattern
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::TensorDescType resultType = op.getType();
    // If no layout, nothing to do.
    if (!resultType.getLayout())
      return failure();

    auto newOp = xegpu::CreateNdDescOp::create(
        rewriter, op.getLoc(), resultType.dropLayouts(), op.getOperands(),
        op->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct LoadNdOpPattern : public OpConversionPattern<xegpu::LoadNdOp> {
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout = op.getAnchorLayout();
    // If no layout, nothing to do.
    if (!layout)
      return failure();
    // Check if the layout attached to the tensor descriptor is same as the
    // anchor layout. Otherwise, this is a conflict.
    if (op.getTensorDescType().getLayout() != layout)
      return rewriter.notifyMatchFailure(
          op, "conflicting layout attributes on tensor descriptor and anchor");
    auto supportedWiResultTyOrFailure =
        xegpu::getDistributedVectorType(op.getTensorDescType());
    auto expectedWiResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, op.getType());
    if (failed(supportedWiResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute the workitem vector type for LoadNdOp");
    if (failed(expectedWiResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op,
          "unable to compute expected workitem vector type from lane layout");
    auto newOp = xegpu::LoadNdOp::create(
        rewriter, op.getLoc(), supportedWiResultTyOrFailure.value(),
        adaptor.getTensorDesc(), op.getMixedOffsets(), op.getPackedAttr(),
        op.getTransposeAttr(), op.getL1HintAttr(), op.getL2HintAttr(),
        op.getL3HintAttr(), /**layout**/ nullptr);
    rewriter.replaceOp(op, resolveTy(rewriter, newOp.getResult(),
                                     expectedWiResultTyOrFailure.value()));
    return success();
  }
};

struct StoreNdOpPattern : public OpConversionPattern<xegpu::StoreNdOp> {
  using OpConversionPattern<xegpu::StoreNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::StoreNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout = op.getAnchorLayout();
    // If no layout, nothing to do.
    if (!layout)
      return failure();
    // Check if the layout attached to the tensor descriptor and value layout is
    // same as the anchor layout. Otherwise, this is a conflict.
    if (op.getTensorDescType().getLayout() != layout)
      return rewriter.notifyMatchFailure(
          op, "conflicting layout attributes on tensor descriptor and anchor");
    auto valueLayout = xegpu::getDistributeLayoutAttr(op->getOpOperand(0));
    if (valueLayout != layout)
      return rewriter.notifyMatchFailure(
          op, "conflicting layout attributes on value and anchor");
    auto supportedWiValueTyOrFailure =
        xegpu::getDistributedVectorType(op.getTensorDescType());
    if (failed(supportedWiValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          op,
          "unable to compute wi vector type for StoreNdOp value from tensor "
          "descriptor");

    xegpu::StoreNdOp::create(
        rewriter, op.getLoc(),
        resolveTy(rewriter, cast<TypedValue<VectorType>>(adaptor.getValue()),
                  supportedWiValueTyOrFailure.value()),
        adaptor.getTensorDesc(), op.getMixedOffsets(), op.getL1HintAttr(),
        op.getL2HintAttr(), op.getL3HintAttr(), /**layout**/ nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct DpasOpPattern : public OpConversionPattern<xegpu::DpasOp> {
  using OpConversionPattern<xegpu::DpasOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the op has A, B and CD layouts attached.
    auto layoutA = cast<xegpu::LayoutAttr>(op.getLayoutAAttr());
    auto layoutB = cast<xegpu::LayoutAttr>(op.getLayoutBAttr());
    auto layoutCd = cast<xegpu::LayoutAttr>(op.getLayoutCdAttr());
    if (!layoutA || !layoutB || !layoutCd)
      return failure();

    auto wiResultTyOrFailure =
        xegpu::getDistributedVectorType(op.getType(), layoutCd);
    auto wiATypeOrFailure =
        xegpu::getDistributedVectorType(op.getLhs().getType(), layoutA);
    auto wiBTypeOrFailure =
        xegpu::getDistributedVectorType(op.getRhs().getType(), layoutB);
    auto expectedWiResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layoutCd, op.getType());
    if (failed(wiResultTyOrFailure) || failed(wiATypeOrFailure) ||
        failed(wiBTypeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to calculate supported workitem vector types for DpasOp "
              "from layouts");
    if (failed(expectedWiResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute expected workitem vector type for DpasOp from "
              "lane layout");
    auto newOp = xegpu::DpasOp::create(
        rewriter, op->getLoc(), wiResultTyOrFailure.value(),
        resolveTy(rewriter, cast<TypedValue<VectorType>>(adaptor.getLhs()),
                  wiATypeOrFailure.value()),
        resolveTy(rewriter, cast<TypedValue<VectorType>>(adaptor.getRhs()),
                  wiBTypeOrFailure.value()),
        resolveTy(rewriter, cast<TypedValue<VectorType>>(adaptor.getAcc()),
                  wiResultTyOrFailure.value()),
        /** layoutA**/ nullptr,
        /** layoutB**/ nullptr, /** layoutCd**/ nullptr);
    // Explicitly set the new types to enable correct type materializations.
    rewriter.replaceOp(op, resolveTy(rewriter, newOp.getResult(),
                                     expectedWiResultTyOrFailure.value()));
    return success();
  }
};

struct ElementWiseOpPattern : public ConversionPattern {
  ElementWiseOpPattern(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match ops with elementwise trait and single result.
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();

    auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "operation result is not a vector type");

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(llvm::cast<OpResult>(op->getResult(0)));
    if (!layout || !layout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "operation result does not have subgroup distribute layout");

    auto wiShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, resultType);

    if (failed(wiShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute workitem vector type from the layout");

    VectorType newResultType = wiShapeOrFailure.value();
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(newResultType);
    // Copy all attributes except for DistributeLayoutAttr.
    for (auto attr : op->getAttrs()) {
      if (!isa<xegpu::DistributeLayoutAttr>(attr.getValue()))
        state.addAttribute(attr.getName(), attr.getValue());
    }
    Operation *newOp = rewriter.create(state);

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ArithConstantOpPattern : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return failure();

    // Only handle dense vector constants
    auto dense = dyn_cast<SplatElementsAttr>(op.getValue());
    if (!dense)
      return rewriter.notifyMatchFailure(
          op, "only dense splat vector constants are supported");

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(llvm::cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "operation result does not have subgroup distribute layout");

    auto wiShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, resultType);

    if (failed(wiShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute workitem vector type from the layout");

    VectorType newResultType = wiShapeOrFailure.value();
    auto sclarValue = dense.getSplatValue<Attribute>();
    auto newDenseAttr = DenseElementsAttr::get(newResultType, sclarValue);

    auto newOp = arith::ConstantOp::create(rewriter, op.getLoc(), newResultType,
                                           newDenseAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct XeGPUSgToWiDistributeExperimentalPass
    : public xegpu::impl::XeGPUSgToWiDistributeExperimentalBase<
          XeGPUSgToWiDistributeExperimentalPass> {
  void runOnOperation() override;
};

} // namespace

void XeGPUSgToWiDistributeExperimentalPass::runOnOperation() {
  // // Recover layouts.
  // Operation *op = getOperation();
  // if (!xegpu::recoverTemporaryLayouts(op)) {
  //   signalPassFailure();
  //   return;
  // }

  // // Define conversion target
  // ConversionTarget target(getContext());
  // target.addLegalDialect<index::IndexDialect, memref::MemRefDialect,
  //                        vector::VectorDialect>();
  // target.addDynamicallyLegalDialect<xegpu::XeGPUDialect>(
  //     [](Operation *op) { return true; });

  // // Define type converter
  // TypeConverter typeConverter;
  // typeConverter.addConversion([](Type type) { return type; });
}

void xegpu::populateXeGPUSgToWiDistributeExperimentalPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CreateNdDescOpPattern, LoadNdOpPattern, StoreNdOpPattern,
               DpasOpPattern, ElementWiseOpPattern, ArithConstantOpPattern>(
      typeConverter, patterns.getContext());
}
