//===- XeGPUSgToWiDistributeExperimental.cpp - XeGPU SG to WI Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
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

static LogicalResult verifyLayouts(Operation *root) {
  auto walkResult = root->walk([&](Operation *nestedOp) -> WalkResult {
    if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(nestedOp)) {
      auto layout = anchorOp.getAnchorLayout();
      if (!layout) {
        nestedOp->emitError("expected anchor layout attribute on operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    // For each vector result, check if the op contains a result layout
    // attribute.
    for (OpResult result : nestedOp->getResults()) {
      if (isa<VectorType>(result.getType())) {
        auto layout = xegpu::getDistributeLayoutAttr(result);
        if (!layout) {
          nestedOp->emitError(
              "expected result layout attribute on vector result");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
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
  // Verify if all XeGPU and vector operations have layouts.
  Operation *root = getOperation();
  if (failed(verifyLayouts(root))) {
    signalPassFailure();
    return;
  }
}
void xegpu::populateXeGPUSgToWiDistributeTypeConversionAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {

  // Populate type conversions.
  // - Any type other than TensorDescType and VectorType are legal as is.
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (!isa<TensorDescType, VectorType>(type))
      return type;
    return std::nullopt;
  });
  // For TensorDescType, drop the layout attribute if any.
  typeConverter.addConversion([](TensorDescType type) -> Type {
    if (type.getLayoutAttr()) {
      return type.dropLayouts();
    }
    return type;
  });
  // - For VectorType, check if there is a distribute layout attribute on the
  //   value. If so, convert to the distributed vector type based on the layout.
  typeConverter.addConversion([](Value v) -> std::optional<Type> {
    auto type = v.getType();
    auto layout = xegpu::getDistributeLayoutAttr(v);
    // If no valid layout, nothing to do.
    if (!layout || !layout.isForSubgroup())
      return std::nullopt;
    // Vector type is distributed based on lane layout.
    if (isa<VectorType>(type)) {
      auto newTyOrFailure =
          getDistVecTypeBasedOnLaneLayout(layout, cast<VectorType>(type));
      if (succeeded(newTyOrFailure))
        return *newTyOrFailure;
    }
    return std::nullopt;
  });
  // - Materialization casts are only used for testing purposes.
  auto materializeCast = [&](mlir::OpBuilder &builder, mlir::Type type,
                             mlir::ValueRange inputs,
                             mlir::Location loc) -> mlir::Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  };
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);
  // Define legality.
  // - CreateNdDescOp is legal only if its result type has no layout attribute.
  target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
      [&](xegpu::CreateNdDescOp op) { return !op.getType().getLayoutAttr(); });
  // - Any anchor XeGPU op is legal only if it has no anchor layout.
  target.addDynamicallyLegalDialect<xegpu::XeGPUDialect>([](Operation *op) {
    auto anchorOp = dyn_cast<AnchorLayoutInterface>(op);
    if (!anchorOp)
      return true;
    return !anchorOp.getAnchorLayout();
  });
  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [=](arith::ConstantOp op) -> bool {
        // If the result type is not a vector, it's legal.
        if (!isa<VectorType>(op.getResult().getType()))
          return true;
        // For vector result types, check if it has a layout attribute.
        return !xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
      });
  // - In math and arith dialects, only handle elementwise ops with a single
  //   result and with a result layout attribute.
  target.addDynamicallyLegalDialect<math::MathDialect, arith::ArithDialect>(
      [=](Operation *op) -> std::optional<bool> {
        // Only handle elementwise mappable ops
        if (!OpTrait::hasElementwiseMappableTraits(op))
          return true;
        // Only handle ops with single vector result
        if (op->getNumResults() != 1)
          return true;

        VectorType resultType =
            dyn_cast<VectorType>(op->getResult(0).getType());
        if (!resultType)
          return true;

        // Check if all operands are vectors of the same shape
        for (Value operand : op->getOperands()) {
          VectorType operandType = dyn_cast<VectorType>(operand.getType());
          if (!operandType || operandType.getShape() != resultType.getShape()) {
            return true;
          }
        }
        return !xegpu::getTemporaryLayout(dyn_cast<OpResult>(op->getResult(0)));
      });
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  patterns.add<CreateNdDescOpPattern, LoadNdOpPattern, StoreNdOpPattern,
               DpasOpPattern, ElementWiseOpPattern, ArithConstantOpPattern>(
      typeConverter, patterns.getContext());
}
