//===- TileReducerBufferConversion.cpp - Milestone 7 ------------*- C++ -*-===//
//
// ConversionTarget + TypeConverter + OpConversionPattern.
// !tr.buffer -> memref. !tr.tile is unchanged.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/TileReducerPasses.h"

#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tr {
#define GEN_PASS_DEF_CONVERTTRBUFFERSTOMEMREF
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

static Type convertBuffer(BufferType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

struct TRTypeConverter : public TypeConverter {
  TRTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](BufferType type) { return convertBuffer(type); });

    addSourceMaterialization([](OpBuilder &b, BufferType type, ValueRange inputs,
                                Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      return UnrealizedConversionCastOp::create(b, loc, type, inputs)
          .getResult(0);
    });
    addTargetMaterialization([](OpBuilder &b, MemRefType type, ValueRange inputs,
                                Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      return UnrealizedConversionCastOp::create(b, loc, type, inputs)
          .getResult(0);
    });
  }
};

struct DimOpConversion : public OpConversionPattern<DimOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value idx =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), op.getAxis());
    rewriter.replaceOpWithNewOp<memref::DimOp>(op, adaptor.getBuffer(), idx);
    return success();
  }
};

template <typename OpTy>
struct ForwardConvertedOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] {
      op->setOperands(adaptor.getOperands());
    });
    return success();
  }
};

struct ConvertTRBuffersToMemRef
    : impl::ConvertTRBuffersToMemRefBase<ConvertTRBuffersToMemRef> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    TRTypeConverter converter;

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<DimOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    auto isLegalOp = [&](Operation *op) {
      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    };
    target.addDynamicallyLegalOp<LoadOp, StoreOp, ForOp, YieldOp, ProgramIdOp,
                                 ConstantOp, AddOp, ReduceSumOp>(isLegalOp);
    target.addLegalDialect<TileReducerDialect>();
    // Illegal ops / dynamic legality take precedence over addLegalDialect.

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    patterns.add<DimOpConversion, ForwardConvertedOperands<LoadOp>,
                 ForwardConvertedOperands<StoreOp>,
                 ForwardConvertedOperands<ForOp>,
                 ForwardConvertedOperands<YieldOp>>(converter, ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tr
