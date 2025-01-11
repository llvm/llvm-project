//===- TestReduceFloatBitwdithConversion.cpp ----------------*- c++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that reduces the bitwidth of Arith floating-point IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::arith;

namespace {

/// Pattern for arith.constant.
class ConstantOpPattern : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    double val = cast<FloatAttr>(op.getValue()).getValueAsDouble();
    auto newAttr = FloatAttr::get(Float16Type::get(op.getContext()), val);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, newAttr);
    return success();
  }
};

/// Pattern for arith.addf.
class AddOpPattern : public OpConversionPattern<AddFOp> {
  using OpConversionPattern<AddFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<AddFOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct TestReduceFloatBitwidthConversionPass
    : public PassWrapper<TestReduceFloatBitwidthConversionPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestReduceFloatBitwidthConversionPass)

  TestReduceFloatBitwidthConversionPass() = default;
  TestReduceFloatBitwidthConversionPass(
      const TestReduceFloatBitwidthConversionPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }
  StringRef getArgument() const final {
    return "test-arith-reduce-float-bitwidth-conversion";
  }
  StringRef getDescription() const final {
    return "Pass that reduces the bitwidth of floating-point ops (dialect "
           "conversion)";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    TypeConverter converter;
    ConversionConfig config;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [&](Float32Type type) { return FloatType::getF16(ctx); });
    if (optBuildMaterializations) {
      converter.addSourceMaterialization(
          [](OpBuilder &builder, FloatType resultType, ValueRange inputs,
             Location loc) -> Value {
            assert(inputs.size() == 1 && "expected single input");
            return builder.create<ExtFOp>(loc, resultType, inputs[0]);
          });
      converter.addTargetMaterialization(
          [](OpBuilder &builder, FloatType resultType, ValueRange inputs,
             Location loc) -> Value {
            assert(inputs.size() == 1 && "expected single input");
            return builder.create<TruncFOp>(loc, resultType, inputs[0]);
          });
      config.buildMaterializations = true;
    } else {
      config.buildMaterializations = false;
    }

    RewritePatternSet patterns(ctx);
    patterns.insert<ConstantOpPattern, AddOpPattern>(converter, ctx);
    // Pattern for func.func.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<ConstantOp, AddFOp, func::ReturnOp>(
        [&](Operation *op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });

    LogicalResult status = failure();
    if (optFullConversion) {
      status = applyFullConversion(getOperation(), target, std::move(patterns),
                                   config);
    } else {
      status = applyPartialConversion(getOperation(), target,
                                      std::move(patterns), config);
    }
    if (failed(status)) {
      getOperation()->emitError() << getArgument() << " failed";
      signalPassFailure();
    }
  }

  Option<bool> optBuildMaterializations{
      *this, "build-materializations", llvm::cl::init(false),
      llvm::cl::desc("build materializations")};
  Option<bool> optFullConversion{
      *this, "full-conversion", llvm::cl::init(false),
      llvm::cl::desc("full conversion (otherwise: partial)")};
};
} // namespace

namespace mlir {
void registerTestReduceFloatBitwidthConversionPass() {
  PassRegistration<TestReduceFloatBitwidthConversionPass>();
}
} // namespace mlir
