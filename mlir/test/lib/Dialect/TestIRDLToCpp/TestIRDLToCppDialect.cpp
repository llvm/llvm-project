//===- TestIRDLToCppDialect.cpp - MLIR Test Dialect Types ---------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes TestIRDLToCpp dialect.
//
//===----------------------------------------------------------------------===//

// #include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TestIRDLToCppDialect.h"

#define GEN_DIALECT_DEF
#include "test_irdl_to_cpp.irdl.mlir.cpp.inc"

namespace test {
using namespace mlir;
struct TestOpConversion : public OpConversionPattern<test_irdl_to_cpp::BeefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::test_irdl_to_cpp::BeefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getStructuredOperands(0).size() == 1);
    assert(adaptor.getStructuredOperands(1).size() == 1);

    auto bar = rewriter.replaceOpWithNewOp<test_irdl_to_cpp::BarOp>(
        op, op->getResultTypes().front());
    rewriter.setInsertionPointAfter(bar);

    rewriter.create<test_irdl_to_cpp::HashOp>(
        bar.getLoc(), rewriter.getIntegerType(32), adaptor.getLhs(),
        adaptor.getRhs());
    return success();
  }
};

struct ConvertTestDialectToSomethingPass
    : PassWrapper<ConvertTestDialectToSomethingPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TestOpConversion>(ctx);
    ConversionTarget target(getContext());
    target.addIllegalOp<test_irdl_to_cpp::BeefOp>();
    target.addLegalOp<test_irdl_to_cpp::BarOp>();
    target.addLegalOp<test_irdl_to_cpp::HashOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "test-irdl-conversion-check"; }
  StringRef getDescription() const final {
    return "Checks the convertability of an irdl dialect";
  }
};

void registerIrdlTestDialect(mlir::DialectRegistry &registry) {
  registry.insert<mlir::test_irdl_to_cpp::TestIrdlToCppDialect>();
}

} // namespace test

namespace mlir::test {
void registerTestIrdlTestDialectConversionPass() {
  PassRegistration<::test::ConvertTestDialectToSomethingPass>();
}
} // namespace mlir::test
