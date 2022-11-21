//===- TestDialectConversion.cpp - Test DialectConversion functionality ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// Test PDLL Support
//===----------------------------------------------------------------------===//

#include "TestDialectConversionPDLLPatterns.h.inc"

namespace {
struct PDLLTypeConverter : public TypeConverter {
  PDLLTypeConverter() {
    addConversion(convertType);
    addArgumentMaterialization(materializeCast);
    addSourceMaterialization(materializeCast);
  }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    // Convert I64 to F64.
    if (t.isSignlessInteger(64)) {
      results.push_back(FloatType::getF64(t.getContext()));
      return success();
    }

    // Otherwise, convert the type directly.
    results.push_back(t);
    return success();
  }
  /// Hook for materializing a conversion.
  static Optional<Value> materializeCast(OpBuilder &builder, Type resultType,
                                         ValueRange inputs, Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  }
};

struct TestDialectConversionPDLLPass
    : public PassWrapper<TestDialectConversionPDLLPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialectConversionPDLLPass)

  StringRef getArgument() const final { return "test-dialect-conversion-pdll"; }
  StringRef getDescription() const final {
    return "Test DialectConversion PDLL functionality";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
  }
  LogicalResult initialize(MLIRContext *ctx) override {
    // Build the pattern set within the `initialize` to avoid recompiling PDL
    // patterns during each `runOnOperation` invocation.
    RewritePatternSet patternList(ctx);
    registerConversionPDLFunctions(patternList);
    populateGeneratedPDLLPatterns(patternList, PDLConversionConfig(&converter));
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addDynamicallyLegalDialect<TestDialect>(
        [this](Operation *op) { return converter.isLegal(op); });

    if (failed(mlir::applyFullConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }

  FrozenRewritePatternSet patterns;
  PDLLTypeConverter converter;
};
} // namespace

namespace mlir {
namespace test {
void registerTestDialectConversionPasses() {
  PassRegistration<TestDialectConversionPDLLPass>();
}
} // namespace test
} // namespace mlir
