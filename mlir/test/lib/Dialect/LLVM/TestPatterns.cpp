//===- TestPatterns.cpp - LLVM dialect test patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Replace this op (which is expected to have 1 result) with the operands.
struct TestDirectReplacementOp : public ConversionPattern {
  TestDirectReplacementOp(MLIRContext *ctx, const TypeConverter &converter)
      : ConversionPattern(converter, "test.direct_replacement", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (op->getNumResults() != 1)
      return failure();
    rewriter.replaceOpWithMultiple(op, {operands});
    return success();
  }
};

struct TestLLVMLegalizePatternsPass
    : public PassWrapper<TestLLVMLegalizePatternsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLLVMLegalizePatternsPass)

  TestLLVMLegalizePatternsPass() = default;
  TestLLVMLegalizePatternsPass(const TestLLVMLegalizePatternsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-llvm-legalize-patterns"; }
  StringRef getDescription() const final {
    return "Run LLVM dialect legalization patterns";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Set up type converter.
    LLVMTypeConverter converter(ctx);
    converter.addConversion(
        [&](IntegerType type, SmallVectorImpl<Type> &result) {
          if (type.isInteger(17)) {
            // Convert i17 -> (i18, i18).
            result.append(2, Builder(ctx).getIntegerType(18));
            return success();
          }

          result.push_back(type);
          return success();
        });

    // Populate patterns.
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<TestDirectReplacementOp>(ctx, converter);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

    // Define the conversion target used for the test.
    ConversionTarget target(*ctx);
    target.addLegalOp(OperationName("test.legal_op", ctx));
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp funcOp) { return funcOp->hasAttr("is_legal"); });

    // Handle a partial conversion.
    DenseSet<Operation *> unlegalizedOps;
    ConversionConfig config;
    config.unlegalizedOps = &unlegalizedOps;
    config.allowPatternRollback = allowPatternRollback;
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns), config)))
      getOperation()->emitError() << "applyPartialConversion failed";
  }

  Option<bool> allowPatternRollback{*this, "allow-pattern-rollback",
                                    llvm::cl::desc("Allow pattern rollback"),
                                    llvm::cl::init(true)};
};
} // namespace

//===----------------------------------------------------------------------===//
// PassRegistration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace test {
void registerTestLLVMLegalizePatternsPass() {
  PassRegistration<TestLLVMLegalizePatternsPass>();
}
} // namespace test
} // namespace mlir
