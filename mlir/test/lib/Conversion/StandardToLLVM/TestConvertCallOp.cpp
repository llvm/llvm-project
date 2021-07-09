//===- TestConvertCallOp.cpp - Test LLVM Conversion of Standard CallOp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestTypes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class TestTypeProducerOpConverter
    : public ConvertOpToLLVMPattern<test::TestTypeProducerOp> {
public:
  using ConvertOpToLLVMPattern<
      test::TestTypeProducerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(test::TestTypeProducerOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, getVoidPtrType());
    return success();
  }
};

class TestConvertCallOp
    : public PassWrapper<TestConvertCallOp, OperationPass<ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return "test-convert-call-op"; }
  StringRef getDescription() const final {
    return "Tests conversion of `std.call` to `llvm.call` in "
           "presence of custom types";
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    LLVMTypeConverter typeConverter(m.getContext());
    typeConverter.addConversion([&](test::TestType type) {
      return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8));
    });

    // Populate patterns.
    RewritePatternSet patterns(m.getContext());
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<TestTypeProducerOpConverter>(typeConverter);

    // Set target.
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<test::TestDialect>();
    target.addIllegalDialect<StandardOpsDialect>();

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerConvertCallOpPass() { PassRegistration<TestConvertCallOp>(); }
} // namespace test
} // namespace mlir
