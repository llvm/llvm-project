//===- TestConvertCallOp.cpp - Test LLVM Conversion of Standard CallOp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestTypes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
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

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    LLVMTypeConverter type_converter(m.getContext());
    type_converter.addConversion([&](test::TestType type) {
      return LLVM::LLVMType::getInt8PtrTy(m.getContext());
    });

    // Populate patterns.
    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(type_converter, patterns);
    patterns.insert<TestTypeProducerOpConverter>(type_converter);

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
void registerConvertCallOpPass() {
  PassRegistration<TestConvertCallOp>(
      "test-convert-call-op",
      "Tests conversion of `std.call` to `llvm.call` in "
      "presence of custom types");
}
} // namespace test
} // namespace mlir
