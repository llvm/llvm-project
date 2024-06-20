//===- TestConvertCallOp.cpp - Test LLVM Conversion of Func CallOp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "TestTypes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class TestTypeProducerOpConverter
    : public ConvertOpToLLVMPattern<test::TestTypeProducerOp> {
public:
  using ConvertOpToLLVMPattern<
      test::TestTypeProducerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(test::TestTypeProducerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, getVoidPtrType());
    return success();
  }
};

struct TestConvertCallOp
    : public PassWrapper<TestConvertCallOp, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertCallOp)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return "test-convert-call-op"; }
  StringRef getDescription() const final {
    return "Tests conversion of `func.call` to `llvm.call` in "
           "presence of custom types";
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    LowerToLLVMOptions options(m.getContext());

    // Populate type conversions.
    LLVMTypeConverter typeConverter(m.getContext(), options);
    typeConverter.addConversion([&](test::TestType type) {
      return LLVM::LLVMPointerType::get(m.getContext());
    });
    typeConverter.addConversion([&](test::SimpleAType type) {
      return IntegerType::get(type.getContext(), 42);
    });

    // Populate patterns.
    RewritePatternSet patterns(m.getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<TestTypeProducerOpConverter>(typeConverter);

    // Set target.
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<test::TestDialect>();
    target.addIllegalDialect<func::FuncDialect>();

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
