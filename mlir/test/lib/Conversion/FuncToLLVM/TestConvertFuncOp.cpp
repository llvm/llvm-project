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

struct TestConvertFuncOp
    : public PassWrapper<TestConvertFuncOp, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertFuncOp)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }
  [[nodiscard]] StringRef getArgument() const final { return "test-convert-func-op"; }
  [[nodiscard]] StringRef getDescription() const final {
    return "Tests conversion of `func.func` to `llvm.func` for different attributes"
           ;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    LowerToLLVMOptions options(m.getContext());

    // Populate type conversions.
    LLVMTypeConverter typeConverter(m.getContext(), options);

    // Populate patterns.
    RewritePatternSet patterns(m.getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // Set target.
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<func::FuncDialect>();

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace


namespace mlir::test {
void registerConvertFuncOpPass() { PassRegistration<TestConvertFuncOp>(); }
} // namespace mlir::test
