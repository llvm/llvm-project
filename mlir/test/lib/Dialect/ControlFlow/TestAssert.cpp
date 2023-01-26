//===- TestAssert.cpp - Test cf.assert Lowering  ----------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for integration testing of wide integer
// emulation patterns. Applies conversion patterns only to functions whose
// names start with a specified prefix.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct TestAssertPass
    : public PassWrapper<TestAssertPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAssertPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return "test-cf-assert"; }
  StringRef getDescription() const final {
    return "Function pass to test cf.assert lowering to LLVM without abort";
  }

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LLVMTypeConverter converter(&getContext());
    mlir::cf::populateAssertToLLVMConversionPattern(converter, patterns,
                                                    /*abortOnFailure=*/false);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::test {
void registerTestCfAssertPass() { PassRegistration<TestAssertPass>(); }
} // namespace mlir::test
