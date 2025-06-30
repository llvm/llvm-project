//===- TestMemRefToLLVMWithTransforms.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestMemRefToLLVMWithTransforms
    : public PassWrapper<TestMemRefToLLVMWithTransforms,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMemRefToLLVMWithTransforms)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final {
    return "test-memref-to-llvm-with-transforms";
  }

  StringRef getDescription() const final {
    return "Tests conversion of MemRef dialects + `func.func` to LLVM dialect "
           "with MemRef transforms.";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    RewritePatternSet patterns(ctx);
    memref::populateExpandStridedMetadataPatterns(patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::test {
void registerTestMemRefToLLVMWithTransforms() {
  PassRegistration<TestMemRefToLLVMWithTransforms>();
}
} // namespace mlir::test
