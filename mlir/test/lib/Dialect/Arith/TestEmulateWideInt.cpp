//===- TestWideIntEmulation.cpp - Test Wide Int Emulation  ------*- c++ -*-===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/WideIntEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct TestEmulateWideIntPass
    : public PassWrapper<TestEmulateWideIntPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEmulateWideIntPass)

  TestEmulateWideIntPass() = default;
  TestEmulateWideIntPass(const TestEmulateWideIntPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    vector::VectorDialect>();
  }
  StringRef getArgument() const final { return "test-arith-emulate-wide-int"; }
  StringRef getDescription() const final {
    return "Function pass to test Wide Integer Emulation";
  }

  void runOnOperation() override {
    if (!llvm::isPowerOf2_32(widestIntSupported) || widestIntSupported < 2) {
      signalPassFailure();
      return;
    }

    func::FuncOp op = getOperation();
    if (!op.getSymName().startswith(testFunctionPrefix))
      return;

    MLIRContext *ctx = op.getContext();
    arith::WideIntEmulationConverter typeConverter(widestIntSupported);

    // Use `llvm.bitcast` as the bridge so that we can use preserve the
    // function argument and return types of the processed function.
    // TODO: Consider extending `arith.bitcast` to support scalar-to-1D-vector
    // casts (and vice versa) and using it insted of `llvm.bitcast`.
    auto addBitcast = [](OpBuilder &builder, Type type, ValueRange inputs,
                         Location loc) -> Optional<Value> {
      auto cast = builder.create<LLVM::BitcastOp>(loc, type, inputs);
      return cast->getResult(0);
    };
    typeConverter.addSourceMaterialization(addBitcast);
    typeConverter.addTargetMaterialization(addBitcast);

    ConversionTarget target(*ctx);
    target
        .addDynamicallyLegalDialect<arith::ArithDialect, vector::VectorDialect>(
            [&typeConverter](Operation *op) {
              return typeConverter.isLegal(op);
            });

    RewritePatternSet patterns(ctx);
    arith::populateArithWideIntEmulationPatterns(typeConverter, patterns);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }

  Option<std::string> testFunctionPrefix{
      *this, "function-prefix",
      llvm::cl::desc("Prefix of functions to run the emulation pass on"),
      llvm::cl::init("emulate_")};
  Option<unsigned> widestIntSupported{
      *this, "widest-int-supported",
      llvm::cl::desc("Maximum integer bit width supported by the target"),
      llvm::cl::init(32)};
};
} // namespace

namespace mlir::test {
void registerTestArithEmulateWideIntPass() {
  PassRegistration<TestEmulateWideIntPass>();
}
} // namespace mlir::test
