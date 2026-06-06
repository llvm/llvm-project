//===- TestSPIRVFuncSignatureConversion.cpp - Test signature conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

struct TestSPIRVFuncSignatureConversion final
    : PassWrapper<TestSPIRVFuncSignatureConversion, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSPIRVFuncSignatureConversion)

  StringRef getArgument() const final {
    return "test-spirv-func-signature-conversion";
  }

  StringRef getDescription() const final {
    return "Test patterns that convert vector inputs and results in function "
           "signatures";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, spirv::SPIRVDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    (void)spirv::unrollVectorsInSignatures(op);
  }
};

} // namespace

namespace test {
void registerTestSPIRVFuncSignatureConversion() {
  PassRegistration<TestSPIRVFuncSignatureConversion>();
}
} // namespace test
} // namespace mlir
