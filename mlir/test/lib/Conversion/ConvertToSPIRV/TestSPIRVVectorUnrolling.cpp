//===- TestSPIRVVectorUnrolling.cpp - Test signature conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace {

struct TestSPIRVVectorUnrolling final
    : PassWrapper<TestSPIRVVectorUnrolling, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSPIRVVectorUnrolling)

  StringRef getArgument() const final { return "test-spirv-vector-unrolling"; }

  StringRef getDescription() const final {
    return "Test patterns that unroll vectors to types supported by SPIR-V";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<spirv::SPIRVDialect, vector::VectorDialect, ub::UBDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    (void)spirv::unrollVectorsInSignatures(op);
    (void)spirv::unrollVectorsInFuncBodies(op);
  }
};

} // namespace

namespace test {
void registerTestSPIRVVectorUnrolling() {
  PassRegistration<TestSPIRVVectorUnrolling>();
}
} // namespace test
} // namespace mlir
