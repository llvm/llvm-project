//===- TestVectorReductionToSPIRVDotProd.cpp - Test reduction to dot prod -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

struct TestVectorReductionToSPIRVDotProd
    : PassWrapper<TestVectorReductionToSPIRVDotProd,
                  OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorReductionToSPIRVDotProd)

  StringRef getArgument() const final {
    return "test-vector-reduction-to-spirv-dot-prod";
  }

  StringRef getDescription() const final {
    return "Test lowering patterns that converts vector.reduction to SPIR-V "
           "integer dot product ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, spirv::SPIRVDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorReductionToSPIRVDotProductPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace test {
void registerTestVectorReductionToSPIRVDotProd() {
  PassRegistration<TestVectorReductionToSPIRVDotProd>();
}
} // namespace test
} // namespace mlir
