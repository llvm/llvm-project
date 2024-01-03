//===- TestSimplification.cpp - Test simplification -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Transforms/Simplifications.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestMeshSimplificationsPass
    : public PassWrapper<TestMeshSimplificationsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMeshSimplificationsPass)

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, mesh::MeshDialect>();
  }
  StringRef getArgument() const final { return "test-mesh-simplifications"; }
  StringRef getDescription() const final { return "Test mesh simplifications"; }
};
} // namespace

void TestMeshSimplificationsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  mesh::populateSimplificationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
namespace test {
void registerTestMeshSimplificationsPass() {
  PassRegistration<TestMeshSimplificationsPass>();
}
} // namespace test
} // namespace mlir
