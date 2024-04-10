//===- TestSimplification.cpp - Test simplification -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/Transforms/Simplifications.h"
#include "mlir/IR/SymbolTable.h"
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
  SymbolTableCollection symbolTableCollection;
  mesh::populateSimplificationPatterns(patterns, symbolTableCollection);
  [[maybe_unused]] LogicalResult status =
      applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  assert(succeeded(status) && "Rewrite patters application did not converge.");
}

namespace mlir {
namespace test {
void registerTestMeshSimplificationsPass() {
  PassRegistration<TestMeshSimplificationsPass>();
}
} // namespace test
} // namespace mlir
