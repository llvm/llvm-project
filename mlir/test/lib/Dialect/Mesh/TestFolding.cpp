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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

using namespace mlir;

namespace {

struct TestMeshFoldingPass
    : public PassWrapper<TestMeshFoldingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMeshFoldingPass)

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mesh::MeshDialect>();
  }
  StringRef getArgument() const final { return "test-mesh-folding"; }
  StringRef getDescription() const final { return "Test mesh folding."; }
};
} // namespace

void TestMeshFoldingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  SymbolTableCollection symbolTables;
  mesh::populateFoldingPatterns(patterns, symbolTables);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError()
        << "Rewrite patter application did not converge.";
    return signalPassFailure();
  }
}

namespace mlir {
namespace test {
void registerTestMeshFoldingPass() { PassRegistration<TestMeshFoldingPass>(); }
} // namespace test
} // namespace mlir
