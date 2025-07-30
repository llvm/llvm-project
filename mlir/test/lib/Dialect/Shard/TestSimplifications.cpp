//===- TestSimplification.cpp - Test simplification -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/Transforms/Simplifications.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestShardSimplificationsPass
    : public PassWrapper<TestShardSimplificationsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestShardSimplificationsPass)

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, shard::ShardDialect>();
  }
  StringRef getArgument() const final { return "test-grid-simplifications"; }
  StringRef getDescription() const final { return "Test grid simplifications"; }
};
} // namespace

void TestShardSimplificationsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  SymbolTableCollection symbolTableCollection;
  shard::populateSimplificationPatterns(patterns, symbolTableCollection);
  [[maybe_unused]] LogicalResult status =
      applyPatternsGreedily(getOperation(), std::move(patterns));
  assert(succeeded(status) && "Rewrite patters application did not converge.");
}

namespace mlir {
namespace test {
void registerTestShardSimplificationsPass() {
  PassRegistration<TestShardSimplificationsPass>();
}
} // namespace test
} // namespace mlir
