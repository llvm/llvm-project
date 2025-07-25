//===- TestSimplification.cpp - Test simplification -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Transforms/Partition.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::shard;

namespace {

struct TestReshardingRewritePattern : OpRewritePattern<ShardOp> {
  using OpRewritePattern<ShardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShardOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAnnotateForUsers()) {
      return failure();
    }

    SymbolTableCollection symbolTable;
    shard::GridOp grid = symbolTable.lookupNearestSymbolFrom<shard::GridOp>(
        op, cast<ShardingOp>(op.getSharding().getDefiningOp()).getGridAttr());

    bool foundUser = false;
    for (auto user : op->getUsers()) {
      if (auto targetShardOp = llvm::dyn_cast<ShardOp>(user)) {
        if (targetShardOp.getAnnotateForUsers() &&
            grid == symbolTable.lookupNearestSymbolFrom<shard::GridOp>(
                        targetShardOp,
                        cast<ShardingOp>(
                            targetShardOp.getSharding().getDefiningOp())
                            .getGridAttr())) {
          foundUser = true;
          break;
        }
      }
    }

    if (!foundUser) {
      return failure();
    }

    for (auto user : op->getUsers()) {
      auto targetShardOp = llvm::dyn_cast<ShardOp>(user);
      if (!targetShardOp || !targetShardOp.getAnnotateForUsers() ||
          symbolTable.lookupNearestSymbolFrom<shard::GridOp>(
              targetShardOp,
              cast<ShardingOp>(targetShardOp.getSharding().getDefiningOp())
                  .getGridAttr()) != grid) {
        continue;
      }

      ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
      ShapedType sourceShardShape =
          shardShapedType(op.getResult().getType(), grid, op.getSharding());
      TypedValue<ShapedType> sourceShard = cast<TypedValue<ShapedType>>(
          UnrealizedConversionCastOp::create(builder, sourceShardShape,
                                             op.getSrc())
              ->getResult(0));
      TypedValue<ShapedType> targetShard =
          reshard(builder, grid, op, targetShardOp, sourceShard);
      Value newTargetUnsharded =
          UnrealizedConversionCastOp::create(
              builder, targetShardOp.getResult().getType(), targetShard)
              ->getResult(0);
      rewriter.replaceAllUsesWith(targetShardOp.getResult(),
                                  newTargetUnsharded);
    }

    return success();
  }
};

struct TestReshardingPass
    : public PassWrapper<TestReshardingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestReshardingPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<TestReshardingRewritePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation().getOperation(),
                                     std::move(patterns)))) {
      return signalPassFailure();
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    reshardingRegisterDependentDialects(registry);
    registry.insert<BuiltinDialect>();
  }
  StringRef getArgument() const final {
    return "test-grid-resharding-partition";
  }
  StringRef getDescription() const final {
    return "Test Shard dialect resharding partition.";
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestReshardingPartitionPass() {
  PassRegistration<TestReshardingPass>();
}
} // namespace test
} // namespace mlir
