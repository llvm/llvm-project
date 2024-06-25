//===- TestSimplification.cpp - Test simplification -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Transforms/Spmdization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::mesh;

namespace {

struct TestMeshReshardingRewritePattern : OpRewritePattern<ShardOp> {
  using OpRewritePattern<ShardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShardOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAnnotateForUsers()) {
      return failure();
    }

    SymbolTableCollection symbolTable;
    mesh::MeshOp mesh = symbolTable.lookupNearestSymbolFrom<mesh::MeshOp>(
        op, op.getShard().getMesh());

    bool foundUser = false;
    for (auto user : op->getUsers()) {
      if (auto targetShardOp = llvm::dyn_cast<ShardOp>(user)) {
        if (targetShardOp.getAnnotateForUsers() &&
            mesh == symbolTable.lookupNearestSymbolFrom<mesh::MeshOp>(
                        targetShardOp, targetShardOp.getShard().getMesh())) {
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
          symbolTable.lookupNearestSymbolFrom<mesh::MeshOp>(
              targetShardOp, targetShardOp.getShard().getMesh()) != mesh) {
        continue;
      }

      ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
      ShapedType sourceShardShape =
          shardShapedType(op.getResult().getType(), mesh, op.getShard());
      TypedValue<ShapedType> sourceShard = cast<TypedValue<ShapedType>>(
          builder
              .create<UnrealizedConversionCastOp>(sourceShardShape,
                                                  op.getOperand())
              ->getResult(0));
      TypedValue<ShapedType> targetShard =
          reshard(builder, mesh, op, targetShardOp, sourceShard);
      Value newTargetUnsharded =
          builder
              .create<UnrealizedConversionCastOp>(
                  targetShardOp.getResult().getType(), targetShard)
              ->getResult(0);
      rewriter.replaceAllUsesWith(targetShardOp.getResult(),
                                  newTargetUnsharded);
    }

    return success();
  }
};

struct TestMeshReshardingPass
    : public PassWrapper<TestMeshReshardingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMeshReshardingPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<TestMeshReshardingRewritePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation().getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    reshardingRegisterDependentDialects(registry);
    registry.insert<BuiltinDialect>();
  }
  StringRef getArgument() const final {
    return "test-mesh-resharding-spmdization";
  }
  StringRef getDescription() const final {
    return "Test Mesh dialect resharding spmdization.";
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestMeshReshardingSpmdizationPass() {
  PassRegistration<TestMeshReshardingPass>();
}
} // namespace test
} // namespace mlir
