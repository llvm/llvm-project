//===- ShardingPropagation.cpp ------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <vector>

namespace mlir {
namespace mesh {
#define GEN_PASS_DEF_SHARDINGPROPAGATION
#include "mlir/Dialect/Mesh/Transforms/Passes.h.inc"
} // namespace mesh
} // namespace mlir

#define DEBUG_TYPE "sharding-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static std::vector<Operation *> getOperationsVector(Block &block) {
  std::vector<Operation *> res;
  for (auto it = block.begin(); it != block.end(); ++it) {
    Operation *op = &*it;
    res.push_back(op);
  }
  return res;
}

static std::vector<Operation *> getReversedOperationsVector(Block &block) {
  std::vector<Operation *> res;
  for (auto it = block.rbegin(); it != block.rend(); ++it) {
    Operation *op = &*it;
    res.push_back(op);
  }
  return res;
}

// For each operation that implements the ShardingInterface, infer the sharding
// option of the operation from its operands and/or results using the
// `getShardingOption` method. If the inferred sharding option is not empty, add
// a `mesh.shard` operation for all remaining operands and results that do not
// have sharding annotations.
LogicalResult visitOp(Operation *op, OpBuilder &builder) {
  if (op->hasTrait<OpTrait::IsTerminator>() || llvm::isa<mesh::ShardOp>(op))
    return success();

  ShardingInterface shardingOp = llvm::dyn_cast<ShardingInterface>(op);
  if (!shardingOp) {
    op->emitOpError() << "sharding interface is not implemented.";
    return failure();
  }

  FailureOr<ShardingOption> shardingOption =
      shardingOp.getShardingOption(builder);
  if (failed(shardingOption)) {
    op->emitOpError() << "fail to get sharding option from results.";
    return failure();
  }
  // sharding info is empty, return immediately
  if (shardingOption->empty)
    return success();

  ArrayAttr shardingArrayAttr =
      builder.getArrayOfI32ArrayAttr(shardingOption->shardingArray);
  LLVM_DEBUG(DBGS() << "mesh cluster: " << shardingOption->cluster << "\n");
  LLVM_DEBUG(DBGS() << "sharding array: " << shardingArrayAttr << "\n");
  op->setAttr(getMeshClusterName(), shardingOption->cluster);
  op->setAttr(getShardingArrayName(),
              builder.getArrayOfI32ArrayAttr(shardingOption->shardingArray));

  if (failed(shardingOp.addShardingAnnotations(builder, *shardingOption))) {
    op->emitOpError() << "fail to set sharding annotations.";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShardingPropagationPass
//===----------------------------------------------------------------------===//
struct ShardingPropagationPass
    : public mesh::impl::ShardingPropagationBase<ShardingPropagationPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Region &region = funcOp.getBody();
    OpBuilder builder(ctx);
    if (!region.hasOneBlock()) {
      funcOp.emitOpError() << "only one block is supported!";
      signalPassFailure();
    }
    Block &block = region.front();

    // clang-format off
    LLVM_DEBUG(
      DBGS() << "print all the ops' iterator types and indexing maps in the "
                "block.\n";
      DenseSet<ShardingInterface> ops;
      for (Operation &op : block.getOperations()) {
        if (auto shardingOp = llvm::dyn_cast<ShardingInterface>(&op)) {
          ops.insert(shardingOp);
        }
      }
      for (ShardingInterface shardingOp : ops) {
        shardingOp.printLoopTypesAndIndexingMaps(llvm::dbgs());
      }
    );
    // clang-format on

    // 1. propagate in reversed order
    {
      std::vector<Operation *> curOps = getReversedOperationsVector(block);
      for (Operation *op : curOps) {
        if (failed(visitOp(op, builder)))
          return signalPassFailure();
      }
    }

    LLVM_DEBUG(DBGS() << "After reversed order propagation:\n"
                      << funcOp << "\n");

    // 2. propagate in original order
    {
      std::vector<Operation *> curOps = getOperationsVector(block);
      for (Operation *op : curOps) {
        if (failed(visitOp(op, builder)))
          return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::mesh::createShardingPropagationPass() {
  return std::make_unique<ShardingPropagationPass>();
}
