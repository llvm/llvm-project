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

  FailureOr<ShardingOption> shardingOption = shardingOp.getShardingOption();
  if (failed(shardingOption)) {
    op->emitOpError() << "fail to get sharding option from results.";
    return failure();
  }
  // sharding info is empty, return immediately
  if (shardingOption->empty)
    return success();

  if (failed(shardingOp.addShardingAnnotations(builder, *shardingOption))) {
    op->emitOpError() << "fail to set sharding annotations.";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShardingPropagation
//===----------------------------------------------------------------------===//
struct ShardingPropagation
    : public mesh::impl::ShardingPropagationBase<ShardingPropagation> {
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

    LLVM_DEBUG(
        DBGS() << "print all the ops' iterator types and indexing maps in the "
                  "block.\n";
        for (Operation &op
             : block.getOperations()) {
          if (auto shardingOp = llvm::dyn_cast<ShardingInterface>(&op))
            shardingOp.printLoopTypesAndIndexingMaps(llvm::dbgs());
        });

    // 1. propagate in reversed order
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
      if (failed(visitOp(&op, builder)))
        return signalPassFailure();

    LLVM_DEBUG(DBGS() << "After reversed order propagation:\n"
                      << funcOp << "\n");

    // 2. propagate in original order
    for (Operation &op : llvm::make_early_inc_range(block))
      if (failed(visitOp(&op, builder)))
        return signalPassFailure();
  }
};

} // namespace
