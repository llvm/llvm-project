//===- ShardingInterface.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHARD_INTERFACES_SHARDINGINTERFACE_H_
#define MLIR_DIALECT_SHARD_INTERFACES_SHARDINGINTERFACE_H_

#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
class IRMapping;
class SymbolTableCollection;

namespace shard {

using ShardingArray = SmallVector<SmallVector<GridAxis>>;
using ShardingArrayRef = ArrayRef<SmallVector<GridAxis>>;

struct ShardingOption {
  // An array of int array. The sub-array at the i-th position signifies the
  // grid axes the i-th loop will be sharded on.
  ShardingArray shardingArray = {};
  FlatSymbolRefAttr grid = nullptr;
  // `empty` being true indicates that no sharding information can be inferred
  // at present. Note that it is different from the case where an operation is
  // not sharded.
  bool empty = false;
  ShardingOption() = default;
  ShardingOption(ShardingArray shardingArray, FlatSymbolRefAttr grid)
      : shardingArray(std::move(shardingArray)), grid(grid) {
    assert(this->grid);
  }
  static ShardingOption makeEmpty() {
    auto res = ShardingOption();
    res.empty = true;
    return res;
  }
};

// This method retrieves the 'Sharding' from a given operation
// result and includes the 'annotate_for_users' information.
FailureOr<std::pair<bool, Sharding>> getSharding(OpResult result);

// This method retrieves the 'Sharding' from a given operation
// operand and includes the 'annotate_for_users' information.
FailureOr<std::pair<bool, Sharding>> getSharding(OpOperand &opOperand);

namespace detail {

FailureOr<ShardingOption>
defaultGetShardingOption(Operation *op, ArrayRef<Sharding> operandShardings,
                         ArrayRef<Sharding> resultShardings);

FailureOr<std::vector<Sharding>>
defaultGetShardingAnnotations(Operation *op,
                              const ShardingOption &shardingOption);

LogicalResult
defaultAddShardingAnnotations(Operation *op, OpBuilder &b,
                              const ShardingOption &shardingOption);

} // namespace detail

// Assumes full replication on all ranked tensor arguments and results.
void partitionFullyReplicatedOperation(Operation &op,
                                       ArrayRef<Value> partitionedOperands,
                                       ArrayRef<Sharding> operandShardings,
                                       ArrayRef<Sharding> resultShardings,
                                       IRMapping &partitionMap,
                                       SymbolTableCollection &symbolTable,
                                       OpBuilder &builder);

} // namespace shard
} // namespace mlir

/// Include the ODS generated interface header files.
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h.inc"

#endif // MLIR_DIALECT_SHARD_INTERFACES_SHARDINGINTERFACE_H_
