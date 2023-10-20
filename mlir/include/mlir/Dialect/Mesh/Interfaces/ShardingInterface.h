//===- ShardingInterface.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_
#define MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;

namespace mesh {

using ShardingArray = SmallVector<SmallVector<int32_t>>;
using ShardingArrayRef = ArrayRef<SmallVector<int32_t>>;

struct ShardingOption {
  // An array of int array. The sub-array at the i-th position signifies the
  // mesh axes the i-th loop will be sharded on.
  ShardingArray shardingArray;
  SymbolRefAttr cluster;
  // `empty` is true indicates that no sharding infomation can be inferred at
  // present. Note that it is different from that an operation is not sharded.
  bool empty = false;
  ShardingOption() = default;
  ShardingOption(ShardingArray shardingArray, SymbolRefAttr cluster)
      : shardingArray(std::move(shardingArray)), cluster(cluster) {}
};

namespace detail {

FailureOr<ShardingOption> defaultGetShardingOption(Operation *op);

LogicalResult
defaultAddShardingAnnotations(Operation *op, OpBuilder &b,
                              const ShardingOption &shardingOption);

} // namespace detail

} // namespace mesh

} // namespace mlir

/// Include the ODS generated interface header files.
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h.inc"

#endif // MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_
