//===- Simplifications.h - Shard Simplifications ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHARD_TRANSFORMS_PARTITION_H
#define MLIR_DIALECT_SHARD_TRANSFORMS_PARTITION_H

#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace shard {

// Insert resharding partition of the value `sourceShardValue`
// from sharding `source` to sharding `target`.
// `sourceShardValue` is the already sharded value according to `source`.
//
// Example
//
// ```mlir
//   shard.grid @grid_1d(shape = 2)
//   ...
//   %1 = shard.shard %0 to <@grid_1d, [[0]]> : tensor<2xi8>
//   %2 = shard.shard %1 to <@grid_1d, [[]]> annotate_for_users: tensor<2xi8>
// ```
//
// Will result in
//
// ```mlir
//   %1 = shard.all_gather %0 on @grid_1d grid_axes = [0] gather_axis = 0 :
//     tensor<1xi8> -> tensor<2xi8>
// ```
TypedValue<ShapedType> reshard(OpBuilder &builder, GridOp grid, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue);
TypedValue<ShapedType> reshard(OpBuilder &builder, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue,
                               SymbolTableCollection &symbolTableCollection);

void reshardingRegisterDependentDialects(DialectRegistry &registry);

} // namespace shard
} // namespace mlir

#endif // MLIR_DIALECT_SHARD_TRANSFORMS_PARTITION_H
