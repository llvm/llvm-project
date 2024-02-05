//===- Simplifications.h - Mesh Simplifications -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H
#define MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace mesh {

// Return the sharded shape `shape` acording ot sharding `sharding`.
ShapedType shardShapedType(ShapedType shape, MeshOp mesh,
                           MeshShardingAttr sharding);

// Insert resharding spmdization of the value `sourceShardValue`
// from sharding `source` to sharding `target`.
// `sourceShardValue` is the already sharded value according to `source`.
TypedValue<ShapedType> reshard(OpBuilder &builder, MeshOp mesh, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue);

void reshardingRegisterDependentDialects(DialectRegistry &registry);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H
