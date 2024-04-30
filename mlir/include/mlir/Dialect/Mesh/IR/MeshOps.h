//===- MeshOps.h - Mesh Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_IR_MESHOPS_H
#define MLIR_DIALECT_MESH_IR_MESHOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"

namespace mlir {
namespace mesh {

using MeshAxis = int16_t;
using MeshAxesAttr = DenseI16ArrayAttr;

} // namespace mesh
} // namespace mlir

#include "mlir/Dialect/Mesh/IR/MeshEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.h.inc"

namespace mlir {
namespace mesh {

inline bool isReductionLoop(utils::IteratorType iType) {
  return iType == utils::IteratorType::reduction;
}

template <typename T>
void removeTrailingEmptySubArray(SmallVector<SmallVector<T>> &array) {
  while (!array.empty() && array.back().empty())
    array.pop_back();
}

// Is the same tensor replicated on all processes.
inline bool isFullReplication(MeshShardingAttr attr) {
  return attr.getPartialAxes().empty() && attr.getSplitAxes().empty();
}

inline mesh::MeshOp getMesh(Operation *op, FlatSymbolRefAttr meshSymbol,
                            SymbolTableCollection &symbolTableCollection) {
  return symbolTableCollection.lookupNearestSymbolFrom<mesh::MeshOp>(
      op, meshSymbol);
}

// Get the corresponding mesh op using the standard attribute nomenclature.
template <typename Op>
mesh::MeshOp getMesh(Op op, SymbolTableCollection &symbolTableCollection) {
  return getMesh(op.getOperation(), op.getMeshAttr(), symbolTableCollection);
}

template <>
inline mesh::MeshOp
getMesh<ShardOp>(ShardOp op, SymbolTableCollection &symbolTableCollection) {
  return getMesh(op.getOperation(), op.getShardAttr().getMesh(),
                 symbolTableCollection);
}

// Get the number of processes that participate in each group
// induced by `meshAxes`.
template <typename MeshAxesRange, typename MeshShapeRange>
int64_t collectiveProcessGroupSize(MeshAxesRange &&meshAxes,
                                   MeshShapeRange &&meshShape) {
  int64_t res = 1;

  for (MeshAxis axis : meshAxes) {
    auto axisSize = *(std::begin(meshShape) + axis);
    if (ShapedType::isDynamic(axisSize)) {
      return ShapedType::kDynamic;
    }
    res *= axisSize;
  }

  return res;
}

template <typename MeshAxesRange>
int64_t collectiveProcessGroupSize(MeshAxesRange &&meshAxes, MeshOp mesh) {
  return collectiveProcessGroupSize(std::forward<MeshAxesRange>(meshAxes),
                                    mesh.getShape());
}

// Get the size of a sharded dimension.
inline int64_t shardDimension(int64_t dimSize, int64_t shardCount) {
  if (ShapedType::isDynamic(dimSize) || ShapedType::isDynamic(shardCount))
    return ShapedType::kDynamic;

  assert(dimSize % shardCount == 0);
  return ceilDiv(dimSize, shardCount);
}

// Get the size of an unsharded dimension.
inline int64_t gatherDimension(int64_t dimSize, int64_t shardCount) {
  if (ShapedType::isDynamic(dimSize) || ShapedType::isDynamic(shardCount))
    return ShapedType::kDynamic;

  return dimSize * shardCount;
}

// Return the sharded shape `shape` according ot sharding `sharding`.
// The shape for the tensor on each device in the mesh.
// Example:
// On a 2x4x? mesh with split axes = [[0], [1], [2]] the shape ?x5x1 would
// result in a shape for each shard of ?x2x?.
ShapedType shardShapedType(ShapedType shape, MeshOp mesh,
                           MeshShardingAttr sharding);

// If ranked tensor type return its sharded counterpart.
//
// If not ranked tensor type return `type`.
// `sharding` in that case must be null.
Type shardType(Type type, MeshOp mesh, MeshShardingAttr sharding);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_IR_MESHOPS_H
