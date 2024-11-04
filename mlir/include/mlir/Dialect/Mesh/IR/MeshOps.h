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
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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

bool isReductionLoop(IteratorType iType);

bool areReductionAndPartialMatch(IteratorType iType, Partial partial);

template <typename T>
void removeTrailingEmptySubArray(SmallVector<SmallVector<T>> &array) {
  while (!array.empty() && array.back().empty())
    array.pop_back();
}

Partial getPartialTypeFromReduction(IteratorType iType);

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

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_IR_MESHOPS_H
