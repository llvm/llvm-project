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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Mesh/IR/MeshOpsDialect.h.inc"

#include "mlir/Dialect/Mesh/IR/MeshOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.h.inc"

namespace mlir {
namespace mesh {

bool isReductionLoop(IteratorType iType);

bool areReductionAndPartialMatch(IteratorType iType, Partial partial);

template <typename T>
void removeTrailingEmptySubArray(SmallVector<SmallVector<T>> &array) {
  for (int64_t i = array.size() - 1; i >= 0; i--) {
    if (array[i].empty())
      array.pop_back();
    else
      break;
  }
}

Partial getPartialTypeFromReduction(IteratorType iType);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_IR_MESHOPS_H
