//===- MemRefTransformOps.h - MemRef transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMOPS_MEMREFTRANSFORMOPS_H
#define MLIR_DIALECT_MEMREF_TRANSFORMOPS_MEMREFTRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace memref {
class AllocOp;
} // namespace memref
namespace transform {
class OperationType;
} // namespace transform
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace memref {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMOPS_MEMREFTRANSFORMOPS_H
