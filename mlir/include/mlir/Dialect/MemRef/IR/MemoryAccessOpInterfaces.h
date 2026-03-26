//===- MemoryAccessOpInterfaces.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_IR_MEMORYACCESSOPINTERFACES_H
#define MLIR_DIALECT_MEMREF_IR_MEMORYACCESSOPINTERFACES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class RewriterBase;

namespace memref::detail {
LogicalResult verifyIndexedAccessOpInterface(Operation *op);
LogicalResult verifyIndexedMemCopyOpInterface(Operation *op);
} // namespace memref::detail
} // namespace mlir

//===----------------------------------------------------------------------===//
// Memory Access Op Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h.inc"

#endif // MLIR_DIALECT_MEMREF_IR_MEMORYACCESSOPINTERFACES_H
