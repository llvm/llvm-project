//===- MemRef.h - MemRef dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_IR_MEMREF_H_
#define MLIR_DIALECT_MEMREF_IR_MEMREF_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// MemRef Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRefOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// MemRef Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/IR/MemRefOps.h.inc"

#endif // MLIR_DIALECT_MEMREF_IR_MEMREF_H_
