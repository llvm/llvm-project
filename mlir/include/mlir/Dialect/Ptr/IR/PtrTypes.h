//===- PtrTypes.h - Pointer types -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Pointer dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_PTRTYPES_H
#define MLIR_DIALECT_PTR_IR_PTRTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.h.inc"

#endif // MLIR_DIALECT_PTR_IR_PTRTYPES_H
