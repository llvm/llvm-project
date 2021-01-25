//===- AsyncTypes.h - Async Dialect Types -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the Async dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_IR_ASYNCTYPES_H_
#define MLIR_DIALECT_ASYNC_IR_ASYNCTYPES_H_

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Async Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOpsTypes.h.inc"

#endif // MLIR_DIALECT_ASYNC_IR_ASYNCTYPES_H_
