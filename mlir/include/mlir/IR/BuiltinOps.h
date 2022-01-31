//===- BuiltinOps.h - MLIR Builtin Operations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Builtin dialect's operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINOPS_H_
#define MLIR_IR_BUILTINOPS_H_

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

//===----------------------------------------------------------------------===//
// Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/IR/BuiltinOps.h.inc"

namespace llvm {
/// Allow stealing the low bits of FuncOp.
template <>
struct PointerLikeTypeTraits<mlir::FuncOp> {
  static inline void *getAsVoidPointer(mlir::FuncOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline mlir::FuncOp getFromVoidPointer(void *p) {
    return mlir::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};

/// Allow stealing the low bits of ModuleOp.
template <>
struct PointerLikeTypeTraits<mlir::ModuleOp> {
public:
  static inline void *getAsVoidPointer(mlir::ModuleOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline mlir::ModuleOp getFromVoidPointer(void *p) {
    return mlir::ModuleOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};
} // namespace llvm

#endif // MLIR_IR_BUILTINOPS_H_
