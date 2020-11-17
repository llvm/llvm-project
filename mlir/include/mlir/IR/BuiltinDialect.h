//===- BuiltinDialect.h - MLIR Builtin Dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Builtin dialect and its operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINDIALECT_H_
#define MLIR_IR_BUILTINDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OwningOpRefBase.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h.inc"

//===----------------------------------------------------------------------===//
// Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/IR/BuiltinOps.h.inc"

//===----------------------------------------------------------------------===//
// Dialect Utilities
//===----------------------------------------------------------------------===//

namespace mlir {
/// This class acts as an owning reference to a module, and will automatically
/// destroy the held module on destruction if the held module is valid.
class OwningModuleRef : public OwningOpRefBase<ModuleOp> {
public:
  using OwningOpRefBase<ModuleOp>::OwningOpRefBase;
};
} // end namespace mlir

namespace llvm {
// Functions hash just like pointers.
template <>
struct DenseMapInfo<mlir::FuncOp> {
  static mlir::FuncOp getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static mlir::FuncOp getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::FuncOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::FuncOp lhs, mlir::FuncOp rhs) { return lhs == rhs; }
};

/// Allow stealing the low bits of FuncOp.
template <>
struct PointerLikeTypeTraits<mlir::FuncOp> {
  static inline void *getAsVoidPointer(mlir::FuncOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline mlir::FuncOp getFromVoidPointer(void *p) {
    return mlir::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable = 3;
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
  static constexpr int NumLowBitsAvailable = 3;
};
} // end namespace llvm

#endif // MLIR_IR_BUILTINDIALECT_H_
