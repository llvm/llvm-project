//===- SPIRVOps.h - MLIR SPIR-V operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVOPS_H_
#define MLIR_DIALECT_SPIRV_SPIRVOPS_H_

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class OpBuilder;

namespace spirv {
class VerCapExtAttr;

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "mlir/Dialect/SPIRV/SPIRVAvailability.h.inc"
} // namespace spirv
} // namespace mlir

// TablenGen'erated operation declarations.
#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/SPIRVOps.h.inc"

namespace mlir {
namespace spirv {
// TableGen'erated helper functions.
//
// Get the name used in the Op to refer to an enum value of the given
// `EnumClass`.
// template <typename EnumClass> StringRef attributeName();
//
#include "mlir/Dialect/SPIRV/SPIRVOpUtils.inc"

} // end namespace spirv
} // end namespace mlir

namespace llvm {

/// spirv::Function ops hash just like pointers.
template <>
struct DenseMapInfo<mlir::spirv::FuncOp> {
  static mlir::spirv::FuncOp getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::spirv::FuncOp::getFromOpaquePointer(pointer);
  }
  static mlir::spirv::FuncOp getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::spirv::FuncOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::spirv::FuncOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::spirv::FuncOp LHS, mlir::spirv::FuncOp RHS) {
    return LHS == RHS;
  }
};

/// Allow stealing the low bits of spirv::Function ops.
template <>
struct PointerLikeTypeTraits<mlir::spirv::FuncOp> {
public:
  static inline void *getAsVoidPointer(mlir::spirv::FuncOp I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::spirv::FuncOp getFromVoidPointer(void *P) {
    return mlir::spirv::FuncOp::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

} // namespace llvm

#endif // MLIR_DIALECT_SPIRV_SPIRVOPS_H_
