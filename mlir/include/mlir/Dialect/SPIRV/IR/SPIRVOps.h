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

#ifndef MLIR_DIALECT_SPIRV_IR_SPIRVOPS_H_
#define MLIR_DIALECT_SPIRV_IR_SPIRVOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOpTraits.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "mlir/Dialect/SPIRV/IR/SPIRVAvailability.h.inc"

namespace mlir {
class OpBuilder;

namespace spirv {
class VerCapExtAttr;
} // namespace spirv
} // namespace mlir

// TablenGen'erated operation declarations.
#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h.inc"

namespace llvm {

/// Allow stealing the low bits of spirv::Function ops.
template <>
struct PointerLikeTypeTraits<mlir::spirv::FuncOp> {
public:
  static inline void *getAsVoidPointer(mlir::spirv::FuncOp i) {
    return const_cast<void *>(i.getAsOpaquePointer());
  }
  static inline mlir::spirv::FuncOp getFromVoidPointer(void *p) {
    return mlir::spirv::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};

} // namespace llvm

#endif // MLIR_DIALECT_SPIRV_IR_SPIRVOPS_H_
