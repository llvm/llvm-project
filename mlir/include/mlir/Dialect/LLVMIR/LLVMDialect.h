//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc"

namespace llvm {
class Type;
class LLVMContext;
namespace sys {
template <bool mt_only>
class SmartMutex;
} // end namespace sys
} // end namespace llvm

namespace mlir {
namespace LLVM {
class LLVMDialect;

namespace detail {
struct LLVMTypeStorage;
struct LLVMDialectImpl;
} // namespace detail

} // namespace LLVM
} // namespace mlir

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.h.inc"

#include "mlir/Dialect/LLVMIR/LLVMOpsDialect.h.inc"

namespace mlir {
namespace LLVM {
/// Create an LLVM global containing the string "value" at the module containing
/// surrounding the insertion point of builder. Obtain the address of that
/// global and use it to compute the address of the first character in the
/// string (operations inserted at the builder insertion point).
Value createGlobalString(Location loc, OpBuilder &builder, StringRef name,
                         StringRef value, Linkage linkage);

/// LLVM requires some operations to be inside of a Module operation. This
/// function confirms that the Operation has the desired properties.
bool satisfiesLLVMModule(Operation *op);

} // end namespace LLVM
} // end namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
