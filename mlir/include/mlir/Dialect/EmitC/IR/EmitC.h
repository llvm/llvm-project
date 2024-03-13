//===- EmitC.h - EmitC Dialect ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EmitC in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITC_H
#define MLIR_DIALECT_EMITC_IR_EMITC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/EmitC/IR/EmitCTraits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/EmitC/IR/EmitCDialect.h.inc"
#include "mlir/Dialect/EmitC/IR/EmitCEnums.h.inc"

namespace mlir {
namespace emitc {
void buildTerminatedBody(OpBuilder &builder, Location loc);
} // namespace emitc
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitC.h.inc"

#endif // MLIR_DIALECT_EMITC_IR_EMITC_H
