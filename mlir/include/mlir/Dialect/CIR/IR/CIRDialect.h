//===- CIRDialect.h - MLIR Dialect for CIR ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for CIR in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_CIRDIALECT_H_
#define MLIR_DIALECT_CIR_CIRDIALECT_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
using FuncOp = func::FuncOp;
} // namespace mlir

#include "mlir/Dialect/CIR/IR/CIROpsDialect.h.inc"
#include "mlir/Dialect/CIR/IR/CIROpsEnums.h.inc"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"

namespace mlir {
namespace cir {
void buildTerminatedBody(OpBuilder &builder, Location loc);
} // namespace cir
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/CIR/IR/CIROps.h.inc"

#endif // MLIR_DIALECT_CIR_CIRDIALECT_H_
