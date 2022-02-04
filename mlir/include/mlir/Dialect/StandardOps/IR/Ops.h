//===- Ops.h - Standard MLIR Operations -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with standard operations
// in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARDOPS_IR_OPS_H
#define MLIR_DIALECT_STANDARDOPS_IR_OPS_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/StandardOps/IR/OpsEnums.h.inc"

namespace mlir {
class AffineMap;
class Builder;
class FuncOp;
class OpBuilder;
class PatternRewriter;
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/StandardOps/IR/Ops.h.inc"

#include "mlir/Dialect/StandardOps/IR/OpsDialect.h.inc"

#endif // MLIR_DIALECT_STANDARDOPS_IR_OPS_H
