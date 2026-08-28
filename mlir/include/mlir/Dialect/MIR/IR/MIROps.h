//===- MIROps.h - MIR dialect operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIR_IR_MIROPS_H
#define MLIR_DIALECT_MIR_IR_MIROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/MIR/IR/MIRAttrs.h"
#include "mlir/Dialect/MIR/IR/MIRDialect.h"
#include "mlir/Dialect/MIR/IR/MIRTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIR/IR/MIROps.h.inc"

#endif // MLIR_DIALECT_MIR_IR_MIROPS_H
