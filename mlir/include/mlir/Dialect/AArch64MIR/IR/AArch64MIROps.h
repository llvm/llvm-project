//===- AArch64MIROps.h - AArch64 MIR operations ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIROPS_H
#define MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/AArch64MIR/IR/AArch64MIRDialect.h"
#include "mlir/Dialect/MIR/IR/MIRTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/AArch64MIR/IR/AArch64MIROps.h.inc"

#endif // MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIROPS_H
