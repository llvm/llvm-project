//===- X86Dialect.h - MLIR Dialect for X86 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for X86 in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_X86_X86DIALECT_H_
#define MLIR_DIALECT_X86_X86DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the generated interface declarations.
#include "mlir/Dialect/X86/X86Interfaces.h.inc"

#include "mlir/Dialect/X86/X86Dialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/X86/X86.h.inc"

#endif // MLIR_DIALECT_X86_X86DIALECT_H_
