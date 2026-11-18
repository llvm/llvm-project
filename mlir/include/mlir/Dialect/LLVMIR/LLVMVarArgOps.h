//===- LLVMVarArgOps.h - LLVM dialect variadic argument ops -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `VarArg` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMVARARGOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMVARARGOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMVarArgOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMVARARGOPS_H_
