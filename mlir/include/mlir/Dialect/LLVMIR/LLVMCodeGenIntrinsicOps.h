//===- LLVMCodeGenIntrinsicOps.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `CodeGenIntrinsic` of the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMCODEGENINTRINSICOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMCODEGENINTRINSICOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMCodeGenIntrinsicOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMCODEGENINTRINSICOPS_H_
