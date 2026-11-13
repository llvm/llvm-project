//===- LLVMCoroIntrinsicOps.h - LLVM coroutine intrinsics -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `CoroIntrinsic` of the LLVM
// dialect: the coroutine intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMCOROINTRINSICOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMCOROINTRINSICOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMCoroIntrinsicOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMCOROINTRINSICOPS_H_
