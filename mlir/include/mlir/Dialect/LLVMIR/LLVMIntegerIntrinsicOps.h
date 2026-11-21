//===- LLVMIntegerIntrinsicOps.h - LLVM integer intrinsics ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `IntegerIntrinsic` of the LLVM
// dialect: the integer intrinsics, e.g. `llvm.intr.ctpop`, `llvm.intr.smax`,
// `llvm.intr.sadd.with.overflow` and `llvm.intr.sadd.sat`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMINTEGERINTRINSICOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMINTEGERINTRINSICOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMIntegerIntrinsicOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMINTEGERINTRINSICOPS_H_
