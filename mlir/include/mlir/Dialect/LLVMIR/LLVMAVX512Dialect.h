//===- LLVMAVX512Dialect.h - MLIR Dialect for LLVMAVX512 --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for LLVMAVX512 in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMAVX512DIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMAVX512DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMAVX512.h.inc"

#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMAVX512DIALECT_H_
