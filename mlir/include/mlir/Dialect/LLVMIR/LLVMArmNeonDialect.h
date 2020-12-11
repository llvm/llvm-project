//===- LLVMArmNeonDialect.h - MLIR Dialect for LLVMArmNeon ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for LLVMArmNeon in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMARMNEONDIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMARMNEONDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMArmNeon.h.inc"

#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMARMNEONDIALECT_H_
