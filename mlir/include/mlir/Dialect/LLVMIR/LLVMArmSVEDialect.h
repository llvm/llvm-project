//===- LLVMSVEDialect.h - MLIR Dialect for LLVMSVE --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for LLVMArmSVE in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMARMSVEDIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMARMSVEDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMArmSVE.h.inc"

#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMARMSVEDIALECT_H_
