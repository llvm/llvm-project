//===- NVVMArithOps.h - NVVM dialect arithmetic ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `Arith` of the NVVM dialect: the
// floating-point math, byte permutation and dot product ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMARITHOPS_H_
#define MLIR_DIALECT_LLVMIR_NVVMARITHOPS_H_

#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/NVVMIntrinsicLowering.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMArithOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMARITHOPS_H_
