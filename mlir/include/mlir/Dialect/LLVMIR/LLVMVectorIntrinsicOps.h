//===- LLVMVectorIntrinsicOps.h - LLVM vector intrinsics --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `VectorIntrinsic` of the LLVM
// dialect: the vector intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMVECTORINTRINSICOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMVECTORINTRINSICOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMVectorIntrinsicOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMVECTORINTRINSICOPS_H_
