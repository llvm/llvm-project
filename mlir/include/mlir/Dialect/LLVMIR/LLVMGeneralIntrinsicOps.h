//===- LLVMGeneralIntrinsicOps.h - LLVM general intrinsics ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `GeneralIntrinsic` of the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMGENERALINTRINSICOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMGENERALINTRINSICOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

namespace mlir {
namespace LLVM {
struct AssumeAlignTag {};
struct AssumeSeparateStorageTag {};
} // namespace LLVM
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMGeneralIntrinsicOps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMGENERALINTRINSICOPS_H_
