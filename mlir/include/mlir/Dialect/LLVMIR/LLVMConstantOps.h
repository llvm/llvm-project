//===- LLVMConstantOps.h - LLVM dialect constant ops ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `Constant` of the LLVM dialect:
// the constant-like ops and `llvm.mlir.none`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMCONSTANTOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMCONSTANTOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h.inc"

namespace mlir {
namespace LLVM {
/// Determines the element type of `type` the way the `llvm.mlir.constant`
/// verifier does, i.e. by looking through LLVM array types and then through a
/// `VectorType` or `TensorType`. Everything else is treated as a scalar. Use
/// this when building a constant so that the attribute and the result type are
/// compared consistently with the verifier.
Type getConstantElementType(Type type);
} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMCONSTANTOPS_H_
