//===- LLVMInterfaces.h - OpenACC external models ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_EXTENSIONS_LLVMINTERFACES_H
#define MLIR_DIALECT_OPENACC_EXTENSIONS_LLVMINTERFACES_H

namespace mlir {
class DialectRegistry;
namespace acc {
/// Register OpenACC interfaces for LLVM IR. Loading OpenACC loads LLVM and
/// Arith so that these interfaces can generate IR.
void registerLLVMInterfaces(DialectRegistry &registry);
} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_EXTENSIONS_LLVMINTERFACES_H
