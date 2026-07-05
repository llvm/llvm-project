//===- LLVMIRToLLVMTranslation.h - LLVM IR to LLVM Dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for LLVM IR to LLVM dialect translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_LLVMIR_LLVMIRTOLLVMTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_LLVMIR_LLVMIRTOLLVMTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Registers the LLVM dialect and its import from LLVM IR in the given
/// registry.
void registerLLVMDialectImport(DialectRegistry &registry);

/// Registers the LLVM dialect and its import from LLVM IR with the given
/// context.
void registerLLVMDialectImport(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_LLVMIR_LLVMIRTOLLVMTRANSLATION_H
