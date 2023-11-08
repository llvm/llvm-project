//===- LLVMIRToNVVMTranslation.h - LLVM IR to NVVM Dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for LLVM IR to NVVM dialect translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_NVVM_LLVMIRTONVVMTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_NVVM_LLVMIRTONVVMTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Registers the NVVM dialect and its import from LLVM IR in the given
/// registry.
void registerNVVMDialectImport(DialectRegistry &registry);

/// Registers the NVVM dialect and its import from LLVM IR with the given
/// context.
void registerNVVMDialectImport(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_NVVM_LLVMIRTONVVMTRANSLATION_H
