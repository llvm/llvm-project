//===- X86VectorToLLVMIRTranslation.h - X86Vector to LLVM IR ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for X86Vector dialect to LLVM IR
// translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_X86VECTOR_X86VECTORTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_X86VECTOR_X86VECTORTOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the X86Vector dialect and the translation from it to the LLVM IR
/// in the given registry;
void registerX86VectorDialectTranslation(DialectRegistry &registry);

/// Register the X86Vector dialect and the translation from it in the registry
/// associated with the given context.
void registerX86VectorDialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_X86VECTOR_X86VECTORTOLLVMIRTRANSLATION_H
