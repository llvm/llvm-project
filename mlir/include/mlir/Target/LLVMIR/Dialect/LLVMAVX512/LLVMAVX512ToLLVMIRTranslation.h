//===- LLVMAVX512ToLLVMIRTranslation.h - LLVMAVX512 to LLVM IR --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect interface for translating the LLVMAVX512
// dialect to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_LLVMAVX512_LLVMAVX512TOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_LLVMAVX512_LLVMAVX512TOLLVMIRTRANSLATION_H

#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

namespace mlir {

/// Implementation of the dialect interface that converts operations belonging
/// to the LLVMAVX512 dialect to LLVM IR.
class LLVMAVX512DialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_LLVMAVX512_LLVMAVX512TOLLVMIRTRANSLATION_H
