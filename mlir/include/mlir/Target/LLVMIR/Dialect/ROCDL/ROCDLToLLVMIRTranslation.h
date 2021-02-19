//===- ROCDLToLLVMIRTranslation.h - ROCDL to LLVM IR ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect interface for translating the ROCDL
// dialect to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_ROCDL_ROCDLTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_ROCDL_ROCDLTOLLVMIRTRANSLATION_H

#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

namespace mlir {

/// Implementation of the dialect interface that converts operations belonging
/// to the ROCDL dialect to LLVM IR.
class ROCDLDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_ROCDL_ROCDLTOLLVMIRTRANSLATION_H
