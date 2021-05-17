//===- Export.h - MLIR to LLVM IR translation entry point -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_EXPORT_H
#define MLIR_TARGET_LLVMIR_EXPORT_H

#include "llvm/ADT/StringRef.h"
#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class Operation;

/// Translate operation that satisfies LLVM dialect module requirements into an
/// LLVM IR module living in the given context. This translates operations from
/// any dilalect that has a registered implementation of
/// LLVMTranslationDialectInterface.
std::unique_ptr<llvm::Module>
translateModuleToLLVMIR(Operation *module, llvm::LLVMContext &llvmContext,
                        llvm::StringRef name = "LLVMDialectModule");
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_EXPORT_H
