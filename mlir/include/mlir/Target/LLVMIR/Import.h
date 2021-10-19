//===- Import.h - LLVM IR To MLIR translation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry point for the LLVM IR to MLIR conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_IMPORT_H
#define MLIR_TARGET_LLVMIR_IMPORT_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

// Forward-declare LLVM classes.
namespace llvm {
class Module;
} // namespace llvm

namespace mlir {

class DialectRegistry;
class OwningModuleRef;
class MLIRContext;

/// Convert the given LLVM module into MLIR's LLVM dialect.  The LLVM context is
/// extracted from the registered LLVM IR dialect. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `{}`.
OwningModuleRef
translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                        MLIRContext *context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_IMPORT_H
