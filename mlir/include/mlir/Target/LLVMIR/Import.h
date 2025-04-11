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

#include "mlir/IR/OwningOpRef.h"
#include <memory>

// Forward-declare LLVM classes.
namespace llvm {
class DataLayout;
class Module;
} // namespace llvm

namespace mlir {

class DataLayoutSpecInterface;
class MLIRContext;
class ModuleOp;

/// Translates the LLVM module into an MLIR module living in the given context.
/// The translation supports operations from any dialect that has a registered
/// implementation of the LLVMImportDialectInterface. It returns nullptr if the
/// translation fails and reports errors using the error handler registered with
/// the MLIR context.
/// The `emitExpensiveWarnings` option controls if expensive
/// but uncritical diagnostics should be emitted.
/// The `dropDICompositeTypeElements` option controls if DICompositeTypes should
/// be imported without elements. If set, the option avoids the recursive
/// traversal of composite type debug information, which can be expensive for
/// adversarial inputs.
/// The `loadAllDialects` flag (default on) will load all dialects in the
/// context.
/// The `preferUnregisteredIntrinsics` flag (default off) controls whether to
/// import all intrinsics using `llvm.intrinsic_call` even if a dialect
/// registered an explicit intrinsic operation. Warning: passes that rely on
/// matching explicit intrinsic operations may not work properly if this flag is
/// enabled.
OwningOpRef<ModuleOp> translateLLVMIRToModule(
    std::unique_ptr<llvm::Module> llvmModule, MLIRContext *context,
    bool emitExpensiveWarnings = true, bool dropDICompositeTypeElements = false,
    bool loadAllDialects = true, bool preferUnregisteredIntrinsics = false);

/// Translate the given LLVM data layout into an MLIR equivalent using the DLTI
/// dialect.
DataLayoutSpecInterface translateDataLayout(const llvm::DataLayout &dataLayout,
                                            MLIRContext *context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_IMPORT_H
