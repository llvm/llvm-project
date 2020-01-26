//===- ROCDLIR.h - MLIR to LLVM + ROCDL IR conversion -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry point for the MLIR to LLVM + ROCDL IR
// conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_ROCDLIR_H
#define MLIR_TARGET_ROCDLIR_H

#include <memory>

// Forward-declare LLVM classes.
namespace llvm {
class Module;
} // namespace llvm

namespace mlir {
class Operation;

/// Convert the given LLVM-module-like operation into ROCDL IR. This conversion
/// requires the registration of the LLVM IR dialect and will extract the LLVM
/// context from the registered LLVM IR dialect.  In case of error, report it to
/// the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::Module> translateModuleToROCDLIR(Operation *m);

} // namespace mlir

#endif // MLIR_TARGET_ROCDLIR_H
