//===- ExportSMTLIB.h - SMT-LIB Exporter ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the SMT-LIB emitter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_EXPORTSMTLIB_H
#define MLIR_TARGET_EXPORTSMTLIB_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
namespace smt {

/// Emission options for the ExportSMTLIB pass. Allows controlling the emitted
/// format and overall behavior.
struct SMTEmissionOptions {
  // Don't produce 'let' expressions to bind expressions that are only used
  // once, but inline them directly at the use-site.
  bool inlineSingleUseValues = false;
  // Increase indentation for each 'let' expression body.
  bool indentLetBody = false;
};

/// Run the ExportSMTLIB pass.
LogicalResult
exportSMTLIB(Operation *module, llvm::raw_ostream &os,
             const SMTEmissionOptions &options = SMTEmissionOptions());

/// Register the ExportSMTLIB pass.
void registerExportSMTLIBTranslation();

} // namespace smt
} // namespace mlir

#endif // MLIR_TARGET_EXPORTSMTLIB_H
