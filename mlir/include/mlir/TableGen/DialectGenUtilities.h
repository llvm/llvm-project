//===- DialectGenUtilities.h - Utilities for dialect generation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_DIALECTGENUTILITIES_H_
#define MLIR_TOOLS_MLIRTBLGEN_DIALECTGENUTILITIES_H_

#include "mlir/Support/LLVM.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

namespace mlir {
namespace tblgen {
class Dialect;

/// Find the dialect selected by the user to generate for. Returns std::nullopt
/// if no dialect was found, or if more than one potential dialect was found.
std::optional<Dialect>
findDialectToGenerate(ArrayRef<Dialect> dialects,
                      const std::string &selectedDialect);
bool emitDialectDecls(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                      const std::string &selectedDialect);
bool emitDialectDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                     const std::string &selectedDialect);
bool emitDirectiveDecls(const llvm::RecordKeeper &recordKeeper,
                        llvm::StringRef dialect, raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_DIALECTGENUTILITIES_H_
