//===- BytecodeDialectGen.h - Dialect bytecode read/writer gen -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_BYTECODEDIALECTGEN_H
#define MLIR_TABLEGEN_GENERATORS_BYTECODEDIALECTGEN_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit bytecode dialect readers/writers. If dialectName is non-empty,
/// only emit code for that dialect.
bool emitBytecodeDialect(const llvm::RecordKeeper &records,
                         llvm::StringRef dialectName, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_BYTECODEDIALECTGEN_H
