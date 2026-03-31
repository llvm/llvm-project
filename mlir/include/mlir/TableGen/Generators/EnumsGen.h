//===- EnumsGen.h - Enum utility generator ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for generating enum utility declarations and
// definitions from TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_ENUMSGEN_H
#define MLIR_TABLEGEN_GENERATORS_ENUMSGEN_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit declarations for a single enum defined by enumDef.
void emitEnumDecl(const llvm::Record &enumDef, llvm::raw_ostream &os);

/// Emit declarations for all enums in records.
bool emitEnumDecls(const llvm::RecordKeeper &records, llvm::raw_ostream &os);

/// Emit definitions for a single enum defined by enumDef.
void emitEnumDef(const llvm::Record &enumDef, llvm::raw_ostream &os);

/// Emit definitions for all enums in records.
bool emitEnumDefs(const llvm::RecordKeeper &records, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_ENUMSGEN_H
