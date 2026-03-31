//===- DialectGen.h - MLIR dialect C++ generation utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for generating C++ definitions for dialects from
// TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_DIALECTGEN_H
#define MLIR_TABLEGEN_GENERATORS_DIALECTGEN_H

#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"
#include <optional>
#include <string>
#include <utility>

namespace llvm {
class RecordKeeper;
class raw_ostream;
class DagInit;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Populate discardableAttributes from the given DagInit of discardable
/// attribute descriptors.
void populateDiscardableAttributes(
    Dialect &dialect, const llvm::DagInit *discardableAttrDag,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>>
        &discardableAttributes);

/// Find the dialect to generate from dialects. If selectedDialect is
/// empty, the dialect is auto-detected (succeeds only when exactly one dialect
/// is present). Returns std::nullopt and prints an error on failure.
std::optional<Dialect> findDialectToGenerate(llvm::ArrayRef<Dialect> dialects,
                                             llvm::StringRef selectedDialect);

/// Emit the C++ class declaration for dialect.
void emitDialectDecl(Dialect &dialect, llvm::raw_ostream &os);

/// Emit the C++ class declarations for all dialects in records, selecting
/// the one identified by selectedDialect.
bool emitDialectDecls(const llvm::RecordKeeper &records,
                      llvm::StringRef selectedDialect, llvm::raw_ostream &os);

/// Emit the C++ constructor and destructor definitions for dialect.
void emitDialectDef(Dialect &dialect, const llvm::RecordKeeper &records,
                    llvm::raw_ostream &os);

/// Emit the C++ constructor and destructor definitions for all dialects in
/// records, selecting the one identified by selectedDialect.
bool emitDialectDefs(const llvm::RecordKeeper &records,
                     llvm::StringRef selectedDialect, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_DIALECTGEN_H
