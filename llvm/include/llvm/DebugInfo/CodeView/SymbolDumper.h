//===-- SymbolDumper.h - CodeView symbol info dumper ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_SYMBOLDUMPER_H
#define LLVM_DEBUGINFO_CODEVIEW_SYMBOLDUMPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/CodeView/SymbolDumpDelegate.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"

namespace llvm {
class ScopedPrinter;

namespace codeview {
class TypeDatabase;

/// Dumper for CodeView symbol streams found in COFF object files and PDB files.
class CVSymbolDumper {
public:
  CVSymbolDumper(ScopedPrinter &W, TypeDatabase &TypeDB,
                 std::unique_ptr<SymbolDumpDelegate> ObjDelegate,
                 bool PrintRecordBytes)
      : W(W), TypeDB(TypeDB), ObjDelegate(std::move(ObjDelegate)),
        PrintRecordBytes(PrintRecordBytes) {}

  /// Dumps one type record.  Returns false if there was a type parsing error,
  /// and true otherwise.  This should be called in order, since the dumper
  /// maintains state about previous records which are necessary for cross
  /// type references.
  Error dump(CVRecord<SymbolKind> &Record);

  /// Dumps the type records in Data. Returns false if there was a type stream
  /// parse error, and true otherwise.
  Error dump(const CVSymbolArray &Symbols);

private:
  ScopedPrinter &W;
  TypeDatabase &TypeDB;
  std::unique_ptr<SymbolDumpDelegate> ObjDelegate;

  bool PrintRecordBytes;
};
} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_SYMBOLDUMPER_H
