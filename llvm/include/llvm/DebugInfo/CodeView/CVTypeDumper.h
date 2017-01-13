//===-- CVTypeDumper.h - CodeView type info dumper --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CVTYPEDUMPER_H
#define LLVM_DEBUGINFO_CODEVIEW_CVTYPEDUMPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {

namespace codeview {

/// Dumper for CodeView type streams found in COFF object files and PDB files.
class CVTypeDumper {
public:
  explicit CVTypeDumper(TypeDatabase &TypeDB) : TypeDB(TypeDB) {}

  /// Dumps one type record.  Returns false if there was a type parsing error,
  /// and true otherwise.  This should be called in order, since the dumper
  /// maintains state about previous records which are necessary for cross
  /// type references.
  Error dump(const CVType &Record, TypeVisitorCallbacks &Dumper);

  /// Dumps the type records in Types. Returns false if there was a type stream
  /// parse error, and true otherwise.
  Error dump(const CVTypeArray &Types, TypeVisitorCallbacks &Dumper);

  /// Dumps the type records in Data. Returns false if there was a type stream
  /// parse error, and true otherwise. Use this method instead of the
  /// CVTypeArray overload when type records are laid out contiguously in
  /// memory.
  Error dump(ArrayRef<uint8_t> Data, TypeVisitorCallbacks &Dumper);

  static void printTypeIndex(ScopedPrinter &Printer, StringRef FieldName,
                             TypeIndex TI, TypeDatabase &DB);

private:
  TypeDatabase &TypeDB;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPEDUMPER_H
