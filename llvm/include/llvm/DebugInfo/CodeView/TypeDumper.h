//===-- TypeDumper.h - CodeView type info dumper ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEDUMPER_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEDUMPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeStream.h"

namespace llvm {
class ScopedPrinter;

namespace codeview {

/// Dumper for CodeView type streams found in COFF object files and PDB files.
class CVTypeDumper {
public:
  CVTypeDumper(ScopedPrinter &W, bool PrintRecordBytes)
      : W(&W), PrintRecordBytes(PrintRecordBytes) {}

  StringRef getTypeName(TypeIndex TI);
  void printTypeIndex(StringRef FieldName, TypeIndex TI);

  /// Dumps one type record.  Returns false if there was a type parsing error,
  /// and true otherwise.  This should be called in order, since the dumper
  /// maintains state about previous records which are necessary for cross
  /// type references.
  bool dump(const TypeIterator::Record &Record);

  /// Dumps the type records in Data. Returns false if there was a type stream
  /// parse error, and true otherwise.
  bool dump(ArrayRef<uint8_t> Data);

  /// Gets the type index for the next type record.
  unsigned getNextTypeIndex() const {
    return 0x1000 + CVUDTNames.size();
  }

  /// Records the name of a type, and reserves its type index.
  void recordType(StringRef Name) { CVUDTNames.push_back(Name); }

  /// Saves the name in a StringSet and creates a stable StringRef.
  StringRef saveName(StringRef TypeName) {
    return TypeNames.insert(TypeName).first->getKey();
  }

  void setPrinter(ScopedPrinter *P);
  ScopedPrinter *getPrinter() { return W; }

private:
  ScopedPrinter *W;

  bool PrintRecordBytes = false;

  /// All user defined type records in .debug$T live in here. Type indices
  /// greater than 0x1000 are user defined. Subtract 0x1000 from the index to
  /// index into this vector.
  SmallVector<StringRef, 10> CVUDTNames;

  StringSet<> TypeNames;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPEDUMPER_H
