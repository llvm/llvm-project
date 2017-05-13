//===-- TypeDatabaseVisitor.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEDATABASEVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEDATABASEVISITOR_H

#include "llvm/ADT/PointerUnion.h"

#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

namespace llvm {
namespace codeview {

/// Dumper for CodeView type streams found in COFF object files and PDB files.
class TypeDatabaseVisitor : public TypeVisitorCallbacks {
public:
  explicit TypeDatabaseVisitor(TypeDatabase &TypeDB) : TypeDB(&TypeDB) {}

  /// Paired begin/end actions for all types. Receives all record data,
  /// including the fixed-length record prefix.
  Error visitTypeBegin(CVType &Record) override;
  Error visitTypeBegin(CVType &Record, TypeIndex Index) override;
  Error visitTypeEnd(CVType &Record) override;
  Error visitMemberBegin(CVMemberRecord &Record) override;
  Error visitMemberEnd(CVMemberRecord &Record) override;

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(CVType &CVR, Name##Record &Record) override;
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownMember(CVMemberRecord &CVR, Name##Record &Record) override;
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "TypeRecords.def"

private:
  StringRef getTypeName(TypeIndex Index) const;
  StringRef saveTypeName(StringRef Name);

  bool IsInFieldList = false;

  /// Name of the current type. Only valid before visitTypeEnd.
  StringRef Name;
  /// Current type index.  Only valid before visitTypeEnd, and if we are
  /// visiting a random access type database.
  Optional<TypeIndex> CurrentTypeIndex;

  TypeDatabase *TypeDB;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPEDUMPER_H
